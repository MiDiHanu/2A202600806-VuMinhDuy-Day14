"""
main.py — Điều phối toàn bộ Benchmark (Expert Level).

Quy trình:
  1. Nạp Golden Dataset.
  2. Chạy Benchmark cho Agent V1 (Base) và V2 (Optimized) — bất đồng bộ.
  3. Regression: so sánh V1 vs V2 (Delta Analysis).
  4. Release Gate: tự động quyết định APPROVE / BLOCK theo Chất lượng + Chi phí + Hiệu năng.
  5. Audit Position Bias của giám khảo.
  6. Xuất reports/summary.json & reports/benchmark_results.json.

Chạy: python main.py
"""
import io
import os
import sys
import json
import time
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from agent.main_agent import MainAgent
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.ragas_metrics import RagEvaluator
from engine.llm_judge import MultiModelJudge

GOLDEN_PATH = "data/golden_set.jsonl"


def load_dataset() -> list:
    if not os.path.exists(GOLDEN_PATH):
        print(f"❌ Thiếu {GOLDEN_PATH}. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_version(version: str, dataset: list) -> dict:
    print(f"\n🚀 Benchmark {version} trên {len(dataset)} cases...")
    agent = MainAgent(version=version)
    runner = BenchmarkRunner(
        agent=agent,
        retrieval_eval=RetrievalEvaluator(top_k=agent.cfg["top_k"]),
        rag_eval=RagEvaluator(),
        judge=MultiModelJudge(),
        max_concurrency=int(os.getenv("BENCH_CONCURRENCY", "8")),
    )
    out = await runner.run_all(dataset)
    agg = out["aggregate"]
    print(
        f"   ✓ {version}: score={agg['avg_score']} | pass={agg['pass_rate']*100:.0f}% | "
        f"hit_rate={agg['hit_rate']} | mrr={agg['mrr']} | "
        f"agree={agg['agreement_rate']} | kappa={agg['cohens_kappa']} | "
        f"cost=${agg['cost']['total_cost_usd']} | {agg['wall_time_sec']}s"
    )
    return out


async def audit_position_bias(dataset: list, results: list, sample: int = 6) -> dict:
    """Kiểm tra position bias của giám khảo trên một mẫu (agent answer vs ground truth)."""
    judge = MultiModelJudge()
    by_id = {c["id"]: c for c in dataset}
    # Ưu tiên các case 'pass' để 2 ứng viên đều hợp lý (so kè khó hơn).
    picks = [r for r in results if r["status"] == "pass"][:sample]
    checks = []
    for r in picks:
        gt = by_id.get(r["id"], {}).get("expected_answer", "")
        res = await judge.check_position_bias(r["question"], r["agent_answer"], gt)
        checks.append(res)
    biased = sum(1 for c in checks if c.get("position_bias"))
    n = len(checks) or 1
    return {
        "sampled": len(checks),
        "biased_count": biased,
        "position_bias_rate": round(biased / n, 3),
        "details": checks,
    }


def release_gate(v1: dict, v2: dict) -> dict:
    """
    Cổng phát hành tự động dựa trên 3 trụ cột: Chất lượng, Chi phí, Hiệu năng.
    APPROVE chỉ khi không có hồi quy nghiêm trọng và đạt ngân sách.
    """
    checks = []

    def add(name, passed, detail):
        checks.append({"criterion": name, "passed": bool(passed), "detail": detail})

    delta_score = round(v2["avg_score"] - v1["avg_score"], 4)
    delta_hit = round(v2["hit_rate"] - v1["hit_rate"], 4)
    delta_faith = round(v2["avg_faithfulness"] - v1["avg_faithfulness"], 4)
    cost_ratio = (v2["cost"]["total_cost_usd"] / v1["cost"]["total_cost_usd"]) if v1["cost"]["total_cost_usd"] else 1.0

    # 1) Chất lượng: không được tụt điểm tổng thể.
    add("quality_no_regression", delta_score >= -0.05, f"Δavg_score = {delta_score:+}")
    # 2) Faithfulness (chống hallucination) không tụt.
    add("faithfulness_no_regression", delta_faith >= -0.05, f"Δfaithfulness = {delta_faith:+}")
    # 3) Retrieval không tụt.
    add("retrieval_no_regression", delta_hit >= -0.05, f"Δhit_rate = {delta_hit:+}")
    # 4) Độ tin cậy giám khảo đủ cao (Kappa >= 0.2 ~ fair trở lên).
    add("judge_reliability", v2["cohens_kappa"] >= 0.2, f"Cohen's Kappa = {v2['cohens_kappa']}")
    # 5) Chi phí: không phình quá 50% so với V1.
    add("cost_budget", cost_ratio <= 1.5, f"cost_ratio = {cost_ratio:.2f}x")
    # 6) Hiệu năng: < 120s cho ~50 case.
    perf_budget = 120 * max(1, v2["total"] / 50)
    add("performance_budget", v2["wall_time_sec"] <= perf_budget, f"{v2['wall_time_sec']}s <= {perf_budget:.0f}s")

    hard_fail = any(not c["passed"] for c in checks if c["criterion"] in (
        "quality_no_regression", "faithfulness_no_regression"))
    all_pass = all(c["passed"] for c in checks)
    improved = delta_score > 0

    if hard_fail:
        decision = "BLOCK_RELEASE"
    elif all_pass and improved:
        decision = "APPROVE"
    elif all_pass:
        decision = "APPROVE_NO_IMPROVEMENT"
    else:
        decision = "REVIEW_REQUIRED"

    return {
        "decision": decision,
        "delta": {"avg_score": delta_score, "hit_rate": delta_hit, "faithfulness": delta_faith, "cost_ratio": round(cost_ratio, 3)},
        "checks": checks,
    }


async def main():
    dataset = load_dataset()
    if not dataset:
        return

    t0 = time.perf_counter()
    v1_out = await run_version("V1_Base", dataset)
    v2_out = await run_version("V2_Optimized", dataset)
    v1, v2 = v1_out["aggregate"], v2_out["aggregate"]

    # Regression + Gate
    gate = release_gate(v1, v2)
    bias = await audit_position_bias(dataset, v2_out["results"])

    print("\n📊 --- REGRESSION (V1 Base -> V2 Optimized) ---")
    print(f"   avg_score : {v1['avg_score']} -> {v2['avg_score']}  (Δ {gate['delta']['avg_score']:+})")
    print(f"   hit_rate  : {v1['hit_rate']} -> {v2['hit_rate']}  (Δ {gate['delta']['hit_rate']:+})")
    print(f"   faithful. : {v1['avg_faithfulness']} -> {v2['avg_faithfulness']}  (Δ {gate['delta']['faithfulness']:+})")
    print(f"   cost ratio: {gate['delta']['cost_ratio']}x")
    print(f"   position_bias_rate (judge): {bias['position_bias_rate']}")

    print("\n🚦 --- RELEASE GATE ---")
    for c in gate["checks"]:
        print(f"   {'✅' if c['passed'] else '❌'} {c['criterion']}: {c['detail']}")
    print(f"\n   👉 QUYẾT ĐỊNH: {gate['decision']}")

    # ----- Xuất reports -----
    summary = {
        "metadata": {
            "version": "V2_Optimized",
            "total": v2["total"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_models": list(v2["cost"]["per_model"].keys()),
            "mode": "ONLINE" if not MainAgent().llm.offline else "OFFLINE",
        },
        # check_lab.py yêu cầu: avg_score, hit_rate, agreement_rate.
        "metrics": {
            "avg_score": v2["avg_score"],
            "pass_rate": v2["pass_rate"],
            "hit_rate": v2["hit_rate"],
            "mrr": v2["mrr"],
            "avg_faithfulness": v2["avg_faithfulness"],
            "avg_relevancy": v2["avg_relevancy"],
            "agreement_rate": v2["agreement_rate"],
            "cohens_kappa": v2["cohens_kappa"],
            "num_conflicts": v2["num_conflicts"],
            "position_bias_rate": bias["position_bias_rate"],
            "avg_latency_sec": v2["avg_latency_sec"],
            "wall_time_sec": v2["wall_time_sec"],
            "total_cost_usd": v2["cost"]["total_cost_usd"],
            "cost_per_eval_usd": v2["cost_per_eval_usd"],
        },
        "cost_breakdown": v2["cost"],
        "regression": {
            "v1": _slim(v1),
            "v2": _slim(v2),
            "gate": gate,
        },
        "position_bias_audit": bias,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_out["results"], f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã ghi reports/summary.json & reports/benchmark_results.json")
    print(f"⏱️  Tổng thời gian: {time.perf_counter() - t0:.1f}s")


def _slim(agg: dict) -> dict:
    """Trích các chỉ số chính của một version cho mục regression."""
    keys = ["avg_score", "pass_rate", "hit_rate", "mrr", "avg_faithfulness",
            "avg_relevancy", "agreement_rate", "cohens_kappa", "avg_latency_sec", "wall_time_sec"]
    out = {k: agg[k] for k in keys}
    out["total_cost_usd"] = agg["cost"]["total_cost_usd"]
    out["cost_per_eval_usd"] = agg["cost_per_eval_usd"]
    return out


if __name__ == "__main__":
    asyncio.run(main())
