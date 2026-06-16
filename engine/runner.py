"""
BenchmarkRunner — điều phối chạy đánh giá song song (async) cho toàn bộ golden set.

Mỗi test case đi qua pipeline:
  Agent.query -> Retrieval metrics -> RAGAS metrics -> Multi-Judge consensus.

Tối ưu hiệu năng: chạy nhiều case đồng thời qua asyncio.Semaphore (giới hạn để
không bị rate-limit). Thu thập latency, token & chi phí (qua usage_ledger) để báo cáo.
"""
import os
import time
import asyncio
from typing import List, Dict

from engine.llm_client import usage_ledger
from engine.llm_judge import MultiModelJudge


class BenchmarkRunner:
    def __init__(self, agent, retrieval_eval, rag_eval, judge, max_concurrency: int = 8):
        self.agent = agent
        self.retrieval_eval = retrieval_eval
        self.rag_eval = rag_eval
        self.judge = judge
        self.max_concurrency = max_concurrency

    async def run_single_test(self, case: Dict) -> Dict:
        start = time.perf_counter()

        # 1) Gọi Agent (retrieval + generation)
        response = await self.agent.query(case["question"])
        latency = time.perf_counter() - start

        expected_ids = case.get("expected_retrieval_ids", [])

        # 2-3-4) Chấm song song: retrieval (đồng bộ, nhanh) + RAGAS + Multi-Judge
        retrieval = self.retrieval_eval.score_single(expected_ids, response.get("retrieved_ids", []))
        ragas_task = self.rag_eval.score(case, response)
        judge_task = self.judge.evaluate_multi_judge(
            case["question"], response["answer"], case["expected_answer"]
        )
        ragas, judge = await asyncio.gather(ragas_task, judge_task)

        status = "pass" if judge["final_score"] >= 3.5 else "fail"
        return {
            "id": case.get("id"),
            "question": case["question"],
            "category": case.get("category"),
            "difficulty": case.get("metadata", {}).get("difficulty"),
            "type": case.get("metadata", {}).get("type"),
            "expected_answer": case["expected_answer"],
            "agent_answer": response["answer"],
            "expected_retrieval_ids": expected_ids,
            "retrieved_ids": response.get("retrieved_ids", []),
            "retrieval": retrieval,
            "ragas": ragas,
            "judge": judge,
            "latency_sec": round(latency, 3),
            "tokens": {
                "prompt": response["metadata"]["prompt_tokens"],
                "completion": response["metadata"]["completion_tokens"],
            },
            "status": status,
        }

    async def run_all(self, dataset: List[Dict]) -> Dict:
        """Chạy toàn bộ dataset song song. Trả {results, aggregate}."""
        usage_ledger.reset()  # cô lập chi phí của lần chạy này
        wall_start = time.perf_counter()

        sem = asyncio.Semaphore(self.max_concurrency)
        done = {"n": 0}
        total = len(dataset)

        async def _guarded(case):
            async with sem:
                try:
                    # Timeout tổng/case để không kẹt vô hạn dù bất kỳ lý do gì.
                    return await asyncio.wait_for(
                        self.run_single_test(case),
                        timeout=float(os.getenv("CASE_TIMEOUT", "60")),
                    )
                except Exception as e:
                    # Cô lập lỗi: 1 case hỏng (rate-limit/timeout...) không làm sập cả benchmark.
                    print(f"⚠️  Case {case.get('id')} lỗi: {type(e).__name__} {str(e)[:60]}", flush=True)
                    return None
                finally:
                    done["n"] += 1
                    if done["n"] % 10 == 0 or done["n"] == total:
                        print(f"   ... {done['n']}/{total} cases", flush=True)

        raw = await asyncio.gather(*[_guarded(c) for c in dataset])
        results = [r for r in raw if r is not None]
        wall_time = time.perf_counter() - wall_start
        if len(results) < len(dataset):
            print(f"⚠️  {len(dataset) - len(results)}/{len(dataset)} case bị bỏ do lỗi.")

        aggregate = self._aggregate(results, wall_time)
        return {"results": results, "aggregate": aggregate}

    def _aggregate(self, results: List[Dict], wall_time: float) -> Dict:
        n = len(results)
        if n == 0:
            return {}

        # Retrieval (chỉ tính case có chấm)
        ret_metrics = self.retrieval_eval.aggregate([r["retrieval"] for r in results])

        # Cohen's Kappa giữa 2 judge (điểm độc lập vòng 1)
        labels_a, labels_b = [], []
        for r in results:
            scores = list(r["judge"]["individual_scores"].values())
            if len(scores) >= 2:
                labels_a.append(scores[0])
                labels_b.append(scores[1])
        kappa = MultiModelJudge.cohens_kappa(labels_a, labels_b)

        passed = sum(1 for r in results if r["status"] == "pass")
        cost = usage_ledger.report()

        return {
            "total": n,
            "passed": passed,
            "failed": n - passed,
            "pass_rate": round(passed / n, 4),
            "avg_score": round(sum(r["judge"]["final_score"] for r in results) / n, 4),
            "avg_faithfulness": round(sum(r["ragas"]["faithfulness"] for r in results) / n, 4),
            "avg_relevancy": round(sum(r["ragas"]["relevancy"] for r in results) / n, 4),
            "hit_rate": round(ret_metrics["avg_hit_rate"], 4),
            "mrr": round(ret_metrics["avg_mrr"], 4),
            "retrieval_scored_cases": ret_metrics["num_scored"],
            "agreement_rate": round(sum(r["judge"]["agreement_rate"] for r in results) / n, 4),
            "cohens_kappa": kappa,
            "num_conflicts": sum(1 for r in results if r["judge"]["conflict"]),
            "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / n, 3),
            "wall_time_sec": round(wall_time, 2),
            "cost": cost,
            "cost_per_eval_usd": round(cost["total_cost_usd"] / n, 6),
        }
