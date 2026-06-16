"""
Tự động sinh báo cáo Phân tích Thất bại (analysis/failure_analysis.md) TỪ DỮ LIỆU THẬT
trong reports/. Bao gồm: thống kê tổng quan, phân cụm lỗi (failure clustering),
phân tích 5-Whys cho 3 case tệ nhất, và kế hoạch cải tiến.

Chạy (sau khi đã có reports):
    python analysis/generate_analysis.py            # scaffold deterministic
    python analysis/generate_analysis.py --llm      # nhờ Gemini viết 5-Whys sâu hơn

Lưu ý: đây là BẢN NHÁP dựa trên số liệu. Nhóm nên đọc lại, bổ sung nhận định
chuyên môn (đặc biệt phần Root Cause) trước khi nộp.
"""
import io
import os
import sys
import json
import asyncio
from collections import Counter
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SUMMARY_PATH = "reports/summary.json"
RESULTS_PATH = "reports/benchmark_results.json"
OUTPUT_PATH = "analysis/failure_analysis.md"

ADVERSARIAL_TYPES = {
    "out-of-context", "prompt-injection", "goal-hijacking",
    "ambiguous", "conflicting", "negation", "robustness",
}


def classify_failure(r: Dict) -> str:
    """Gán nhãn nguyên nhân lỗi cho 1 case fail (ưu tiên nguyên nhân gốc nhất)."""
    ret = r["retrieval"]
    faith = r["ragas"]["faithfulness"]
    rel = r["ragas"]["relevancy"]
    typ = r.get("type")

    if ret.get("scored") and ret.get("hit_rate") == 0:
        return "Retrieval Miss (lấy sai/không lấy được tài liệu)"
    if faith is not None and faith < 0.5:
        return "Hallucination (trả lời không bám tài liệu)"
    if typ in ADVERSARIAL_TYPES:
        return "Red-Team Failure (không xử lý đúng case tấn công/biên)"
    if rel is not None and rel < 0.5:
        return "Off-topic / Incomplete (lạc đề hoặc thiếu ý)"
    if r["judge"].get("conflict"):
        return "Judge Conflict (giám khảo bất đồng, chất lượng mơ hồ)"
    return "Other (điểm thấp, nguyên nhân khác)"


def five_whys_scaffold(r: Dict) -> List[str]:
    """Sinh khung 5-Whys deterministic dựa trên tín hiệu định lượng của case."""
    ret = r["retrieval"]
    faith = r["ragas"]["faithfulness"]
    whys = []
    whys.append(f"**Symptom:** Điểm giám khảo {r['judge']['final_score']}/5 "
                f"(accuracy={r['judge']['criteria'].get('accuracy')}, safety={r['judge']['criteria'].get('safety')}). "
                f"Nhận xét: {r['judge'].get('reasoning', '')[:160]}")
    if ret.get("scored") and ret.get("hit_rate") == 0:
        whys.append("**Why 1:** Câu trả lời sai vì LLM không có context đúng để dựa vào.")
        whys.append(f"**Why 2:** Retrieval miss — tài liệu kỳ vọng {r['expected_retrieval_ids']} "
                    f"không nằm trong top-k đã lấy {r['retrieved_ids']}.")
        whys.append("**Why 3:** Bộ truy hồi (TF-IDF lexical) không khớp được do khác biệt từ vựng "
                    "giữa câu hỏi và tài liệu (synonym / diễn đạt khác).")
        whys.append("**Why 4:** Chiến lược chunking/indexing chưa tối ưu cho truy vấn ngữ nghĩa.")
        whys.append("**Root Cause:** Tầng Retrieval lexical thiếu khả năng hiểu ngữ nghĩa "
                    "→ cần embedding-based retrieval hoặc reranking.")
    elif faith is not None and faith < 0.5:
        whys.append("**Why 1:** Câu trả lời chứa thông tin không có trong context (bịa).")
        whys.append("**Why 2:** Model ưu tiên 'trả lời cho có' thay vì bám tài liệu.")
        whys.append("**Why 3:** System prompt chưa đủ mạnh để ép grounding / từ chối khi thiếu thông tin.")
        whys.append("**Why 4:** Không có bước kiểm tra faithfulness trước khi trả lời.")
        whys.append("**Root Cause:** Thiếu ràng buộc grounding ở tầng Prompting → cần prompt "
                    "chống hallucination + guardrail hậu kiểm.")
    elif r.get("type") in ADVERSARIAL_TYPES:
        whys.append(f"**Why 1:** Case tấn công/biên loại '{r.get('type')}' không được xử lý đúng.")
        whys.append("**Why 2:** Agent không nhận diện được ý đồ (injection/hijack) hoặc sự mơ hồ.")
        whys.append("**Why 3:** System prompt thiếu quy tắc an toàn rõ ràng cho tình huống này.")
        whys.append("**Why 4:** Bộ test red-team chưa được dùng để tinh chỉnh prompt trước đó.")
        whys.append("**Root Cause:** Lỗ hổng an toàn ở tầng Prompting/Policy → cần bổ sung "
                    "guardrail và few-shot cho các mẫu tấn công.")
    else:
        whys.append("**Why 1:** Câu trả lời chưa đạt kỳ vọng về độ đầy đủ/đúng trọng tâm.")
        whys.append("**Why 2:** Context có thể đúng nhưng generation chưa khai thác hết.")
        whys.append("**Why 3:** Prompt chưa yêu cầu rõ mức độ chi tiết cần thiết.")
        whys.append("**Why 4:** Thiếu ví dụ mẫu (few-shot) định hướng định dạng trả lời.")
        whys.append("**Root Cause:** Tinh chỉnh ở tầng Prompting (độ chi tiết + few-shot).")
    return whys


async def llm_five_whys(r: Dict) -> str:
    """(Tùy chọn) nhờ Gemini viết 5-Whys mạch lạc hơn dựa trên dữ liệu case."""
    from engine.llm_client import LLMClient
    client = LLMClient(model="gemini-2.5-flash", temperature=0.3)
    if client.offline:
        return ""
    sys_p = "Bạn là kỹ sư AI phân tích nguyên nhân gốc rễ (Root Cause Analysis). Viết tiếng Việt, súc tích."
    user_p = (
        "Phân tích 5-Whys cho ca lỗi sau của một RAG agent. Chỉ ra lỗi nằm ở tầng nào "
        "(Ingestion / Chunking / Retrieval / Prompting / Generation).\n\n"
        f"Câu hỏi: {r['question']}\n"
        f"Đáp án chuẩn: {r['expected_answer']}\n"
        f"Agent trả lời: {r['agent_answer']}\n"
        f"Retrieval: expected={r['expected_retrieval_ids']} retrieved={r['retrieved_ids']} "
        f"hit_rate={r['retrieval'].get('hit_rate')}\n"
        f"Faithfulness={r['ragas']['faithfulness']} Relevancy={r['ragas']['relevancy']} "
        f"JudgeScore={r['judge']['final_score']}\n\n"
        "Viết theo dạng: Symptom, Why 1..4, Root Cause (mỗi dòng 1 ý ngắn)."
    )
    text, _ = await client.complete(sys_p, user_p)
    return text.strip()


def build_report(summary: Dict, results: List[Dict], llm_sections: Dict[str, str]) -> str:
    m = summary["metrics"]
    meta = summary["metadata"]
    total = len(results)
    failed = [r for r in results if r["status"] == "fail"]
    passed = total - len(failed)

    # Phân cụm lỗi
    clusters = Counter(classify_failure(r) for r in failed)

    # 3 case tệ nhất (điểm thấp nhất, ưu tiên faithfulness thấp)
    worst = sorted(results, key=lambda r: (r["judge"]["final_score"], r["ragas"]["faithfulness"]))[:3]

    lines = []
    lines.append("# Báo cáo Phân tích Thất bại (Failure Analysis Report)")
    lines.append("")
    lines.append(f"> Tự động sinh từ `reports/` bởi `analysis/generate_analysis.py` — "
                 f"phiên bản **{meta.get('version')}**, {meta.get('timestamp')}.")
    lines.append("")
    # 1. Tổng quan
    lines.append("## 1. Tổng quan Benchmark")
    lines.append(f"- **Tổng số cases:** {total}")
    lines.append(f"- **Pass / Fail:** {passed} / {len(failed)}  (pass rate {m.get('pass_rate', 0)*100:.1f}%)")
    lines.append(f"- **Điểm RAGAS trung bình:**")
    lines.append(f"    - Faithfulness: {m.get('avg_faithfulness'):.3f}")
    lines.append(f"    - Answer Relevancy: {m.get('avg_relevancy'):.3f}")
    lines.append(f"- **Retrieval:** Hit Rate {m.get('hit_rate')*100:.1f}% · MRR {m.get('mrr'):.3f}")
    lines.append(f"- **LLM-Judge trung bình:** {m.get('avg_score'):.2f} / 5.0")
    lines.append(f"- **Độ tin cậy giám khảo:** Agreement {m.get('agreement_rate')*100:.1f}% · "
                 f"Cohen's Kappa {m.get('cohens_kappa')} · {m.get('num_conflicts')} ca xung đột")
    lines.append(f"- **Position Bias (audit):** {m.get('position_bias_rate')*100:.1f}%")
    lines.append(f"- **Hiệu năng & Chi phí:** {m.get('wall_time_sec')}s tổng · "
                 f"${m.get('cost_per_eval_usd')}/eval · tổng ${m.get('total_cost_usd')}")
    lines.append("")
    # 2. Phân cụm lỗi
    lines.append("## 2. Phân nhóm lỗi (Failure Clustering)")
    if clusters:
        lines.append("| Nhóm lỗi | Số lượng | Tỉ lệ trên tổng fail |")
        lines.append("|----------|:--------:|:--------------------:|")
        for name, cnt in clusters.most_common():
            lines.append(f"| {name} | {cnt} | {cnt/len(failed)*100:.0f}% |")
    else:
        lines.append("_Không có case fail — agent vượt qua toàn bộ test._")
    lines.append("")
    # phân bố fail theo loại câu hỏi
    if failed:
        by_type = Counter(r.get("type") for r in failed)
        lines.append("**Fail theo loại câu hỏi:** " + ", ".join(f"`{t}`×{c}" for t, c in by_type.most_common()))
        lines.append("")
    # 3. 5 Whys
    lines.append("## 3. Phân tích 5 Whys (3 case tệ nhất)")
    lines.append("")
    for i, r in enumerate(worst, 1):
        lines.append(f"### Case #{i}: `{r['id']}` — {r.get('type')} (điểm {r['judge']['final_score']}/5)")
        lines.append(f"- **Câu hỏi:** {r['question']}")
        lines.append(f"- **Agent trả lời:** {r['agent_answer'][:200]}")
        lines.append(f"- **Đáp án chuẩn:** {r['expected_answer'][:160]}")
        lines.append("")
        if llm_sections.get(r["id"]):
            lines.append(llm_sections[r["id"]])
        else:
            for w in five_whys_scaffold(r):
                lines.append(f"{w}  ")
        lines.append("")
    # 3.5 Hiệu năng & Chi phí + đề xuất tối ưu (bonus)
    cost = summary.get("cost_breakdown", {})
    reg = summary.get("regression", {})
    lines.append("## 4. Hiệu năng & Chi phí (Performance & Cost)")
    lines.append(f"- **Thời gian:** {m.get('wall_time_sec')}s cho {total} case "
                 f"(trung bình {m.get('avg_latency_sec')}s/case) — chạy song song async.")
    lines.append(f"- **Token & chi phí:** {cost.get('total_tokens', 0):,} tokens · "
                 f"{cost.get('total_calls', 0)} calls · tổng **${m.get('total_cost_usd')}** "
                 f"(**${m.get('cost_per_eval_usd')}/eval**).")
    if cost.get("per_model"):
        lines.append("- **Theo model:**")
        for mdl, v in cost["per_model"].items():
            lines.append(f"    - `{mdl}`: {v['calls']} calls · ${v['cost_usd']}")
    lines.append("")
    lines.append("**Đề xuất giảm ~30% chi phí eval (không giảm độ chính xác) — Judge Cascade:**")
    lines.append("1. Vòng 1 chỉ chạy **1 judge rẻ** (`flash-lite`) cho mọi case.")
    lines.append("2. Nếu điểm rơi vào vùng rõ ràng (1–2 hoặc 5) và faithfulness cao → **chốt luôn**, bỏ qua judge 2.")
    lines.append("3. Chỉ **escalate** sang judge 2 + debate cho case điểm lưng chừng (3–4) hoặc nghi xung đột.")
    lines.append("Do phân bố điểm lệch mạnh về 4–5, ước tính >40% case chỉ cần 1 call → giảm ~30–40% chi phí giám khảo. "
                 "Bổ sung **caching** cho câu hỏi trùng để giảm thêm.")
    lines.append("")

    # 5. Action plan
    lines.append("## 5. Kế hoạch cải tiến (Action Plan)")
    top_cluster = clusters.most_common(1)[0][0] if clusters else None
    generic = [
        "Bổ sung **embedding-based retrieval + reranking** thay cho lexical thuần để giảm Retrieval Miss.",
        "Tăng cường **system prompt chống hallucination** (chỉ trả lời theo context, từ chối khi thiếu).",
        "Thêm **few-shot guardrail** cho các mẫu red-team (prompt injection, out-of-context).",
        "Thử nghiệm **chunking ngữ nghĩa** (semantic chunking) thay cho fixed-size.",
        "Mở rộng golden set ở các loại câu hỏi đang fail nhiều để tinh chỉnh có mục tiêu.",
    ]
    lines.append(f"_Ưu tiên dựa trên nhóm lỗi lớn nhất: **{top_cluster or 'N/A'}**._")
    lines.append("")
    for g in generic:
        lines.append(f"- [ ] {g}")
    lines.append("")
    lines.append("---")
    lines.append("_Bản nháp tự động — nhóm cần rà soát & bổ sung nhận định chuyên môn trước khi nộp._")
    return "\n".join(lines)


async def main():
    use_llm = "--llm" in sys.argv
    if not (os.path.exists(SUMMARY_PATH) and os.path.exists(RESULTS_PATH)):
        print("❌ Chưa có reports/. Hãy chạy 'python main.py' trước.")
        return
    summary = json.load(open(SUMMARY_PATH, encoding="utf-8"))
    results = json.load(open(RESULTS_PATH, encoding="utf-8"))

    llm_sections: Dict[str, str] = {}
    if use_llm:
        worst = sorted(results, key=lambda r: (r["judge"]["final_score"], r["ragas"]["faithfulness"]))[:3]
        print("🤖 Đang nhờ Gemini viết 5-Whys cho 3 case tệ nhất...")
        for r in worst:
            llm_sections[r["id"]] = await llm_five_whys(r)

    report = build_report(summary, results, llm_sections)
    os.makedirs("analysis", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Đã ghi {OUTPUT_PATH} ({len(report)} ký tự).")


if __name__ == "__main__":
    asyncio.run(main())
