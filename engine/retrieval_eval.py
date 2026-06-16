"""
Retrieval Evaluation — Hit Rate & MRR.

Đo chất lượng tầng Retrieval TRƯỚC khi đánh giá Generation. Nếu retrieval sai
thì hallucination ở tầng sinh là hệ quả tất yếu => phải tách bạch để biết lỗi ở đâu.

Quy ước với Red-Team case (expected_retrieval_ids == []):
  Đây là câu KHÔNG nên có tài liệu nào trả lời (out-of-context...). Ta không tính
  các case này vào Hit Rate/MRR trung bình (chúng được đánh giá ở tầng Judge: agent
  có biết từ chối không), nhưng vẫn báo cáo riêng số lượng.
"""
from typing import List, Dict, Optional


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: Optional[int] = None
    ) -> float:
        """1.0 nếu ít nhất 1 expected_id nằm trong top_k retrieved_ids, ngược lại 0.0."""
        k = top_k or self.top_k
        top = retrieved_ids[:k]
        return 1.0 if any(eid in top for eid in expected_ids) else 0.0

    def calculate_mrr(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> float:
        """Mean Reciprocal Rank: 1/(vị trí 1-indexed đầu tiên trúng). Không trúng -> 0."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def score_single(self, expected_ids: List[str], retrieved_ids: List[str]) -> Dict:
        """Chấm 1 case. Trả None cho hit/mrr nếu là case không cần retrieval."""
        if not expected_ids:
            return {"hit_rate": None, "mrr": None, "scored": False}
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "scored": True,
        }

    def aggregate(self, per_case: List[Dict]) -> Dict:
        """Tổng hợp avg hit_rate & mrr chỉ trên các case có chấm (scored=True)."""
        scored = [c for c in per_case if c.get("scored")]
        n = len(scored)
        if n == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "num_scored": 0}
        return {
            "avg_hit_rate": sum(c["hit_rate"] for c in scored) / n,
            "avg_mrr": sum(c["mrr"] for c in scored) / n,
            "num_scored": n,
        }

    async def evaluate_batch(self, dataset: List[Dict], agent) -> Dict:
        """
        Chạy retrieval cho toàn bộ dataset bằng `agent` và tổng hợp metrics.
        Dataset cần có 'expected_retrieval_ids'; agent.query trả 'retrieved_ids'.
        Dùng độc lập (standalone). Pipeline chính chấm qua score_single trong runner
        để tránh gọi agent 2 lần.
        """
        per_case = []
        for case in dataset:
            resp = await agent.query(case["question"])
            per_case.append(
                self.score_single(case.get("expected_retrieval_ids", []), resp.get("retrieved_ids", []))
            )
        return self.aggregate(per_case)
