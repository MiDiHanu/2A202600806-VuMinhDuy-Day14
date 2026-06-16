"""
RAGAS-style metrics (tự implement, không phụ thuộc thư viện ragas vốn khóa OpenAI).

- faithfulness: câu trả lời có được CHỨNG MINH bởi context đã truy hồi không?
  (thấp = hallucination — agent bịa thông tin ngoài tài liệu).
- answer_relevancy: câu trả lời có đúng trọng tâm câu hỏi không?

Cả hai chấm bằng Gemini (LLM-as-judge) trong 1 call JSON để tiết kiệm chi phí.
Offline: dùng heuristic overlap từ vựng để vẫn có số liệu.
"""
import re
import unicodedata
from typing import Dict, List

from engine.llm_client import LLMClient

_FAITH_REL_SYSTEM = (
    "Bạn là giám khảo đánh giá hệ thống RAG. Cho điểm khách quan, trả về DUY NHẤT JSON."
)

_FAITH_REL_TEMPLATE = """Đánh giá câu trả lời của trợ lý theo 2 tiêu chí, mỗi tiêu chí từ 0.0 đến 1.0.

CÂU HỎI:
{question}

NGỮ CẢNH (tài liệu đã truy hồi):
{contexts}

CÂU TRẢ LỜI CỦA TRỢ LÝ:
{answer}

Tiêu chí:
1. faithfulness: Câu trả lời có được CHỨNG MINH hoàn toàn bởi NGỮ CẢNH không?
   - 1.0 = mọi khẳng định đều có trong ngữ cảnh.
   - 0.0 = bịa đặt, mâu thuẫn, hoặc khẳng định thông tin không có trong ngữ cảnh.
   - Nếu trợ lý đúng đắn từ chối/nói "không có thông tin" khi ngữ cảnh trống thì faithfulness = 1.0.
2. answer_relevancy: Câu trả lời có trực tiếp giải quyết CÂU HỎI không (đúng trọng tâm, không lan man)?

Trả JSON: {{"faithfulness": <float>, "answer_relevancy": <float>, "reason": "<ngắn gọn>"}}"""


def _strip(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


def _tokens(s: str) -> set:
    return set(re.findall(r"[0-9a-z]+", _strip(s)))


class RagEvaluator:
    """Tính faithfulness & answer_relevancy."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.llm = LLMClient(model=model, temperature=0.0)

    async def score(self, case: Dict, response: Dict) -> Dict:
        question = case["question"]
        answer = response.get("answer", "")
        contexts: List[str] = response.get("contexts", [])
        ctx_text = "\n".join(f"- {c}" for c in contexts) if contexts else "(trống)"

        if self.llm.offline:
            return self._heuristic(question, answer, contexts)

        prompt = _FAITH_REL_TEMPLATE.format(
            question=question, contexts=ctx_text, answer=answer
        )
        data, _ = await self.llm.complete_json(_FAITH_REL_SYSTEM, prompt)
        if not data:
            return self._heuristic(question, answer, contexts)
        return {
            "faithfulness": _clamp(data.get("faithfulness", 0.0)),
            "relevancy": _clamp(data.get("answer_relevancy", 0.0)),
            "reason": data.get("reason", ""),
        }

    def _heuristic(self, question: str, answer: str, contexts: List[str]) -> Dict:
        a_tok = _tokens(answer)
        q_tok = _tokens(question)
        c_tok = set().union(*[_tokens(c) for c in contexts]) if contexts else set()
        faith = (len(a_tok & c_tok) / len(a_tok)) if a_tok and c_tok else (
            1.0 if not contexts and ("không" in answer.lower()) else 0.3
        )
        rel = (len(a_tok & q_tok) / len(q_tok)) if q_tok else 0.0
        return {
            "faithfulness": round(min(1.0, faith), 3),
            "relevancy": round(min(1.0, rel), 3),
            "reason": "heuristic offline (token overlap)",
        }


def _clamp(x) -> float:
    try:
        return round(max(0.0, min(1.0, float(x))), 3)
    except (TypeError, ValueError):
        return 0.0
