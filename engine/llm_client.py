"""
LLM Client thống nhất cho Lab 14.

Mục tiêu:
- Gọi Google Gemini qua endpoint OpenAI-compatible (giữ nguyên SDK `openai`).
- Tự động chuyển sang chế độ OFFLINE (mock deterministic) khi không có API key,
  để `main.py` / `check_lab.py` luôn chạy được mà không cần internet hay tiền.
- Theo dõi token usage & chi phí (USD) toàn cục để phục vụ báo cáo Cost.

Cách dùng:
    from engine.llm_client import LLMClient, usage_ledger
    client = LLMClient(model="gemini-2.5-flash")
    text, usage = await client.complete("system prompt", "user prompt")
    print(usage_ledger.report())
"""
from __future__ import annotations

import os
import re
import json
import hashlib
import asyncio
from typing import Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv luôn có trong deps
    pass

# Endpoint OpenAI-compatible của Gemini.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Bảng giá tham khảo (USD / 1 triệu token). Dùng để ước tính chi phí eval.
# Nguồn: Google AI pricing (giá có thể thay đổi, chỉ dùng cho mục đích báo cáo).
PRICING_PER_M: Dict[str, Dict[str, float]] = {
    "gemini-2.5-pro":   {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro":   {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # fallback cho model không có trong bảng
    "_default":         {"input": 0.30, "output": 2.50},
}


def estimate_tokens(text: str) -> int:
    """Ước tính số token thô (~4 ký tự/token). Dùng cho chế độ offline."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class UsageLedger:
    """Sổ cái token/chi phí toàn cục, tích lũy qua mọi lần gọi LLM."""

    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, int]] = {}
        self.calls = 0

    def reset(self) -> None:
        self.records.clear()
        self.calls = 0

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        bucket = self.records.setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        bucket["prompt_tokens"] += int(prompt_tokens)
        bucket["completion_tokens"] += int(completion_tokens)
        bucket["calls"] += 1
        self.calls += 1

    def _cost_for(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = PRICING_PER_M.get(model, PRICING_PER_M["_default"])
        return (
            prompt_tokens / 1_000_000 * price["input"]
            + completion_tokens / 1_000_000 * price["output"]
        )

    def report(self) -> Dict:
        total_prompt = total_completion = 0
        total_cost = 0.0
        per_model = {}
        for model, b in self.records.items():
            cost = self._cost_for(model, b["prompt_tokens"], b["completion_tokens"])
            per_model[model] = {
                "calls": b["calls"],
                "prompt_tokens": b["prompt_tokens"],
                "completion_tokens": b["completion_tokens"],
                "cost_usd": round(cost, 6),
            }
            total_prompt += b["prompt_tokens"]
            total_completion += b["completion_tokens"]
            total_cost += cost
        return {
            "total_calls": self.calls,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": round(total_cost, 6),
            "per_model": per_model,
        }


# Singleton dùng chung cho toàn pipeline.
usage_ledger = UsageLedger()


class LLMClient:
    """Wrapper gọi LLM. Online = Gemini; Offline = mock deterministic."""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.offline = not self.api_key
        self._client = None
        if not self.offline:
            try:
                from openai import AsyncOpenAI

                # timeout=30s để request treo fail nhanh (mặc định SDK là 600s!).
                # max_retries=0 vì ta tự xử lý retry/backoff ở complete().
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=GEMINI_BASE_URL,
                    timeout=float(os.getenv("LLM_TIMEOUT", "30")),
                    max_retries=0,
                )
            except Exception as e:  # pragma: no cover
                print(f"⚠️  Không khởi tạo được Gemini client ({e}). Chuyển sang OFFLINE.")
                self.offline = True

    @property
    def mode(self) -> str:
        return "OFFLINE (mock)" if self.offline else f"ONLINE ({self.model})"

    async def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> Tuple[str, Dict[str, int]]:
        """Trả về (text, usage). Luôn ghi nhận token vào usage_ledger."""
        if self.offline:
            return self._mock_complete(system, user, json_mode)

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                }
                usage_ledger.record(
                    self.model, usage["prompt_tokens"], usage["completion_tokens"]
                )
                return text, usage
            except Exception as e:
                last_err = e
                # Rate-limit (429) cần chờ lâu hơn nhiều so với lỗi mạng thường.
                msg = str(e).lower()
                is_rate = "429" in msg or "resource_exhausted" in msg or "rate" in msg or "quota" in msg
                base = 4.0 if is_rate else 1.0
                await asyncio.sleep(min(12.0, base * (attempt + 1)))
        raise RuntimeError(f"Gemini call thất bại sau {max_retries} lần: {last_err}")

    async def complete_json(
        self, system: str, user: str, max_retries: int = 3
    ) -> Tuple[Optional[dict], Dict[str, int]]:
        """Như complete() nhưng parse JSON. Trả (dict|None, usage)."""
        text, usage = await self.complete(system, user, json_mode=True, max_retries=max_retries)
        return _safe_json(text), usage

    # ------------------------------------------------------------------ mock
    def _mock_complete(
        self, system: str, user: str, json_mode: bool
    ) -> Tuple[str, Dict[str, int]]:
        """Sinh phản hồi giả deterministic (theo hash) + ghi token ước tính."""
        seed = int(hashlib.sha256((system + user).encode("utf-8")).hexdigest(), 16)
        if json_mode:
            text = json.dumps({"mock": True, "seed": seed % 1000}, ensure_ascii=False)
        else:
            text = (
                "[MOCK] Phản hồi mô phỏng deterministic dựa trên nội dung đầu vào. "
                "Cắm GEMINI_API_KEY vào .env để dùng model thật."
            )
        pt, ct = estimate_tokens(system + user), estimate_tokens(text)
        usage = {"prompt_tokens": pt, "completion_tokens": ct}
        usage_ledger.record(self.model, pt, ct)
        return text, usage


def _safe_json(text: str) -> Optional[dict]:
    """Trích JSON đầu tiên từ text (chịu được ```json fences)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
