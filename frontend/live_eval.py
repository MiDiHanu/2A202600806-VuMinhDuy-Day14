"""
Live evaluation cho dashboard demo — chạy 1 câu hỏi qua TOÀN BỘ pipeline và trả
về trace chi tiết từng tầng (kèm thời gian) để frontend visualize.
"""
import os
import sys
import time
import json
import asyncio
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.ragas_metrics import RagEvaluator
from engine.llm_judge import MultiModelJudge
from engine.llm_client import usage_ledger

# Cache agent/evaluator để không nạp lại corpus mỗi request.
_AGENTS: Dict[str, MainAgent] = {}
_RAG: Optional[RagEvaluator] = None
_JUDGE: Optional[MultiModelJudge] = None
_DATASET: Optional[List[Dict]] = None
_CORPUS_TITLES: Dict[str, str] = {}


def _get_agent(version: str) -> MainAgent:
    if version not in _AGENTS:
        _AGENTS[version] = MainAgent(version=version)
        for c in _AGENTS[version].corpus:
            _CORPUS_TITLES[c["id"]] = c["title"]
    return _AGENTS[version]


def _get_dataset() -> List[Dict]:
    global _DATASET
    if _DATASET is None:
        path = "data/golden_set.jsonl"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _DATASET = [json.loads(l) for l in f if l.strip()]
        else:
            _DATASET = []
    return _DATASET


def _match_golden(question: str) -> Optional[Dict]:
    q = question.strip().lower()
    for c in _get_dataset():
        if c["question"].strip().lower() == q:
            return c
    return None


async def evaluate_live(question: str, version: str = "V2_Optimized") -> Dict:
    global _RAG, _JUDGE
    if _RAG is None:
        _RAG = RagEvaluator()
    if _JUDGE is None:
        _JUDGE = MultiModelJudge()

    agent = _get_agent(version)
    retr_eval = RetrievalEvaluator(top_k=agent.cfg["top_k"])

    golden = _match_golden(question)
    ground_truth = golden["expected_answer"] if golden else "(Không có đáp án chuẩn — đánh giá tương đối theo ngữ cảnh.)"
    expected_ids = golden.get("expected_retrieval_ids", []) if golden else []

    cost_before = usage_ledger.report()["total_cost_usd"]
    t_total = time.perf_counter()

    # --- Tầng 1+2: Agent (Retrieval + Generation) ---
    t0 = time.perf_counter()
    resp = await agent.query(question)
    gen_ms = round((time.perf_counter() - t0) * 1000)

    retrieved = [
        {
            "id": rid,
            "title": _CORPUS_TITLES.get(rid, ""),
            "score": resp["retrieval_scores"][i] if i < len(resp["retrieval_scores"]) else None,
            "is_expected": rid in expected_ids,
        }
        for i, rid in enumerate(resp["retrieved_ids"])
    ]
    retr_metric = retr_eval.score_single(expected_ids, resp["retrieved_ids"])

    # --- Tầng 3+4: RAGAS + Multi-Judge (song song) ---
    case = {"question": question, "expected_answer": ground_truth, "expected_retrieval_ids": expected_ids}
    t1 = time.perf_counter()
    ragas, judge = await asyncio.gather(
        _RAG.score(case, resp),
        _JUDGE.evaluate_multi_judge(question, resp["answer"], ground_truth),
    )
    eval_ms = round((time.perf_counter() - t1) * 1000)

    cost_after = usage_ledger.report()["total_cost_usd"]

    return {
        "question": question,
        "version": version,
        "mode": agent.llm.mode,
        "matched_golden": golden["id"] if golden else None,
        "ground_truth": ground_truth,
        "expected_retrieval_ids": expected_ids,
        "stages": {
            "retrieval": {
                "top_k": agent.cfg["top_k"],
                "retrieved": retrieved,
                "metric": retr_metric,
            },
            "generation": {
                "answer": resp["answer"],
                "contexts_used": len(resp["contexts"]),
                "model": resp["metadata"]["model"],
                "tokens": {
                    "prompt": resp["metadata"]["prompt_tokens"],
                    "completion": resp["metadata"]["completion_tokens"],
                },
                "time_ms": gen_ms,
            },
            "ragas": {**ragas, "time_ms": eval_ms},
            "judge": judge,
        },
        "cost_usd": round(cost_after - cost_before, 6),
        "total_time_ms": round((time.perf_counter() - t_total) * 1000),
    }


def evaluate_live_sync(question: str, version: str = "V2_Optimized") -> Dict:
    return asyncio.run(evaluate_live(question, version))
