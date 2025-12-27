#!/usr/bin/env python3
"""
Incoming: nq_test.json, psgs_w100.tsv, LM_STUDIO_URL --- {JSON, TSV, str}
Processing: BM25 retrieval + k-shot RAG + QA scoring --- {4 jobs: indexing, retrieval, generation, evaluation}
Outgoing: results JSON + best-k summary --- {JSON, text}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Project root on path
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config, ensure_pyterrier_init  # noqa: E402
from src.retrievers.bm25 import BM25Retriever  # noqa: E402
from src.generation import QAGenerator  # noqa: E402
from src.evaluation.qa_evaluator import QAEvaluator  # noqa: E402
from src.data_utils import LazyTSVCorpus, load_nq_test  # noqa: E402


@dataclass(frozen=True)
class NQExample:
    qid: str
    question: str
    gold_answers: List[str]


def stream_psgs_tsv(psgs_tsv_path: str, limit: Optional[int] = None) -> Iterable[Dict[str, str]]:
    # Stream index documents without loading all into RAM.
    # Expected format: id<TAB>text<TAB>title
    with open(psgs_tsv_path, "rb") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            parts = line.split(b"\t")
            if len(parts) < 2:
                continue
            doc_id = parts[0].decode("utf-8", errors="ignore").strip()
            if not doc_id or doc_id.lower() == "id":
                continue
            text = parts[1].decode("utf-8", errors="ignore").strip()
            title = ""
            if len(parts) >= 3:
                title = parts[2].decode("utf-8", errors="ignore").strip()
            yield {"docno": doc_id, "text": (title + " " + text).strip()}


def build_pyterrier_index(psgs_tsv_path: str, index_dir: str, limit_docs: Optional[int] = None) -> str:
    pt = ensure_pyterrier_init()
    os.makedirs(index_dir, exist_ok=True)
    t0 = time.time()
    indexer = pt.IterDictIndexer(index_dir, meta={"docno": 100}, verbose=True)
    index_ref = indexer.index(stream_psgs_tsv(psgs_tsv_path, limit=limit_docs))
    print(f"[index] built in {time.time()-t0:.1f}s -> {index_ref}")
    return str(index_ref)


def format_context(doc_ids: List[str], corpus: LazyTSVCorpus, k: int) -> List[Dict[str, str]]:
    ctx: List[Dict[str, str]] = []
    for doc_id in doc_ids[:k]:
        doc = corpus.get(doc_id)
        if not doc:
            continue
        ctx.append({"title": doc.get("title", ""), "text": doc.get("text", "")})
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 baseline on nq_test.json with k-shot context sweep.")
    ap.add_argument("--psgs_tsv", required=True, help="Path to psgs_w100.tsv")
    ap.add_argument("--nq_test", default=str(PROJECT_ROOT / "data" / "nq_test.json"), help="Path to nq_test.json")
    ap.add_argument("--index_dir", default=str(PROJECT_ROOT / "data" / "NQ" / "index" / "pyterrier_psgs_w100"),
                    help="Path to PyTerrier index directory")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "data" / "NQ" / "results" / "bm25_kshot.json"),
                    help="Output results JSON")
    ap.add_argument("--k_values", default="1,2,3,4,5,6", help="Comma-separated k values (context docs)")
    ap.add_argument("--retrieve_k", type=int, default=100, help="BM25 retrieve depth (must be >= max(k_values))")
    ap.add_argument("--limit_queries", type=int, default=None, help="Limit queries (debug)")
    ap.add_argument("--limit_docs", type=int, default=None, help="Limit docs for indexing (debug)")
    ap.add_argument("--model", default=config.models.lm_studio.default_model, help="OpenAI-compatible model name")
    args = ap.parse_args()

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    if not k_values:
        raise ValueError("k_values empty")
    if args.retrieve_k < max(k_values):
        raise ValueError("retrieve_k must be >= max(k_values)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)

    # Index
    index_has_files = any(Path(args.index_dir).iterdir())
    if not index_has_files:
        print(f"[index] building PyTerrier index at {args.index_dir} ...")
        build_pyterrier_index(args.psgs_tsv, args.index_dir, limit_docs=args.limit_docs)
    else:
        print(f"[index] exists: {args.index_dir}")

    # Load dataset
    raw_examples = load_nq_test(args.nq_test, limit=args.limit_queries)
    examples = [NQExample(**ex) for ex in raw_examples]
    print(f"[data] nq_test loaded: {len(examples)} examples")
    queries: Dict[str, str] = {e.qid: e.question for e in examples}
    gold: Dict[str, List[str]] = {e.qid: e.gold_answers for e in examples}

    # Retrieval
    retriever = BM25Retriever(args.index_dir)
    run = retriever.retrieve_batch(queries, top_k=args.retrieve_k)
    print(f"[bm25] retrieved: {len(run)} queries")

    # Corpus (lazy)
    corpus = LazyTSVCorpus(args.psgs_tsv)

    # Generation + scoring
    generator = QAGenerator()
    evaluator = QAEvaluator(metrics=["em", "f1", "containment"])

    results = []
    agg: Dict[int, List[float]] = {k: [] for k in k_values}
    agg_em: Dict[int, List[float]] = {k: [] for k in k_values}
    agg_cont: Dict[int, List[float]] = {k: [] for k in k_values}

    t0 = time.time()
    for i, e in enumerate(examples):
        qid = e.qid
        doc_ids = [docno for (docno, _, _) in run.get(qid, []).results]
        entry = {"qid": qid, "query": e.question, "shots": {}}
        for k in k_values:
            ctx_docs = format_context(doc_ids, corpus, k)
            gen = generator.generate(
                query=e.question,
                context=ctx_docs,
                model=args.model,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_tokens,
            )
            ans = (gen.get("answer") or "").strip()
            m = evaluator.evaluate(ans, gold[qid])
            entry["shots"][str(k)] = {
                "k": k,
                "answer": ans,
                "latency_ms": gen.get("latency_ms", 0),
                "em": m.get("em", 0.0),
                "f1": m.get("f1", 0.0),
                "containment": m.get("containment", 0.0),
                "gold_answers": gold[qid],
            }
            agg[k].append(m.get("f1", 0.0))
            agg_em[k].append(m.get("em", 0.0))
            agg_cont[k].append(m.get("containment", 0.0))
        results.append(entry)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[run] {i+1}/{len(examples)} queries in {elapsed/60:.1f} min")

    summary = {}
    best_k = None
    best_f1 = -1.0
    for k in k_values:
        f1 = sum(agg[k]) / len(agg[k]) if agg[k] else 0.0
        em = sum(agg_em[k]) / len(agg_em[k]) if agg_em[k] else 0.0
        cont = sum(agg_cont[k]) / len(agg_cont[k]) if agg_cont[k] else 0.0
        summary[str(k)] = {
            "avg_f1": round(f1 * 100, 2),
            "avg_em": round(em * 100, 2),
            "avg_containment": round(cont * 100, 2),
            "n": len(agg[k]),
        }
        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    payload = {
        "_metadata": {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": Path(__file__).name,
            "model": args.model,
            "lm_studio_url": os.environ.get("LM_STUDIO_URL", ""),
            "psgs_tsv": args.psgs_tsv,
            "nq_test": args.nq_test,
            "retrieve_k": args.retrieve_k,
            "k_values": k_values,
        },
        "summary": {"best_k_by_f1": best_k, "metrics_by_k": summary},
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[done] wrote: {out_path}")
    print(f"[done] best_k_by_f1={best_k} (avg_f1={best_f1*100:.2f}%)")


if __name__ == "__main__":
    main()

