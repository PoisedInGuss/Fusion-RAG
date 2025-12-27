#!/usr/bin/env python3
"""
Incoming: /mnt/datasets/dpr/nq_test.json, TREC runfiles, psgs_w100.tsv --- {Dict, JSON/TSV}
Processing: RAG generation + QA scoring --- {3 jobs: generation, evaluation, aggregation}
Outgoing: shared-rag-results shards --- {Dict, JSONL}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config  # noqa: E402
from src.generation import (  # noqa: E402
    QAGenerator,
    GenerationError,
    LMStudioConnectionError,
)
from src.evaluation.qa_evaluator import QAEvaluator  # noqa: E402
from src.data_utils import LazyTSVCorpus, load_nq_test, load_run_file  # noqa: E402


@dataclass(frozen=True)
class NQExample:
    qid: str
    question: str
    gold_answers: List[str]


class ShotAccumulator:
    def __init__(self, k_values: List[int]):
        self.data: Dict[str, Dict[str, float]] = {}
        for k in k_values:
            key = str(k)
            self.data[key] = {
                "n": 0,
                "em_sum": 0.0,
                "f1_sum": 0.0,
                "containment_sum": 0.0,
                "latency_ms_sum": 0.0,
                "context_chars_sum": 0.0,
            }

    def update(self, record: Dict) -> None:
        for k_str, shot in record["shots"].items():
            stats = self.data.setdefault(k_str, {
                "n": 0,
                "em_sum": 0.0,
                "f1_sum": 0.0,
                "containment_sum": 0.0,
                "latency_ms_sum": 0.0,
                "context_chars_sum": 0.0,
            })
            stats["n"] += 1
            stats["em_sum"] += float(shot.get("em", 0.0))
            stats["f1_sum"] += float(shot.get("f1", 0.0))
            stats["containment_sum"] += float(shot.get("containment", 0.0))
            stats["latency_ms_sum"] += float(shot.get("latency_ms", 0.0))
            stats["context_chars_sum"] += float(shot.get("context_chars", 0.0))

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for k_str, stats in self.data.items():
            n = max(1, stats["n"])
            result[k_str] = {
                "n": stats["n"],
                "em": stats["em_sum"] / n,
                "f1": stats["f1_sum"] / n,
                "containment": stats["containment_sum"] / n,
                "latency_ms": stats["latency_ms_sum"] / n,
                "context_chars": stats["context_chars_sum"] / n,
            }
        return result


def load_existing_shards(
    shards_dir: Path,
    global_stats: ShotAccumulator,
) -> Tuple[int, int, Set[str]]:
    """Return (processed_count, shard_idx, completed_qids) from existing shards."""
    processed = 0
    shard_idx = 0
    completed_qids: Set[str] = set()

    if not shards_dir.exists():
        return processed, shard_idx, completed_qids

    shard_files = sorted(shards_dir.glob("shard_*.jsonl"))
    for shard_file in shard_files:
        shard_idx += 1
        with shard_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qid = record.get("qid")
                if qid:
                    completed_qids.add(qid)
                global_stats.update(record)
                processed += 1

    return processed, shard_idx, completed_qids


def hydrate_doc_cache_from_dump(
    docs_path: Path,
    doc_cache: Dict[str, Dict[str, str]],
    doc_ids_seen: set,
) -> None:
    if not docs_path.exists():
        return
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            doc_id = payload.get("docid")
            if not doc_id:
                continue
            doc_ids_seen.add(doc_id)
            doc_cache[doc_id] = {
                "title": payload.get("title", ""),
                "text": payload.get("text", ""),
            }


def build_context(
    doc_ids: List[str],
    corpus: LazyTSVCorpus,
    k: int,
    cache: Dict[str, Dict[str, str]],
    seen_ids: set,
    missing_ids: set,
) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for doc_id in doc_ids[:k]:
        seen_ids.add(doc_id)
        doc = cache.get(doc_id)
        if doc is None:
            doc = corpus.get(doc_id)
            if doc:
                cache[doc_id] = doc
            else:
                missing_ids.add(doc_id)
                continue
        docs.append({"docid": doc_id, "title": doc.get("title", ""), "text": doc.get("text", "")})
    return docs


def write_shard(
    shard_idx: int,
    total_shards: int,
    records: List[Dict],
    shards_dir: Path,
    k_values: List[int],
) -> None:
    shard_file = shards_dir / f"shard_{shard_idx:02d}.jsonl"
    with shard_file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    shard_stats = ShotAccumulator(k_values)
    for record in records:
        shard_stats.update(record)

    meta = {
        "shard_index": shard_idx,
        "shard_count": total_shards,
        "summary": shard_stats.summary(),
    }
    meta_file = shards_dir / f"shard_{shard_idx:02d}.meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[shard] wrote {shard_file.name} ({len(records)} records)")


def generate_with_retry(
    generator: QAGenerator,
    query: str,
    context: List[Dict[str, str]],
    model: str,
    max_retries: int,
) -> Dict:
    for attempt in range(1, max_retries + 1):
        try:
            return generator.generate(
                query=query,
                context=context,
                model=model,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_tokens,
            )
        except (LMStudioConnectionError, GenerationError) as exc:
            wait = min(10, attempt * 2)
            print(f"[gen] attempt {attempt}/{max_retries} failed: {exc}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Generation failed after {max_retries} attempts for query: {query[:80]}")


def ensure_model_available(model_id: str, base_url: str) -> None:
    """Verify requested model exists on the LM Studio/vLLM endpoint."""
    models_endpoint = f"{base_url.rstrip('/')}/models"
    headers = {}
    api_key = os.environ.get("IDA_LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(models_endpoint, headers=headers, timeout=10)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to query models endpoint {models_endpoint}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Models endpoint {models_endpoint} returned {resp.status_code}: {resp.text}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON from {models_endpoint}: {exc}") from exc

    ids = {entry.get("id") for entry in payload.get("data", [])}
    if model_id not in ids:
        raise RuntimeError(
            f"Model '{model_id}' not served by {models_endpoint}. "
            f"Available: {sorted(ids) if ids else 'none'}"
        )


def write_docs_dump(
    docs_path: Path,
    meta_path: Path,
    doc_ids: set,
    cache: Dict[str, Dict[str, str]],
    max_k: int,
    run_path: str,
    tsv_path: str,
) -> None:
    found = 0
    with docs_path.open("w", encoding="utf-8") as fh:
        for doc_id in sorted(doc_ids):
            doc = cache.get(doc_id)
            if not doc:
                continue
            found += 1
            payload = {"docid": doc_id, "title": doc.get("title", ""), "text": doc.get("text", "")}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    meta = {
        "needed": len(doc_ids),
        "found": found,
        "max_k": max_k,
        "run_path": run_path,
        "tsv_path": tsv_path,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[docs] dumped {found}/{len(doc_ids)} unique docs to {docs_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG generation from precomputed runfiles.")
    parser.add_argument("--nq_path", required=True, help="Path to nq_test.json")
    parser.add_argument("--psgs_path", required=True, help="Path to psgs_w100.tsv")
    parser.add_argument("--run_path", required=True, help="TREC runfile path")
    parser.add_argument("--retriever", required=True, help="Retriever name (e.g., BM25, TCTColBERT)")
    parser.add_argument("--dataset", default="nq_test", help="Dataset identifier (default: nq_test)")
    parser.add_argument("--rag_variant", default=None, help="Optional variant label for metrics")
    parser.add_argument("--output_root", required=True, help="Output directory under shared-rag-results")
    parser.add_argument("--shots", default="0,1,2,3,4,5,6", help="Comma-separated k values")
    parser.add_argument("--shard_size", type=int, default=500, help="Records per shard")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (debug)")
    parser.add_argument("--model", default=config.models.lm_studio.default_model, help="LLM model id")
    parser.add_argument("--prompt_variant", default="default", help="Prompt variant: 'default' or 'variant_b'")
    parser.add_argument("--overwrite", action="store_true", help="Delete output_root if it exists")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output directory")
    parser.add_argument("--max_retries", type=int, default=3, help="LLM retry attempts")
    parser.add_argument("--log_every", type=int, default=25, help="Progress log frequency")
    return parser.parse_args()


def main():
    args = parse_args()

    k_values = sorted({int(k.strip()) for k in args.shots.split(",") if k.strip()})
    if not k_values:
        raise ValueError("No k values provided.")
    max_k = max(k_values)

    rag_variant = args.rag_variant or f"{args.retriever}_k{k_values[0]}to{k_values[-1]}"

    output_root = Path(args.output_root)
    shards_dir = output_root / "shards"
    docs_path = output_root / f"docs_top{max_k}.jsonl"
    docs_meta_path = output_root / f"docs_top{max_k}.meta.json"

    if output_root.exists():
        if args.overwrite:
            print(f"[setup] removing existing output_dir: {output_root}")
            shutil.rmtree(output_root)
            shards_dir.mkdir(parents=True, exist_ok=True)
        elif args.resume:
            shards_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise FileExistsError(
                f"{output_root} already exists (use --overwrite or --resume)"
            )
    else:
        shards_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading nq_test from {args.nq_path}")
    raw_examples = load_nq_test(args.nq_path, limit=args.limit)
    examples = [NQExample(**ex) for ex in raw_examples]
    if not examples:
        raise RuntimeError("No usable examples found in nq_test.json")

    print(f"[data] loading runfile {args.run_path}")
    run_data = load_run_file(args.run_path)

    print(f"[data] indexing corpus {args.psgs_path}")
    corpus = LazyTSVCorpus(args.psgs_path)

    api_key = os.environ.get("IDA_LLM_API_KEY")
    ensure_model_available(args.model, base_url=config.models.lm_studio.base_url)
    generator = QAGenerator(api_key=api_key, prompt_variant=args.prompt_variant)
    evaluator = QAEvaluator(metrics=["em", "f1", "containment"])

    doc_cache: Dict[str, Dict[str, str]] = {}
    doc_ids_seen: set = set()
    missing_doc_ids: set = set()
    if args.resume:
        hydrate_doc_cache_from_dump(docs_path, doc_cache, doc_ids_seen)

    global_stats = ShotAccumulator(k_values)
    total_queries = len(examples)
    total_shards = max(1, math.ceil(total_queries / args.shard_size))

    processed = 0
    shard_idx = 0
    completed_qids: Set[str] = set()
    if args.resume and output_root.exists():
        processed, shard_idx, completed_qids = load_existing_shards(shards_dir, global_stats)
        if processed:
            print(f"[resume] found {processed} completed queries across {shard_idx} shards")

    shard_records: List[Dict] = []

    for example in examples:
        if example.qid in completed_qids:
            continue
        doc_ids = [doc_id for doc_id, _, _ in run_data.get(example.qid, [])]

        record = {
            "qid": example.qid,
            "question": example.question,
            "gold_answers": example.gold_answers,
            "shots": {},
        }

        for k in k_values:
            context_docs = build_context(
                doc_ids=doc_ids,
                corpus=corpus,
                k=k,
                cache=doc_cache,
                seen_ids=doc_ids_seen,
                missing_ids=missing_doc_ids,
            )
            context_chars = 0
            if context_docs:
                context_chars = len(generator._format_context(context_docs))  # pylint: disable=protected-access

            gen_result = generate_with_retry(
                generator=generator,
                query=example.question,
                context=context_docs,
                model=args.model,
                max_retries=args.max_retries,
            )
            answer = (gen_result.get("answer") or "").strip()
            metrics = evaluator.evaluate(answer, example.gold_answers)

            record["shots"][str(k)] = {
                "k": k,
                "answer": answer,
                "latency_ms": gen_result.get("latency_ms", 0.0),
                "em": metrics.get("em", 0.0),
                "f1": metrics.get("f1", 0.0),
                "containment": metrics.get("containment", 0.0),
                "context_chars": context_chars,
            }

        global_stats.update(record)
        shard_records.append(record)
        processed += 1

        if processed % args.log_every == 0 or processed == total_queries:
            print(f"[progress] {processed}/{total_queries} queries processed")

        if len(shard_records) >= args.shard_size:
            write_shard(shard_idx, total_shards, shard_records, shards_dir, k_values)
            shard_records = []
            shard_idx += 1

    if shard_records:
        write_shard(shard_idx, total_shards, shard_records, shards_dir, k_values)

    write_docs_dump(
        docs_path=docs_path,
        meta_path=docs_meta_path,
        doc_ids=doc_ids_seen,
        cache=doc_cache,
        max_k=max_k,
        run_path=args.run_path,
        tsv_path=args.psgs_path,
    )

    metrics_payload = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "rag_variant": rag_variant,
        "num_examples": processed,
        "shots": global_stats.summary(),
    }
    metrics_file = output_root / "metrics.json"
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[metrics] wrote aggregate metrics to {metrics_file}")

    run_config = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "rag_variant": rag_variant,
        "run_path": args.run_path,
        "psgs_path": args.psgs_path,
        "nq_path": args.nq_path,
        "shots": k_values,
        "model": args.model,
        "shard_size": args.shard_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if missing_doc_ids:
        print(f"[warn] missing {len(missing_doc_ids)} doc ids in TSV (see run logs)")

    print(f"[done] processed {processed} queries -> {output_root}")


if __name__ == "__main__":
    main()

