#!/usr/bin/env python3
"""
Incoming: fused .res, corpus, queries, qrels --- {TREC run, BEIR data}
Processing: k-shot RAG evaluation --- {N jobs: per-query generation + eval}
Outgoing: evaluation results --- {JSON}

Step 7: RAG End-Task Evaluation
-------------------------------
Evaluates RAG downstream performance using LLM generation.
Tests how fusion quality impacts answer generation.
Supports checkpointing for resume on restart.

Usage:
    python scripts/07_rag_eval.py --corpus_path /data/beir/datasets/nq
    python scripts/07_rag_eval.py --corpus_path /data/beir/datasets/nq --shots 1,2,6 --limit 100
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config first - sets up environment
from src.config import config, detect_dataset, get_model_safe_name

from src.generation import GenerationOperation
from src.data_utils import (
    LazyCorpus,
    load_corpus as _load_corpus,
    load_queries as _load_queries,
    load_qrels as _load_qrels,
    load_run_file as _load_run,
)


def load_corpus(corpus_path: str) -> LazyCorpus:
    """Load BEIR corpus with lazy loading."""
    print(f"[07_rag] Loading corpus (lazy mode)...")
    corpus = _load_corpus(corpus_path, lazy=True)
    print(f"[07_rag] Indexed {len(corpus)} documents (0 loaded in RAM)")
    return corpus


def load_queries(corpus_path: str) -> Dict[str, str]:
    """Load BEIR queries."""
    queries = _load_queries(corpus_path, split="test")
    print(f"[07_rag] Queries: {len(queries)}")
    return queries


def load_qrels(corpus_path: str) -> Dict[str, Dict[str, int]]:
    """Load BEIR qrels."""
    qrels_file = os.path.join(corpus_path, "qrels", "test.tsv")
    qrels = _load_qrels(qrels_file)
    print(f"[07_rag] Qrels: {len(qrels)} queries with judgments")
    return qrels


def load_run(run_path: str) -> Dict[str, List[tuple]]:
    """Load TREC run file."""
    run_data = _load_run(run_path)
    print(f"[07_rag] Loaded run for {len(run_data)} queries")
    return run_data


def build_context(doc_ids: List[str], corpus: Dict[str, Dict[str, str]], k: int) -> str:
    """Build context string from top-k documents."""
    if k == 0 or not doc_ids:
        return ""
    
    context_parts = []
    for i, doc_id in enumerate(doc_ids[:k]):
        if doc_id in corpus:
            doc = corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                context_parts.append(f"[Document {i+1}] {title}\n{text}")
            else:
                context_parts.append(f"[Document {i+1}]\n{text}")
    
    return "\n\n".join(context_parts)


def check_relevance(doc_ids: List[str], qrels: Dict[str, int], k: int) -> Dict[str, Any]:
    """Check if top-k contains relevant documents."""
    top_k = doc_ids[:k] if k > 0 else []
    
    relevant_in_top_k = sum(1 for d in top_k if qrels.get(d, 0) > 0)
    total_relevant = sum(1 for d, r in qrels.items() if r > 0)
    
    # Reciprocal rank of first relevant doc
    rr = 0.0
    for i, d in enumerate(top_k):
        if qrels.get(d, 0) > 0:
            rr = 1.0 / (i + 1)
            break
    
    return {
        "relevant_in_top_k": relevant_in_top_k,
        "total_relevant": total_relevant,
        "recall_at_k": relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0,
        "reciprocal_rank": rr,
        "has_relevant": relevant_in_top_k > 0
    }


def evaluate_query(
    qid: str,
    query: str,
    doc_ids: List[str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, int],
    k_values: List[int],
    generator: GenerationOperation,
    model: str,
    system_prompt: str
) -> Dict[str, Any]:
    """Evaluate a single query across all k values."""
    results = {"qid": qid, "query": query, "shots": {}}
    
    for k in k_values:
        # Build context
        context = build_context(doc_ids, corpus, k)
        
        # Build prompt
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"
        
        # Generate answer - use config defaults
        gen_result = generator.execute(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens
        )
        
        answer = gen_result.get("answer", "")
        
        # Check relevance metrics
        rel_metrics = check_relevance(doc_ids, qrels, k)
        
        results["shots"][k] = {
            "k": k,
            "answer": answer,
            "context_length": len(context),
            "latency_ms": gen_result.get("latency_ms", 0),
            **rel_metrics
        }
    
    return results


def load_checkpoint(checkpoint_file: Path) -> tuple:
    """Load checkpoint if exists. Returns (completed_qids, results, needs_retry)."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        completed = set(data.get("completed_qids", []))
        results = data.get("results", [])
        
        # Check for entries with empty answers that need retry
        needs_retry = set()
        results_by_qid = {}
        
        for r in results:
            qid = r.get("qid")
            results_by_qid[qid] = r
            
            # Check if any k-shot has empty answer
            has_empty = False
            for k, shot in r.get("shots", {}).items():
                answer = shot.get("answer", "")
                if not answer or answer == "":
                    has_empty = True
                    break
            
            if has_empty:
                needs_retry.add(qid)
        
        # Remove entries that need retry from completed
        actually_completed = completed - needs_retry
        
        print(f"[07_rag] Resuming from checkpoint: {len(completed)} total, {len(needs_retry)} need retry")
        return actually_completed, results, needs_retry, results_by_qid
    
    return set(), [], set(), {}


def save_checkpoint(checkpoint_file: Path, completed_qids: Set[str], results: List[Dict], cfg: Dict):
    """Save checkpoint with current progress."""
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "completed_qids": list(completed_qids),
            "results": results,
            "config": cfg,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step 7: K-Shot RAG Evaluation")
    parser.add_argument("--corpus_path", required=True, help="Path to BEIR dataset")
    parser.add_argument("--run_path", default=None, help="Path to fused .res file")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--shots", default=",".join(map(str, config.evaluation.default_k_shots)),
                        help="Comma-separated k values")
    parser.add_argument("--model", default=config.models.lm_studio.default_model, help="LLM model")
    parser.add_argument("--limit", type=int, default=None, help="Limit queries (for testing)")
    parser.add_argument("--batch_size", type=int, default=10, help="Checkpoint every N queries")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start fresh")
    args = parser.parse_args()
    
    # Detect dataset from corpus_path
    dataset = detect_dataset(args.corpus_path)
    
    # Setup paths
    data_dir = config.project_root / "data" / dataset
    run_path = args.run_path or str(data_dir / "fused" / f"wcombsum_{config.qpp.default_method.lower()}.res")
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract fusion method from run path
    fusion_method = Path(run_path).stem
    model_safe = get_model_safe_name(args.model)
    
    # Files named: {fusion}__{model}.json
    checkpoint_file = output_dir / f"checkpoint_{fusion_method}__{model_safe}.json"
    results_file = output_dir / f"{fusion_method}__{model_safe}.json"
    
    print(f"[07_rag] Dataset: {dataset}")
    print(f"[07_rag] Fusion: {fusion_method}")
    print(f"[07_rag] Model: {args.model}")
    print(f"[07_rag] Checkpoint: {checkpoint_file}")
    print(f"[07_rag] Results: {results_file}")
    
    # Parse k values
    k_values = [int(k.strip()) for k in args.shots.split(",")]
    print(f"[07_rag] K values: {k_values}")
    
    # Load checkpoint
    if args.fresh and checkpoint_file.exists():
        os.remove(checkpoint_file)
        print("[07_rag] Fresh start - removed old checkpoint")
    
    completed_qids, all_results, needs_retry, results_by_qid = load_checkpoint(checkpoint_file)
    
    # Load data
    corpus = load_corpus(args.corpus_path)
    queries = load_queries(args.corpus_path)
    qrels = load_qrels(args.corpus_path)
    run = load_run(run_path)
    
    # Apply limit
    if args.limit:
        query_ids = list(queries.keys())[:args.limit]
        queries = {qid: queries[qid] for qid in query_ids}
        print(f"[07_rag] Limited to {len(queries)} queries")
    
    # Pending = not completed OR needs retry
    pending_queries = {qid: q for qid, q in queries.items() 
                       if qid not in completed_qids or qid in needs_retry}
    print(f"[07_rag] Pending: {len(pending_queries)} queries")
    print(f"[07_rag]   - New: {len(pending_queries) - len(needs_retry)}")
    print(f"[07_rag]   - Retry (empty answers): {len(needs_retry)}")
    
    if not pending_queries:
        print("[07_rag] All queries already completed!")
    else:
        # Initialize generator
        generator = GenerationOperation()
        
        # System prompt from config
        system_prompt = config.generation.system_prompt
        
        # Config for checkpoint
        cfg = {
            "fusion_method": fusion_method,
            "model": args.model,
            "k_values": k_values,
            "run_path": run_path
        }
        
        # Evaluate
        print(f"\n[07_rag] Evaluating {len(pending_queries)} pending queries...")
        
        start_time = time.time()
        batch_count = 0
        
        for i, (qid, query) in enumerate(pending_queries.items()):
            if qid not in run:
                continue
            
            doc_ids = [d[0] for d in run[qid]]
            query_qrels = qrels.get(qid, {})
            
            is_retry = qid in needs_retry
            
            result = evaluate_query(
                qid=qid,
                query=query,
                doc_ids=doc_ids,
                corpus=corpus,
                qrels=query_qrels,
                k_values=k_values,
                generator=generator,
                model=args.model,
                system_prompt=system_prompt
            )
            
            # Check if we got valid answers
            has_empty = any(not shot.get("answer") for shot in result["shots"].values())
            if has_empty:
                print(f"  Warning: {qid}: Got empty answer, will retry next time")
                continue
            
            # If retry, update existing result; otherwise append
            if is_retry and qid in results_by_qid:
                for idx, r in enumerate(all_results):
                    if r.get("qid") == qid:
                        all_results[idx] = result
                        break
                needs_retry.discard(qid)
            else:
                all_results.append(result)
            
            completed_qids.add(qid)
            batch_count += 1
            
            # Checkpoint every batch_size queries
            if batch_count >= args.batch_size:
                save_checkpoint(checkpoint_file, completed_qids, all_results, cfg)
                elapsed = time.time() - start_time
                total_done = len(completed_qids)
                remaining = len(queries) - total_done
                eta_sec = (elapsed / total_done) * remaining if total_done > 0 else 0
                eta_min = eta_sec / 60
                retry_str = f" (retries left: {len(needs_retry)})" if needs_retry else ""
                print(f"  Done: {total_done}/{len(queries)} queries ({elapsed:.1f}s) | ETA: {eta_min:.1f}min{retry_str}")
                batch_count = 0
        
        # Final checkpoint
        save_checkpoint(checkpoint_file, completed_qids, all_results, cfg)
    
    # Aggregate metrics
    aggregated = {k: {"recall_at_k": [], "reciprocal_rank": [], "has_relevant": [], "latency_ms": []} for k in k_values}
    
    for result in all_results:
        for k_str, shot_result in result["shots"].items():
            k_int = int(k_str) if isinstance(k_str, str) else k_str
            if k_int in aggregated:
                aggregated[k_int]["recall_at_k"].append(shot_result.get("recall_at_k", 0.0))
                aggregated[k_int]["reciprocal_rank"].append(shot_result.get("reciprocal_rank", 0.0))
                aggregated[k_int]["has_relevant"].append(1 if shot_result.get("has_relevant", False) else 0)
                aggregated[k_int]["latency_ms"].append(shot_result.get("latency_ms", 0.0))
    
    # Compute averages
    metrics_by_k = {}
    for k in k_values:
        n = len(aggregated[k]["recall_at_k"])
        if n > 0:
            mean_recall = sum(aggregated[k]["recall_at_k"]) / n
            mean_rr = sum(aggregated[k]["reciprocal_rank"]) / n
            hit_rate = sum(aggregated[k]["has_relevant"]) / n
            avg_latency = sum(aggregated[k]["latency_ms"]) / n
            
            metrics_by_k[str(k)] = {
                "recall_at_k": round(mean_recall * 100, 2),
                "mrr_at_k": round(mean_rr * 100, 2),
                "hit_rate": round(hit_rate * 100, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "n_queries": n
            }
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump({
            "_metadata": {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "script": "07_rag_eval.py",
                "fusion_method": fusion_method,
                "model": args.model,
                "dataset": dataset,
                "total_queries": len(all_results),
                "k_values": k_values,
                "qa_metrics_computed": False,
                "qa_metrics_list": []
            },
            "config": {
                "model": args.model,
                "k_values": k_values,
                "run_path": run_path
            },
            "summary": {
                "total_queries": len(all_results),
                "k_values": k_values,
                "metrics_by_k": metrics_by_k
            },
            "results": all_results
        }, f, indent=2)
    
    # Clean up checkpoint if complete
    if len(completed_qids) >= len(queries) and checkpoint_file.exists():
        os.remove(checkpoint_file)
        print("[07_rag] Removed checkpoint (complete)")
    
    print(f"\n=== Step 7 Complete ===")
    print(f"Results: {results_file}")
    print("\nSummary (metrics_by_k):")
    for k_str, metrics in sorted(metrics_by_k.items(), key=lambda x: int(x[0])):
        print(f"  k={k_str}: Recall@k={metrics['recall_at_k']:.2f}%, MRR@k={metrics['mrr_at_k']:.2f}%, HitRate={metrics['hit_rate']:.2f}%")


if __name__ == "__main__":
    main()
