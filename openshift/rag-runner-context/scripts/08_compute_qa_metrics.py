#!/usr/bin/env python3
"""
Incoming: RAG results JSON, NQ gold answers --- {generated answers, gold answers}
Processing: QA metric computation --- {5 metrics: EM, F1, Containment, Semantic, LLM-Judge}
Outgoing: updated results JSON with QA metrics --- {JSON}

Step 8: Compute QA Metrics
--------------------------
Computes multiple QA metrics for RAG-generated answers using HuggingFace evaluate:
  - EM (Exact Match): Normalized string match (via evaluate.load("squad"))
  - F1: Token-level F1 score (via evaluate.load("squad"))
  - Containment: Does output contain gold answer?
  - Semantic: Cosine similarity using LM Studio embeddings (BGE-small)
  - LLM-Judge: LLM rates answer correctness (1-5)

STRICT: No manual fallbacks. Missing dependencies raise ImportError.

Usage:
    python scripts/08_compute_qa_metrics.py --results_file data/nq/results/wcombsum_rsd__liquid_lfm2-1.2b.json
    python scripts/08_compute_qa_metrics.py --results_file ... --metrics em,f1,containment,semantic
    python scripts/08_compute_qa_metrics.py --results_file ... --metrics all --judge_model liquid/lfm2-1.2b
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config FIRST - sets up cache environment
from src.config import config

# STRICT: Use research-grade QA evaluation
from src.evaluation import QAEvaluator


# =============================================================================
# QA Metrics via HuggingFace evaluate
# =============================================================================

# Shared evaluator instance (lazy loaded)
_qa_evaluator: Optional[QAEvaluator] = None


def _get_qa_evaluator() -> QAEvaluator:
    """Get or create shared QA evaluator."""
    global _qa_evaluator
    if _qa_evaluator is None:
        _qa_evaluator = QAEvaluator(metrics=["em", "f1", "containment"])
    return _qa_evaluator


def compute_em(prediction: str, gold_answers: List[str]) -> float:
    """Exact Match using HuggingFace evaluate (squad metric)."""
    evaluator = _get_qa_evaluator()
    result = evaluator.evaluate(prediction, gold_answers)
    return result.get("em", 0.0)


def compute_f1(prediction: str, gold_answers: List[str]) -> float:
    """Token F1 using HuggingFace evaluate (squad metric)."""
    evaluator = _get_qa_evaluator()
    result = evaluator.evaluate(prediction, gold_answers)
    return result.get("f1", 0.0)


def compute_containment(prediction: str, gold_answers: List[str]) -> float:
    """Containment: 1 if any gold answer is contained in the prediction."""
    evaluator = _get_qa_evaluator()
    result = evaluator.evaluate(prediction, gold_answers)
    return result.get("containment", 0.0)


# =============================================================================
# Semantic Similarity (LM Studio Embeddings - not in HuggingFace evaluate)
# =============================================================================

def get_embeddings_batch(
    texts: List[str], 
    model: str = None, 
    batch_size: int = 32, 
    session: Optional[requests.Session] = None
) -> List[Optional[List[float]]]:
    """Get embeddings for multiple texts in batches from LM Studio (OPTIMIZED: connection pooling)."""
    # Get settings from config
    embed_url = f"{config.models.lm_studio.base_url.rstrip('/v1')}/v1/embeddings"
    model = model or config.models.lm_studio.embed_model
    
    all_embeddings = []
    
    # Use provided session or create one (connection pooling)
    if session is None:
        session = requests.Session()
        close_session = True
    else:
        close_session = False
    
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = session.post(
                    embed_url,
                    json={"model": model, "input": batch},
                    timeout=config.models.lm_studio.timeout_seconds
                )
                if response.status_code == 200:
                    data = response.json()['data']
                    data.sort(key=lambda x: x['index'])
                    all_embeddings.extend([d['embedding'] for d in data])
                else:
                    raise RuntimeError(f"LM Studio embedding failed: {response.status_code}")
            except requests.exceptions.ConnectionError:
                raise RuntimeError(f"LM Studio not running at {config.models.lm_studio.base_url}")
    finally:
        if close_session:
            session.close()
    
    return all_embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_semantic_sim_batch(
    predictions: List[str],
    gold_list: List[List[str]],
    model: str = None,
    session: Optional[requests.Session] = None
) -> List[float]:
    """Batch semantic similarity computation using LM Studio embeddings (OPTIMIZED: connection pooling)."""
    model = model or config.models.lm_studio.embed_model
    
    print(f"  [Embeddings] Getting embeddings for {len(predictions)} predictions...")
    pred_embeddings = get_embeddings_batch(predictions, model=model, session=session)
    
    # Get unique gold answers
    all_golds = set()
    for golds in gold_list:
        all_golds.update(golds)
    all_golds = list(all_golds)
    
    print(f"  [Embeddings] Getting embeddings for {len(all_golds)} unique gold answers...")
    gold_embeddings = get_embeddings_batch(all_golds, model=model, session=session)
    gold_emb_map = {g: e for g, e in zip(all_golds, gold_embeddings) if e is not None}
    
    results = []
    for pred_emb, golds in zip(pred_embeddings, gold_list):
        if pred_emb is None:
            results.append(-1.0)
            continue
        
        best_sim = 0.0
        for g in golds:
            if g in gold_emb_map:
                sim = cosine_similarity(pred_emb, gold_emb_map[g])
                best_sim = max(best_sim, sim)
        
        results.append(best_sim)
    
    return results


# =============================================================================
# LLM-as-Judge (Custom - not in HuggingFace evaluate)
# =============================================================================

def compute_llm_judge(
    question: str,
    prediction: str,
    gold_answers: List[str],
    model: str = None
) -> float:
    """LLM-as-Judge: Rate answer correctness 1-5 using LM Studio."""
    model = model or config.models.lm_studio.default_model
    chat_url = f"{config.models.lm_studio.base_url}/chat/completions"
    
    gold_str = ", ".join(gold_answers[:3])
    
    prompt = f"""Rate the following answer on a scale of 1-5 for correctness.

Question: {question}
Gold Answer(s): {gold_str}
Generated Answer: {prediction}

Rating criteria:
1 = Completely wrong or irrelevant
2 = Partially related but incorrect
3 = Contains correct information but incomplete
4 = Mostly correct with minor issues
5 = Fully correct and complete

Respond with ONLY a single number (1-5)."""

    try:
        response = requests.post(
            chat_url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 5
            },
            timeout=config.models.lm_studio.timeout_seconds
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"LLM Judge failed: {response.status_code}")
        
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip()
        
        match = re.search(r'[1-5]', answer)
        if match:
            return float(match.group())
        raise ValueError(f"LLM did not return valid rating: {answer}")
        
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"LM Studio not running at {config.models.lm_studio.base_url}")


# =============================================================================
# Gold Answer Loading
# =============================================================================

def load_nq_gold_answers(cache_dir: Path) -> Dict[str, List[str]]:
    """Load gold answers from NQ dataset."""
    answers_file = cache_dir / "nq_gold_answers.json"
    
    if answers_file.exists():
        print(f"Loading cached gold answers from {answers_file}")
        with open(answers_file) as f:
            return json.load(f)
    
    print("Downloading NQ gold answers from HuggingFace...")
    
    # STRICT: Fail if datasets not installed
    from datasets import load_dataset
    
    hf_cache = config.cache_root / "huggingface" / "datasets"
    hf_cache.mkdir(parents=True, exist_ok=True)
    
    print("Downloading NQ validation set (~40GB) to cache...")
    dataset = load_dataset(
        "natural_questions", 
        "default", 
        split="validation",
        cache_dir=str(hf_cache),
        trust_remote_code=True
    )
    
    print(f"Processing {len(dataset)} examples...")
    gold_answers = {}
    
    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(dataset)}...")
            
        question = example['question']['text'].strip()
        short_answers = []
        annotations = example['annotations']
        
        for sa in annotations['short_answers']:
            if sa['text']:
                for answer in sa['text']:
                    if answer and answer not in short_answers:
                        short_answers.append(answer)
        
        if short_answers:
            q_norm = question.lower().strip('?').strip()
            gold_answers[q_norm] = short_answers
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(answers_file, 'w') as f:
        json.dump(gold_answers, f, indent=2)
    
    print(f"Cached {len(gold_answers)} gold answers to {answers_file}")
    return gold_answers


def load_hotpotqa_gold_answers(corpus_path: Path) -> Dict[str, List[str]]:
    """Load HotpotQA gold answers from queries.jsonl metadata."""
    gold_file = corpus_path / "hotpotqa_gold_answers.json"
    
    if gold_file.exists():
        print(f"Loading HotpotQA gold answers from {gold_file}")
        with open(gold_file) as f:
            return json.load(f)
    
    # Fallback: extract from queries.jsonl
    print("Extracting HotpotQA gold answers from queries.jsonl...")
    queries_file = corpus_path / "queries.jsonl"
    gold_answers = {}
    
    with open(queries_file) as f:
        for line in f:
            q = json.loads(line)
            query_text = q['text']
            answer = q.get('metadata', {}).get('answer')
            
            if answer:
                q_norm = query_text.lower().strip('?').strip()
                gold_answers[q_norm] = [answer]
    
    print(f"Extracted {len(gold_answers)} HotpotQA gold answers")
    return gold_answers


def match_query_to_gold(query: str, gold_answers: Dict[str, List[str]]) -> Optional[List[str]]:
    """Match query text to gold answers."""
    q_norm = query.lower().strip('?').strip()
    
    if q_norm in gold_answers:
        return gold_answers[q_norm]
    
    for gold_q, answers in gold_answers.items():
        if q_norm in gold_q or gold_q in q_norm:
            return answers
    
    return None


# =============================================================================
# Main Processing
# =============================================================================

def compute_qa_metrics_for_file(
    results_file: Path,
    gold_answers: Dict[str, List[str]],
    metrics: List[str],
    judge_model: str = None,
    embed_model: str = None
) -> Dict:
    """Compute selected QA metrics for all results in a file."""
    # Get defaults from config
    embed_model = embed_model or config.models.lm_studio.embed_model
    judge_model = judge_model or config.models.lm_studio.default_model
    
    print(f"\nProcessing: {results_file.name}")
    print(f"Metrics: {metrics}")
    
    with open(results_file) as f:
        data = json.load(f)
    
    agg = defaultdict(lambda: defaultdict(lambda: {'sum': 0, 'count': 0}))
    
    matched_count = 0
    all_entries = []
    
    for entry in data['results']:
        query = entry['query']
        gold = match_query_to_gold(query, gold_answers)
        
        if gold is None:
            continue
        
        matched_count += 1
        
        for k_str, shot in entry['shots'].items():
            generated = shot.get('answer', '')
            if not generated:
                continue
            
            all_entries.append({
                'query': query,
                'generated': generated,
                'gold': gold,
                'k': int(k_str),
                'shot': shot
            })
    
    print(f"  Matched {matched_count}/{len(data['results'])} queries to gold answers")
    print(f"  Processing {len(all_entries)} (query, k) pairs...")
    
    # Batch semantic similarity if requested
    if 'semantic' in metrics:
        print(f"  Computing semantic similarity (LM Studio embeddings: {embed_model})...")
        with requests.Session() as session:
            predictions = [e['generated'] for e in all_entries]
            gold_list = [e['gold'] for e in all_entries]
            sem_scores = compute_semantic_sim_batch(predictions, gold_list, model=embed_model, session=session)
            for i, e in enumerate(all_entries):
                e['semantic'] = sem_scores[i]
    
    # Process each entry
    total = len(all_entries)
    for i, e in enumerate(all_entries):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{total}...")
        
        shot = e['shot']
        generated = e['generated']
        gold = e['gold']
        k = e['k']
        
        if 'em' in metrics:
            em = compute_em(generated, gold)
            shot['em'] = em
            agg[k]['em']['sum'] += em
            agg[k]['em']['count'] += 1
        
        if 'f1' in metrics:
            f1 = compute_f1(generated, gold)
            shot['f1'] = f1
            agg[k]['f1']['sum'] += f1
            agg[k]['f1']['count'] += 1
        
        if 'containment' in metrics:
            cont = compute_containment(generated, gold)
            shot['containment'] = cont
            agg[k]['containment']['sum'] += cont
            agg[k]['containment']['count'] += 1
        
        if 'semantic' in metrics:
            sem = e.get('semantic', -1)
            if sem >= 0:
                shot['semantic'] = sem
                agg[k]['semantic']['sum'] += sem
                agg[k]['semantic']['count'] += 1
        
        if 'llm_judge' in metrics and judge_model:
            score = compute_llm_judge(e['query'], generated, gold, judge_model)
            shot['llm_judge'] = score
            agg[k]['llm_judge']['sum'] += score
            agg[k]['llm_judge']['count'] += 1
        
        shot['gold_answers'] = gold
    
    # Compute averages
    qa_summary = {}
    for k in sorted(agg.keys()):
        qa_summary[str(k)] = {}
        for metric_name, vals in agg[k].items():
            if vals['count'] > 0:
                avg = vals['sum'] / vals['count']
                if metric_name in ['em', 'f1', 'containment', 'semantic']:
                    avg *= 100
                qa_summary[str(k)][metric_name] = round(avg, 2)
        qa_summary[str(k)]['n_matched'] = agg[k]['em']['count'] if 'em' in agg[k] else matched_count
    
    # Update file
    if 'summary' not in data:
        data['summary'] = {}
    
    existing_qa = data['summary'].get('qa_metrics_by_k', {})
    for k in qa_summary:
        if k not in existing_qa:
            existing_qa[k] = {}
        existing_qa[k].update(qa_summary[k])
    
    data['summary']['qa_metrics_by_k'] = existing_qa
    data['_metadata']['qa_metrics_computed'] = True
    
    computed = set(data['_metadata'].get('qa_metrics_list', []))
    computed.update(metrics)
    data['_metadata']['qa_metrics_list'] = list(computed)
    
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return qa_summary


def main():
    parser = argparse.ArgumentParser(description="Compute QA Metrics (HuggingFace evaluate)")
    parser.add_argument("--results_file", required=True, help="Results JSON file")
    parser.add_argument("--dataset", default="nq", choices=config.datasets.supported, 
                        help="Dataset name (for gold answer loading)")
    parser.add_argument("--cache_dir", default=None, help="Cache directory for gold answers")
    parser.add_argument("--metrics", default="em,f1,containment",
                        help="Comma-separated metrics: em,f1,containment,semantic,llm_judge,all")
    parser.add_argument("--judge_model", default=config.models.lm_studio.default_model,
                        help="LM Studio model for LLM-as-Judge")
    parser.add_argument("--embed_model", default=config.models.lm_studio.embed_model,
                        help="LM Studio embedding model")
    args = parser.parse_args()
    
    if args.metrics == 'all':
        metrics = ['em', 'f1', 'containment', 'semantic', 'llm_judge']
    else:
        metrics = [m.strip() for m in args.metrics.split(',')]
    
    print(f"Metrics to compute: {metrics}")
    print(f"Dataset: {args.dataset}")
    
    # Load gold answers based on dataset
    if args.dataset == "hotpotqa":
        corpus_path = config.project_root / "data" / "hotpotqa" / "BEIR-hotpotqa"
        gold_answers = load_hotpotqa_gold_answers(corpus_path)
    else:
        cache_dir = Path(args.cache_dir) if args.cache_dir else config.project_root / "data" / args.dataset / "cache"
        gold_answers = load_nq_gold_answers(cache_dir)
    
    print(f"Loaded {len(gold_answers)} gold answer mappings")
    
    results_path = Path(args.results_file)
    
    if results_path.is_file():
        files = [results_path]
    else:
        files = list(results_path.parent.glob(results_path.name))
    
    all_results = {}
    for f in files:
        if f.name.startswith('_') or f.name.startswith('checkpoint'):
            continue
        qa_metrics = compute_qa_metrics_for_file(f, gold_answers, metrics, args.judge_model, args.embed_model)
        all_results[f.stem] = qa_metrics
    
    print("\n" + "="*80)
    print("QA METRICS SUMMARY")
    print("="*80)
    
    for fname, metrics_data in all_results.items():
        print(f"\n{fname}:")
        
        header_parts = ['K']
        for m in metrics:
            if m == 'llm_judge':
                header_parts.append('Judge(1-5)')
            else:
                header_parts.append(m.upper() + '%')
        header_parts.append('N')
        
        print(f"{header_parts[0]:<5}", end='')
        for h in header_parts[1:]:
            print(f"{h:<12}", end='')
        print()
        print("-" * (5 + 12 * len(header_parts[1:])))
        
        for k, m in sorted(metrics_data.items(), key=lambda x: int(x[0])):
            print(f"{k:<5}", end='')
            for metric in metrics:
                val = m.get(metric, -1)
                if val >= 0:
                    print(f"{val:<12.2f}", end='')
                else:
                    print(f"{'N/A':<12}", end='')
            print(f"{m.get('n_matched', 0):<12}")


if __name__ == "__main__":
    main()
