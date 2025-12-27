#!/usr/bin/env python3
"""
Incoming: TREC run files --- {ranked lists, .res}
Processing: per-query min-max normalization --- {1 job: normalization}
Outgoing: normalized run files --- {ranked lists, .norm.res}
"""

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def _qid_sort_key(qid: str):
    """Sort numeric qids before string qids."""
    try:
        return (0, int(qid))
    except ValueError:
        return (1, qid)


def _load_run(res_path: Path) -> Tuple[Dict[str, List[Tuple[str, float, int]]], str]:
    """Load a run file into per-query lists and return (runs, run_tag)."""
    runs = defaultdict(list)
    run_tag = None
    with res_path.open("r", encoding="utf-8") as src:
        for line in src:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docno, rank, score, tag = parts[:6]
            run_tag = run_tag or tag
            runs[qid].append((docno, float(score), int(rank)))
    return runs, (run_tag or res_path.stem)


def _minmax_normalize(rows: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
    """Min-max normalize scores for a query."""
    if not rows:
        return []
    
    scores = [score for _, score, _ in rows]
    min_score = min(scores)
    max_score = max(scores)
    denom = max_score - min_score if max_score > min_score else 1.0
    
    normalized = []
    for docno, score, rank in rows:
        normalized.append((docno, (score - min_score) / denom, rank))
    return normalized


def _normalize_run(res_path: Path, force: bool = False) -> bool:
    """Normalize a single run file; return True if output written."""
    if res_path.suffix != ".res" or res_path.name.endswith(".norm.res"):
        return False
    
    norm_path = res_path.with_suffix(".norm.res")
    if norm_path.exists() and not force:
        print(f"[normalize] SKIP {norm_path.name} (exists)")
        return False
    
    runs, run_tag = _load_run(res_path)
    if not runs:
        print(f"[normalize] WARN {res_path.name}: no valid entries")
        return False
    
    with norm_path.open("w", encoding="utf-8") as dst:
        for qid in sorted(runs.keys(), key=_qid_sort_key):
            normalized = _minmax_normalize(runs[qid])
            normalized = sorted(normalized, key=lambda x: x[1], reverse=True)
            for rank, (docno, score, _) in enumerate(normalized, start=1):
                dst.write(f"{qid} Q0 {docno} {rank} {score:.6f} {run_tag}\n")
    
    print(f"[normalize] WROTE {norm_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Normalize .res run files to .norm.res")
    parser.add_argument("--runs_dir", required=True, help="Directory with .res files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .norm.res files")
    parser.add_argument("--pattern", default="*.res", help="Glob pattern for run files")
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.exists():
        raise SystemExit(f"[normalize] runs_dir not found: {runs_dir}")
    
    paths = sorted(runs_dir.glob(args.pattern))
    if not paths:
        print(f"[normalize] No files match {args.pattern} in {runs_dir}")
        return
    
    written = 0
    for res_path in paths:
        written += int(_normalize_run(res_path, args.force))
    
    print(f"[normalize] Completed: {written} files written in {runs_dir}")


if __name__ == "__main__":
    main()

