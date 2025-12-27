#!/usr/bin/env python3
"""
Incoming: .norm.res files, .qpp files, trained models --- {runs, QPP, models}
Processing: fusion --- {9 methods: CombSUM, CombMNZ, RRF + weighted + learned}
Outgoing: fused .res files --- {TREC format}

Step 5: Apply Fusion
--------------------
Applies fusion strategies to combine multiple retriever runs.

Methods:
  combsum   - Sum of normalized scores
  combmnz   - CombSUM × number of rankers returning doc  
  rrf       - Reciprocal Rank Fusion
  wcombsum  - QPP-weighted CombSUM
  wcombmnz  - QPP-weighted CombMNZ
  wrrf      - QPP-weighted RRF
  learned   - ML model learned weights
  all       - Run all methods and output comparison

Usage:
    python scripts/05_fusion.py --method combsum
    python scripts/05_fusion.py --method wcombsum --qpp_model RSD
    python scripts/05_fusion.py --method learned --model_path data/nq/models/fusion_weights_model.pkl
    python scripts/05_fusion.py --method all  # Run all methods
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config first
from src.config import config

from src.fusion import (
    run_fusion,
    load_runs,
    load_qpp_scores,
    combsum,
    combmnz,
    rrf,
    weighted_combsum,
    weighted_combmnz,
    weighted_rrf,
    learned_fusion,
    write_runfile,
    get_qpp_index
)


def run_all_methods(
    runs_dir: Path,
    qpp_dir: Path,
    fused_dir: Path,
    qpp_model: str = None,
    rrf_k: int = None
):
    """Run all fusion methods and output comparison."""
    # Get defaults from config
    qpp_model = qpp_model or config.qpp.default_method
    rrf_k = rrf_k if rrf_k is not None else config.fusion.rrf_k
    
    print(f"[05_fusion] Running all fusion methods...")
    print(f"[05_fusion] QPP model for weighted: {qpp_model}")
    
    results = {}
    
    # Load data once
    runs = load_runs(str(runs_dir), use_normalized=True)
    print(f"[05_fusion] Loaded {len(runs)} rankers: {list(runs.keys())}")
    
    # Unweighted methods
    print("\n--- Unweighted Methods ---")
    
    # CombSUM
    print("[05_fusion] Running CombSUM...")
    fused = combsum(runs)
    output_path = fused_dir / "combsum.res"
    write_runfile(fused, str(output_path), "combsum")
    results["combsum"] = output_path
    
    # CombMNZ
    print("[05_fusion] Running CombMNZ...")
    fused = combmnz(runs)
    output_path = fused_dir / "combmnz.res"
    write_runfile(fused, str(output_path), "combmnz")
    results["combmnz"] = output_path
    
    # RRF
    print(f"[05_fusion] Running RRF (k={rrf_k})...")
    fused = rrf(runs, k=rrf_k)
    output_path = fused_dir / "rrf.res"
    write_runfile(fused, str(output_path), "rrf")
    results["rrf"] = output_path
    
    # Weighted methods (need QPP)
    if qpp_dir.exists():
        print("\n--- QPP-Weighted Methods ---")
        qpp_data = load_qpp_scores(str(qpp_dir))
        qpp_index = get_qpp_index(qpp_model)
        
        # W-CombSUM
        print(f"[05_fusion] Running W-CombSUM ({qpp_model})...")
        fused = weighted_combsum(runs, qpp_data, qpp_index)
        output_path = fused_dir / f"wcombsum_{qpp_model.lower()}.res"
        write_runfile(fused, str(output_path), f"wcombsum-{qpp_model.lower()}")
        results["wcombsum"] = output_path
        
        # W-CombMNZ
        print(f"[05_fusion] Running W-CombMNZ ({qpp_model})...")
        fused = weighted_combmnz(runs, qpp_data, qpp_index)
        output_path = fused_dir / f"wcombmnz_{qpp_model.lower()}.res"
        write_runfile(fused, str(output_path), f"wcombmnz-{qpp_model.lower()}")
        results["wcombmnz"] = output_path
        
        # W-RRF
        print(f"[05_fusion] Running W-RRF ({qpp_model})...")
        fused = weighted_rrf(runs, qpp_data, qpp_index, k=rrf_k)
        output_path = fused_dir / f"wrrf_{qpp_model.lower()}.res"
        write_runfile(fused, str(output_path), f"wrrf-{qpp_model.lower()}")
        results["wrrf"] = output_path
        
        # Learned fusion (for each available model)
        models_dir = fused_dir.parent / "models"
        learned_models = [
            ("per_retriever", models_dir / "fusion_per_retriever.pkl"),
            ("multioutput", models_dir / "fusion_multioutput.pkl"),
            ("mlp", models_dir / "fusion_mlp.pkl"),
        ]
        
        for model_name, model_file in learned_models:
            if model_file.exists():
                print(f"[05_fusion] Running Learned ({model_name})...")
                fused = learned_fusion(runs, qpp_data, str(model_file))
                output_path = fused_dir / f"learned_{model_name}.res"
                write_runfile(fused, str(output_path), f"learned-{model_name}")
                results[f"learned_{model_name}"] = output_path
            else:
                print(f"[05_fusion] Skipping learned ({model_name}) - model not found")
    else:
        print(f"[05_fusion] Skipping weighted methods (no QPP at {qpp_dir})")
    
    print(f"\n=== Generated {len(results)} fusion runs ===")
    for method, path in results.items():
        print(f"  {method}: {path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Multi-Method Fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  combsum   - Sum of normalized scores
  combmnz   - CombSUM × number of rankers returning doc
  rrf       - Reciprocal Rank Fusion
  wcombsum  - QPP-weighted CombSUM
  wcombmnz  - QPP-weighted CombMNZ
  wrrf      - QPP-weighted RRF
  learned   - ML model learned weights
  all       - Run all methods

Examples:
  python scripts/05_fusion.py --method combsum
  python scripts/05_fusion.py --method wcombsum --qpp_model NQC
  python scripts/05_fusion.py --method all
"""
    )
    parser.add_argument("--method", default="wcombsum",
                        choices=config.fusion.methods + ["all"],
                        help="Fusion method (default: wcombsum)")
    parser.add_argument("--dataset", default="nq", choices=config.datasets.supported,
                        help="Dataset name")
    parser.add_argument("--runs_dir", default=None, help="Directory with .norm.res files")
    parser.add_argument("--qpp_dir", default=None, help="Directory with .qpp files")
    parser.add_argument("--qpp_model", default=config.qpp.default_method, help="QPP model for weights")
    parser.add_argument("--model_path", default=None, help="Path to learned model")
    parser.add_argument("--output", default=None, help="Output fused run file")
    parser.add_argument("--rrf_k", type=int, default=config.fusion.rrf_k, help="RRF k constant")
    args = parser.parse_args()
    
    # Setup paths
    output_dir = config.project_root / "data" / args.dataset
    runs_dir = Path(args.runs_dir) if args.runs_dir else output_dir / "runs"
    qpp_dir = Path(args.qpp_dir) if args.qpp_dir else output_dir / "qpp"
    fused_dir = output_dir / "fused"
    
    os.makedirs(fused_dir, exist_ok=True)
    
    # Default model path
    model_path = args.model_path or str(output_dir / "models" / "fusion_weights_model.pkl")
    
    if args.method == "all":
        # Run all methods
        run_all_methods(
            runs_dir=runs_dir,
            qpp_dir=qpp_dir,
            fused_dir=fused_dir,
            qpp_model=args.qpp_model,
            rrf_k=args.rrf_k
        )
    else:
        # Run single method
        if args.output:
            output_path = args.output
        else:
            if args.method.startswith("w"):
                output_path = str(fused_dir / f"{args.method}_{args.qpp_model.lower()}.res")
            elif args.method == "learned":
                output_path = str(fused_dir / "learned.res")
            else:
                output_path = str(fused_dir / f"{args.method}.res")
        
        print(f"[05_fusion] Running {args.method}...")
        
        run_fusion(
            method=args.method,
            runs_dir=str(runs_dir),
            qpp_dir=str(qpp_dir) if qpp_dir.exists() else None,
            qpp_model=args.qpp_model,
            model_path=model_path if args.method == "learned" else None,
            output_path=output_path,
            rrf_k=args.rrf_k
        )
        
        print(f"\n=== Step 5 Complete ===")
        print(f"Fused run: {output_path}")


if __name__ == "__main__":
    main()
