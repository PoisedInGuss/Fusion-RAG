"""
Incoming: none --- {none}
Processing: package initialization --- {1 job: exports}
Outgoing: submodules --- {Python modules}

QPP-Fusion-RAG: Query Performance Prediction Guided Retrieval Fusion

Clean standalone implementation for ECIR paper reproduction.
Uses research-grade packages: ir_measures, ranx, HuggingFace evaluate.
"""

import os

__version__ = "2.0.0"

_SKIP_IMPORTS = os.environ.get("SRC_SKIP_PACKAGE_IMPORTS", "").lower() in ("1", "true", "yes")

if not _SKIP_IMPORTS:
    # Data utilities (centralized loaders)
    from .data_utils import (
        LazyCorpus,
        load_corpus,
        load_queries,
        load_qrels,
        load_run_file,
        load_run_as_dict,
        load_qpp_scores,
        get_model_safe_name,
        detect_dataset,
    )

    # QPP Bridge (13 methods via Java - STRICT: no fallback)
    from .qpp import QPPBridge, compute_qpp_for_res_file

    # LM Studio Generation
    from .generation import GenerationOperation

    # Evaluation (ir_measures + HuggingFace evaluate)
    from .evaluation import IREvaluator, QAEvaluator, evaluate_run, compute_qa_metrics

    # Fusion (ranx-based)
    from .fusion import combsum, combmnz, rrf, weighted_combsum, weighted_rrf, learned_fusion

    __all__ = [
        # Data utilities
        "LazyCorpus",
        "load_corpus",
        "load_queries",
        "load_qrels",
        "load_run_file",
        "load_run_as_dict",
        "load_qpp_scores",
        "get_model_safe_name",
        "detect_dataset",
        # QPP
        "QPPBridge",
        "compute_qpp_for_res_file",
        # Generation
        "GenerationOperation",
        # Evaluation
        "IREvaluator",
        "QAEvaluator",
        "evaluate_run",
        "compute_qa_metrics",
        # Fusion
        "combsum",
        "combmnz",
        "rrf",
        "weighted_combsum",
        "weighted_rrf",
        "learned_fusion",
    ]
else:
    __all__ = ["__version__"]
