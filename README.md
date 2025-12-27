# QPP-Fusion-RAG

Query Performance Prediction Guided Retrieval Fusion for RAG.

## Research Objective

Investigate whether QPP can guide retriever fusion to improve RAG downstream performance. Compare heuristic fusion (CombSUM, CombMNZ, RRF) with QPP-weighted and learned fusion strategies.

## Pipeline

```
┌────────────┐   ┌────────────┐   ┌────────────┐
│ 01_index   │──▶│ 02_retrieve│──▶│  03_qpp    │
│ Build Idx  │   │ 5 Retrievers   │ 13 QPP     │
└────────────┘   └────────────┘   └─────┬──────┘
                                        │
                        ┌───────────────┘
                        ▼
               ┌────────────────┐
               │ 04_train_fusion│  ← Training (supervised)
               │ LightGBM + MLP │
               └───────┬────────┘
                       │
                       ▼
               ┌────────────────┐
               │  05_fusion     │  ← Inference
               │ Apply Weights  │
               └───────┬────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌────────────────┐        ┌────────────────┐
│ 06_eval_fusion │        │  07_rag_eval   │
│ IR Metrics     │        │ RAG End-Task   │
│ NDCG, Recall   │        │ LLM Generation │
└────────────────┘        └────────────────┘
```

## Quick Start

```bash
# Full pipeline on BEIR-NQ
python scripts/01_index.py --corpus_path /data/beir/datasets/nq --indexes bm25,splade,bge
python scripts/02_retrieve.py --corpus_path /data/beir/datasets/nq
python scripts/03_qpp.py
python scripts/04_train_fusion.py --model all
python scripts/05_fusion.py --method all
python scripts/06_eval_fusion.py
python scripts/07_rag_eval.py --corpus_path /data/beir/datasets/nq --limit 100
```

## Results (BEIR-NQ)

| Method | Type | NDCG@10 | Improvement |
|--------|------|---------|-------------|
| Learned (multioutput) | LightGBM | 0.5759 | **+16.5%** |
| Learned (per_retriever) | LightGBM | 0.5733 | +16.0% |
| Learned (mlp) | Neural Net | 0.5490 | +11.1% |
| W-CombSUM (RSD) | QPP-weighted | 0.4958 | +0.3% |
| CombSUM | Baseline | 0.4941 | — |
| CombMNZ | Heuristic | 0.4832 | -2.2% |
| RRF | Heuristic | 0.4605 | -6.8% |

## Structure

```
QPP-Fusion-RAG/
├── scripts/                  # Pipeline scripts
│   ├── 01_index.py          # Build indexes (BM25, SPLADE, BGE)
│   ├── 02_retrieve.py       # Run 5 retrievers
│   ├── 03_qpp.py            # Compute 13 QPP scores
│   ├── 04_train_fusion.py   # Train fusion models
│   ├── 05_fusion.py         # Apply fusion methods
│   ├── 06_eval_fusion.py    # Evaluate IR metrics
│   └── 07_rag_eval.py       # RAG end-task evaluation
├── src/
│   ├── fusion.py            # Fusion methods (CombSUM, CombMNZ, RRF, weighted, learned)
│   ├── qpp.py               # QPP computation (13 methods via Java bridge)
│   ├── models/              # ML models for learned fusion
│   │   ├── lightgbm_models.py
│   │   └── mlp_model.py
│   └── retrievers/          # Retriever implementations
│       ├── bm25.py
│       ├── bge.py
│       ├── splade.py
│       └── ...
└── data/nq/                  # NQ experiment data
    ├── runs/                # Retrieval results (.res)
    ├── qpp/                 # QPP scores (.qpp)
    ├── models/              # Trained models (.pkl)
    ├── fused/               # Fused results
    └── results/             # RAG evaluation results
```

## Retrievers (5)

| Retriever | Type | Description |
|-----------|------|-------------|
| BM25 | Sparse | Classic term-matching |
| BGE | Dense | BAAI/bge-base-en-v1.5 embeddings |
| SPLADE | Sparse-Learned | Neural sparse expansion |
| BM25→TCT | Pipeline | BM25 + TCT-ColBERT rerank |
| BM25→MonoT5 | Pipeline | BM25 + MonoT5 rerank |

## QPP Methods (13)

| # | Method | Description |
|---|--------|-------------|
| 0 | SMV | Similarity Mean Variance |
| 1 | Sigma_max | Maximum Standard Deviation |
| 2 | Sigma(%) | Threshold-based Std Dev |
| 3 | NQC | Normalized Query Commitment |
| 4 | UEF | Utility Estimation Framework |
| 5 | RSD | Retrieval Score Distribution |
| 6 | QPP-PRP | QPP Pseudo-Relevance Pooling |
| 7 | WIG | Weighted Information Gain |
| 8 | SCNQC | Scaled Calibrated NQC |
| 9 | QV-NQC | Query Variant NQC |
| 10 | DM | Document Movement |
| 11 | NQA-QPP | Neural QA QPP |
| 12 | BERTQPP | BERT-based QPP |

## Fusion Methods (9)

**Unweighted:**
- CombSUM: `Σ Score_i(d,q)`
- CombMNZ: `|rankers returning d| × Σ Score_i(d,q)`
- RRF: `Σ 1/(k + rank_i(d,q))`

**QPP-Weighted:**
- W-CombSUM: `Σ QPP_i(q) × Score_i(d,q)`
- W-CombMNZ: `|rankers| × Σ QPP_i(q) × Score_i(d,q)`
- W-RRF: `Σ QPP_i(q) / (k + rank_i(d,q))`

**Learned (ML Models):**
- PerRetriever: Separate LightGBM per retriever
- MultiOutput: Single multi-output LightGBM
- MLP: Neural network with shared layers

## Dependencies

```
lightgbm
torch
pyterrier
pyserini
sentence-transformers
transformers
numpy
pandas
```

## Cache Configuration

All downloads stored on external disk:
```python
# Set before imports
export PYSERINI_CACHE=/Volumes/Disk-D/RAGit/L4-Ind_Proj/QPP-Fusion-RAG/cache/pyserini
export HF_HOME=/Volumes/Disk-D/RAGit/L4-Ind_Proj/QPP-Fusion-RAG/cache/huggingface
```
