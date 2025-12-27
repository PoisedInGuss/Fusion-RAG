# QPP-Guided Multi-Retriever Fusion for Retrieval-Augmented Generation: Technical Documentation

---

## 1. Project Overview

This research investigates whether Query Performance Prediction (QPP) scores can improve multi-retriever fusion for Retrieval-Augmented Generation (RAG) systems by dynamically weighting different retrievers based on predicted query difficulty.

### Research Hypothesis

QPP-weighted fusion methods will outperform unweighted baselines.

### Experimental Setup

- **Dataset**: BEIR Natural Questions (NQ) — 2,681,468 Wikipedia passages, 3,452 test queries
- **LLMs**: Liquid LFM2-1.2B and Qwen3-4B
- **Evaluation**: IR metrics (NDCG@10, Recall@10, MRR@10) and QA metrics (Exact Match, F1, Containment, Semantic Similarity)

---

## 2. Multi-Retriever Architecture

### 2.1 Retrievers Used

Five diverse retrievers were implemented:

| Retriever | Type | Description |
|-----------|------|-------------|
| BM25 | Sparse Lexical | Classic term-frequency matching via PyTerrier inverted index |
| SPLADE | Sparse Learned | Neural term expansion using SPLADE++ EnsembleDistil via Pyserini |
| BGE | Dense | BAAI/bge-base-en-v1.5 embeddings with FAISS index via Pyserini |
| BM25→TCT-ColBERT | Two-Stage | BM25 first-stage (100 candidates) + TCT-ColBERT v2 reranking |
| BM25→MonoT5 | Two-Stage | BM25 first-stage (100 candidates) + MonoT5 neural reranking |

---

## 3. Query Performance Prediction (QPP)

### 3.1 QPP Methods Computed

Thirteen QPP methods were computed per (query, ranker) pair:

- SMV
- Sigma_max
- Sigma (%)
- NQC
- UEF
- RSD
- QPP-PRP
- WIG
- SCNQC
- QV-NQC
- DM
- NQA-QPP
- BERTQPP

### 3.2 QPP Analysis Findings

- **Best Predictor**: RSD (Retrieval Score Deviation) alone (Correlation with NDCG: +0.200)
- Only RSD, UEF, and DM had valid positive correlations with NDCG
- WIG, SCNQC, and BERTQPP had zero variance (constant values), making them useless
- Sigma_max had a NaN correlation (also useless)
- **Harmful Predictor**: NQA-QPP (Negative correlation: -0.285)
- **Key Finding**: RSD alone captures nearly all useful signal; other methods add noise, have negative correlation, or are constant/invalid

---

## 4. Fusion Methods

### 4.1 Baseline Methods (Unweighted)

CombSUM, CombMNZ, RRF

### 4.2 QPP-Weighted Methods (Heuristic)

W-CombSUM, W-CombMNZ, W-RRF, all using RSD per retriever for weighting \(w_i(q)\)

### 4.3 Learned Fusion Methods (ML Models)

- Per-Retriever LightGBM (5 independent regression models)
- Multi-Output LightGBM (Single model, 5 outputs)
- MLP Neural Network (5 features: RSD only)
- W-CombSUM Learned

---

## 5. ML Model Development

### 5.1 Training Strategy

*(Section to be documented)*

### 5.2 MLP Architecture Bug Fix

The original MLP used MSE Loss with Softmax output, which was mathematically incorrect. The solution was changing to Soft Cross-Entropy Loss, which resulted in NDCG improvement jumping from ~0% to +10.3%.

### 5.3 Feature Selection

Simplifying the MLP to use RSD only (5 features) maintained performance (+10.1% NDCG improvement), capturing 97% of the predictive signal with 92% fewer features than using all 13 QPP methods (65 features).

---

## 6. RAG Evaluation Pipeline

### 6.1 Evaluation Process

The process involves retrieving top-\(k\) documents, constructing a prompt, generating an answer using an LLM (via LM Studio API), and computing metrics.

### 6.2 LLM Configuration

- **Liquid LFM2-1.2B**: 1.2B Parameters, 128K Context Window
- **Qwen3-4B**: 4B Parameters, 256K Context Window

---

## 7. Quality Metrics

### 7.1 Information Retrieval Metrics

NDCG@k, Recall@k, MRR@k, Hit Rate

### 7.2 Question Answering Metrics

Exact Match (EM), F1 Score, Containment, Semantic Similarity

### 7.3 Embedding Model Analysis

Quantized models (Gemma 4-bit) were found to introduce systematic downward bias in Semantic Similarity scores (46.1% vs 74.8% for BGE-small-en-v1.5). 

**Conclusion**: Use BGE-small for production evaluation.

---

## 8. Key Research Findings

### 8.1 Fusion Method Performance (LFM2-1.2B)

The Learned Per-Retriever LightGBM method achieved the best F1 Score of 23.20%, representing a +2.38% improvement over the CombSUM baseline (20.82%).

### 8.2 Optimal Context Length

- The optimal number of retrieved documents for RAG context (\(k\)) was **k=1** (F1 Score: 23.2%)
- Using more documents (e.g., k=10) resulted in slight degradation (F1 Score: 21.5%)

### 8.3 Research Questions Answered

- Learned methods significantly beat heuristic QPP (+2.05% F1)
- RSD alone is the best QPP method
- The optimal context length for RAG is k=1

---

## 9. Current Experimental Status

### 9.1 Completed and In-Progress Evaluations (LFM2-1.2B)

| Fusion Method | RAG Complete | QA Metrics |
|---------------|:------------:|:----------:|
| CombSUM | ✓ | ✓ |
| CombMNZ | ✓ | ✓ |
| RRF | ✓ | ✓ |
| W-CombSUM RSD | ✓ | ✓ |
| W-CombMNZ RSD | ✓ | ✓ |
| W-RRF RSD | ✓ | ✓ |
| Learned Per-Retriever | ✓ | ✓ |
| Learned Multi-Output | ✓ | ✓ |
| Learned MLP (RSD-only) | ✓ | ✓ |
| W-CombSUM Learned | ✓ | ✓ |

### 9.2 Qwen3-4B Evaluations

| Fusion Method | Status |
|---------------|--------|
| CombSUM | ✓ Complete |
| W-CombSUM RSD | ✓ Complete |
| Learned Per-Retriever | In Progress |

---

**Document Version**: 1.0  
**Last Updated**: 5 December 2025
