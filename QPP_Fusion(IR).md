
1

Automatic Zoom
Beyond Correlations: A Downstream Evaluation
Framework for Query Performance Prediction
Anonymous Author(s)
Paper under double-blind review
Abstract. The standard practice of query performance prediction (QPP)
evaluation is to measure a set-level correlation between the estimated
retrieval qualities and the true ones. However, neither this correlation-
based evaluation measure quantifies QPP effectiveness at the level of
individual queries, nor does this connect to a downstream application,
meaning that QPP methods yielding high correlation values may not
find a practical application in query-specific decisions in an IR pipeline.
In this paper, we propose a downstream-focussed evaluation framework
where a distribution of QPP estimates across a list of top-documents
retrieved with several rankers is used as priors for IR fusion. While on
the one hand, a distribution of these estimates closely matching that of
the true retrieval qualities indicates the quality of the predictor, their
usage as priors on the other hand indicates a predictor’s ability to make
informed choices in an IR pipeline. Our experiments firstly establish the
importance of QPP estimates in weighted IR fusion, yielding significant
improvements of over 4.5% over unweighted CombSUM and RRF fusion
strategies, and secondly, reveal new insights that the downstream effec-
tiveness of QPP does not correlate well with the standard correlation-
based QPP evaluation.
Keywords: Query Performance Prediction · Downstream QPP Eval-
uation · IR Fusion · Weighted CombSUM
1 Introduction
Query Performance Prediction (QPP) methods aim to estimate the retrieval ef-
fectiveness of a ranked list without access to relevance judgments [19,21,18,2].
Unsupervised QPP approaches mostly leverage the retrieval score distribution
of the top-ranked documents obtained with the original query [21,18,17] or its
variants [27,5], along with other characteristic features, such as term informa-
tiveness [28], co-occurrence [19] and semantics [6]. Supervised approaches, on
the other hand, rely on the content of the query and the top ranked documents
to predict the retrieval quality [10,2,1].
In its standard form, the objective of the QPP task is to effectively distin-
guish well-performing queries from poorly performing ones for a given retrieval
model. Accordingly, its evaluation typically focuses on measuring the correla-
tion between the true retrieval effectiveness (e.g., AP or nDCG) and the pre-
dicted QPP scores. However, as noted by [9], existing QPP models are largely
2 Anonymous Author(s)
agnostic to both the underlying information retrieval (IR) model and the evalu-
ation metric. Consequently, the observed correlations are often highly sensitive
to these factors, i,e., observations are mostly not consistent across a range of
different ranking models and target IR metrics. Another major limitation of the
correlation-based QPP evaluation paradigm is its dependence on a set of queries,
in contrast to ad-hoc IR evaluation, which can be performed independently for
each query — specifically, by assessing whether the retrieved documents satisfy
that query’s information need. Consequently, QPP evaluation is inherently con-
ditioned on the query set, where the performance of one query is interpreted
relative to that of others, unlike per-query measures such as AP or nDCG that
evaluate retrieval quality in isolation.
Beyond the lack of robustness and per-query independence, the most criti-
cal shortcoming of existing QPP evaluation methodologies is their misalignment
with the downstream utility of QPP estimates—the very motivation behind the
task. To address this, we propose a downstream-aware QPP evaluation frame-
work. Given an input query and a set of available rankers, our QPP evaluation
approach first employs a QPP model to predict the expected retrieval quality
for each ranker. This formulation enables per-query evaluation of a QPP model
based on its ability to discriminate between strong and weak rankers, while
also facilitating a natural downstream application—using the predicted scores
as weights in an information retrieval fusion framework. If the predicted likeli-
hoods closely approximate the true retrieval performance, incorporating them as
priors in the fusion process should enhance overall retrieval effectiveness. Con-
sequently, a QPP model can be regarded as more effective than another if its
estimates, when employed as priors in an IR fusion algorithm, yields more effec-
tive retrieval quality than its competitor. In this way, we not only contribute a
downstream-oriented perspective on QPP evaluation but also demonstrate that
ad hoc retrieval performance can be enhanced through QPP-based fusion. The
source code for the proposed QPP evaluation framework and the QPP-based IR
fusion is made available for research purposes. 1
2 Downstream Evaluation of QPP Models on IR Fusion
QPP across multiple rankers. Standard QPP is formalized as a function
of the form φ(q, Lθ (q)) 7 → R, which given an input query q and a list of top
documents Lθ (q) retrieved with a single ranker θ predicts its quality. Instead of
using these estimates to distinguish between multiple queries, we rather employ
a QPP model φ to obtain predicted retrieval qualities across several IR models,
i.e., for a set of available rankers Θ = {θ1, . . . , θm}, a QPP model outputs a
distribution of estimated retrieval qualities across the rankers. Formally,
φ(q, LΘ (q)) 7 → Rm, where LΘ (q) = ∪θ∈Θ Lθ (q), (1)
1 https://anonymous.4open.science/r/QPP-Fusion-1FEC/README.md
