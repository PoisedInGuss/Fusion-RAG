#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probFuse training & evaluation script

Train on a training set of runs + qrels (e.g., TREC-DL19) to estimate segment
probabilities for each ranker, then fuse test runs (e.g., TREC-DL20) and
evaluate.

Usage example:
python probfuse_eval.py \
  --train_res_path ./dl19_runs_norm \
  --test_res_path ./dl20_runs_norm \
  --train_qrels dl19_qrels.txt \
  --test_qrels dl20_qrels.txt \
  --topics dl20_topics.txt \
  --variant judged \
  --x 25 --L 100 \
  --output_run fused_probfuse.res \
  --save_results metrics_probfuse.csv
"""

import os
import math
import argparse
from collections import defaultdict
import pandas as pd
import pyterrier as pt

# if not pt.started():
#     pt.init()

def pos_to_segment(pos, L, x):
    """Map 1-indexed rank position to 1..x segment index."""
    if pos <= 0:
        pos = 1
    k = math.ceil((pos * x) / float(L))
    if k < 1:
        k = 1
    if k > x:
        k = x
    return int(k)


def load_runs(res_path):
    """
    Load .res or .norm.res run files from res_path into a dict: {ranker_name: DataFrame}.
    DataFrame columns: qid (string), iter, docno (string), rank (int), score (float), runid
    """
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".norm.res")]
    if not files:
        raise ValueError(f"No .res files found in {res_path}")
    for f in sorted(files):
        ranker = f.replace(".norm.res", "")
        df = pd.read_csv(os.path.join(res_path, f), sep=r"\s+",
                         names=["qid", "iter", "docno", "rank", "score", "runid"],
                         dtype={"qid": str, "docno": str})
        df["qid"] = df["qid"].astype(str)
        runs[ranker] = df
    return runs


def load_qrels_to_df(qrels_path):
    """
    Use pyterrier to read qrels into a DataFrame. Returns DataFrame and also
    a nested dict {qid: {docno: rel}} for quick lookups.
    """
    qrels_df = pt.io.read_qrels(qrels_path)
    # Ensure first three columns are qid, docno, label
    if len(qrels_df.columns) < 3:
        raise ValueError("qrels file must have at least three columns: qid docno label")
    cols = list(qrels_df.columns)
    qid_col, doc_col, rel_col = cols[0], cols[1], cols[2]
    qrels_df = qrels_df.rename(columns={qid_col: "qid", doc_col: "docno", rel_col: "label"})
    qrels_df["qid"] = qrels_df["qid"].astype(str)
    qrels_df["docno"] = qrels_df["docno"].astype(str)
    # build nested dict
    qrels = defaultdict(dict)
    for _, r in qrels_df.iterrows():
        qid = r["qid"]
        doc = r["docno"]
        label = int(r["label"])
        qrels[qid][doc] = label
    return qrels_df, qrels


def train_probfuse(runs_train, qrels_train_dict, x=25, L=100, variant="judged", eps=0.0):
    """
    Train probFuse: compute P_k^{(m)} for each ranker m and each segment k in 1..x.
    variant: "all" (probFuseAll) or "judged" (probFuseJudged)
    returns: dict P_probs[ranker] = list of length x with probabilities for segments 1..x
    """
    P_probs = {}
    # initialize counters per ranker per segment
    for ranker, df in runs_train.items():
        rel_counts = [0] * x
        nonrel_counts = [0] * x
        total_counts = [0] * x
        # iterate queries in this run
        qids = sorted(df["qid"].unique())
        for qid in qids:
            sub = df[df["qid"] == qid]
            for _, row in sub.iterrows():
                pos = int(row["rank"])
                doc = str(row["docno"])
                k = pos_to_segment(pos, L, x) - 1  # zero-index for lists
                total_counts[k] += 1
                # check relevance judgment
                if qid in qrels_train_dict and doc in qrels_train_dict[qid]:
                    lab = qrels_train_dict[qid][doc]
                    if lab > 0:
                        rel_counts[k] += 1
                    else:
                        nonrel_counts[k] += 1
                else:
                    # unjudged: counted only in total_counts (for probFuseAll)
                    pass
        # compute probabilities per segment
        probs = [0.0] * x
        for k in range(x):
            if variant.lower() == "all":
                denom = total_counts[k]
                num = rel_counts[k]
                if denom > 0:
                    probs[k] = float(num) / float(denom)
                else:
                    probs[k] = eps
            elif variant.lower() == "judged":
                denom = rel_counts[k] + nonrel_counts[k]
                num = rel_counts[k]
                if denom > 0:
                    probs[k] = float(num) / float(denom)
                else:
                    probs[k] = eps
            else:
                raise ValueError("variant must be 'all' or 'judged'")
        P_probs[ranker] = probs
    return P_probs


def build_probfuse_run(runs_test, P_probs, x=25, L=100, run_tag="probFuse"):
    """
    Construct fused run DataFrame for test runs using learned P_probs.
    Scoring: S_d = sum_m (P_km / k)  where k is 1..x segment index
    """
    records = []
    rankers = list(runs_test.keys())
    # determine all qids from the topics (should pass topics separately) - we'll use union of qids in runs_test
    all_qids = set()
    for r in rankers:
        all_qids.update(runs_test[r]["qid"].unique())
    all_qids = sorted(all_qids, key=lambda x: int(x) if x.isdigit() else x)

    for qid in all_qids:
        doc_scores = defaultdict(float)
        # union of docs across rankers for this qid
        docs_union = set()
        for ranker in rankers:
            df_q = runs_test[ranker][runs_test[ranker]["qid"] == qid]
            docs_union.update(df_q["docno"].tolist())
        if not docs_union:
            continue
        # create mapping ranker->(doc->pos) for faster lookup
        for ranker in rankers:
            df_q = runs_test[ranker][runs_test[ranker]["qid"] == qid]
            if df_q.empty:
                continue
            # for each returned doc
            for _, row in df_q.iterrows():
                doc = str(row["docno"])
                pos = int(row["rank"])
                k = pos_to_segment(pos, L, x)
                # fetch probability for this ranker and segment
                prob = 0.0
                if ranker in P_probs:
                    prob = P_probs[ranker][k - 1]
                # accumulate contribution: prob / k
                contribution = prob / float(k) if k > 0 else 0.0
                doc_scores[doc] += contribution
        # rank docs by score
        fused_list = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (docno, score) in enumerate(fused_list, start=1):
            records.append([qid, "Q0", docno, rank, score, run_tag])

    df_fused = pd.DataFrame(records, columns=["qid", "iter", "docno", "rank", "score", "runid"])
    return df_fused


def write_runfile(df, path):
    df_sorted = df.sort_values(["qid", "rank"])
    with open(path, "w") as fout:
        for _, r in df_sorted.iterrows():
            fout.write(f"{r['qid']} Q0 {r['docno']} {int(r['rank'])} {float(r['score']):.6f} {r['runid']}\n")
    print(f"Saved fused run to: {path}")


def parse_measures(metrics_list):
    measures = []
    for m in metrics_list:
        mlow = m.lower()
        if mlow == "map":
            measures.append(pt.measures.AP(rel=2))
        elif mlow.startswith("ndcg"):
            if "@" in mlow:
                k = int(mlow.split("@")[1])
                measures.append(pt.measures.nDCG(cutoff=k))
            else:
                measures.append(pt.measures.nDCG())
        elif mlow in ("recip_rank", "rr"):
            measures.append(pt.measures.RR())
        elif mlow.startswith("p"):
            # P@k e.g., p10
            try:
                k = int(mlow[1:])
                measures.append(pt.measures.P(cutoff=k))
            except:
                raise ValueError(f"Unsupported metric {m}")
        else:
            raise ValueError(f"Unsupported metric {m}")
    return measures


def main():
    parser = argparse.ArgumentParser(description="probFuse training & evaluation")
    parser.add_argument("--train_res_path", type=str, required=True,
                        help="Directory with training run files (.res)")
    parser.add_argument("--test_res_path", type=str, required=True,
                        help="Directory with test run files (.res)")
    parser.add_argument("--train_qrels", type=str, required=True, help="Training qrels file (for prob estimates)")
    parser.add_argument("--test_qrels", type=str, required=True, help="Test qrels file (for evaluation)")
    parser.add_argument("--topics", type=str, required=True, help="Test topics file (qid \\t query)")
    parser.add_argument("--x", type=int, default=25, help="Number of segments")
    parser.add_argument("--L", type=int, default=100, help="Top-L cutoff used to compute segments")
    parser.add_argument("--variant", choices=["all", "judged"], default="judged",
                        help="probFuse variant: 'all' treats unjudged as nonrelevant; 'judged' ignores unjudged")
    parser.add_argument("--metrics", type=str, nargs="+", default=["map", "ndcg@10", "recip_rank"])
    parser.add_argument("--output_run", type=str, default=None, help="Optional path to write fused run (.res)")
    parser.add_argument("--save_results", type=str, default=None, help="Optional CSV path to save metric results")
    parser.add_argument("--eps", type=float, default=0.0, help="Fallback small probability when denom==0")
    args = parser.parse_args()

    # Load runs
    print("Loading training runs...")
    runs_train = load_runs(args.train_res_path)
    print("Loading test runs...")
    runs_test = load_runs(args.test_res_path)

    # Load qrels
    print("Loading qrels...")
    _, qrels_train_dict = load_qrels_to_df(args.train_qrels)
    qrels_test_df, _ = load_qrels_to_df(args.test_qrels)

    # Load topics
    topics = pd.read_csv(args.topics, sep="\t", names=["qid", "query"]).astype({"qid": str})

    # Train probFuse probabilities
    print(f"Training probFuse probabilities (variant={args.variant}) with x={args.x}, L={args.L} ...")
    P_probs = train_probfuse(runs_train, qrels_train_dict, x=args.x, L=args.L, variant=args.variant, eps=args.eps)
    print("Training complete. Sample probabilities (first ranker):")
    # print sample for first ranker
    sample_ranker = next(iter(P_probs.keys()))
    print(f"Ranker: {sample_ranker} probs (k=1..{args.x}):")
    print(P_probs[sample_ranker])

    # Build fused run on test runs
    run_tag = f"probFuse-{args.variant}"
    df_fused = build_probfuse_run(runs_test, P_probs, x=args.x, L=args.L, run_tag=run_tag)

    # Optionally save fused run
    if args.output_run:
        write_runfile(df_fused, args.output_run)

    # Prepare experiment runs and names
    all_runs, all_names = [], []

    # Add fused run first
    all_runs.append(df_fused)
    all_names.append(run_tag)

    # Add baseline runs (original rankers)
    for ranker, df in runs_test.items():
        all_runs.append(df)
        all_names.append(ranker)

    # Prepare evaluation measures
    measures = parse_measures(args.metrics)

    # Run evaluation (PyTerrier Experiment)

    print("Running evaluation with PyTerrier...")
    results = pt.Experiment(
        all_runs,
        qrels=qrels_test_df,
        topics=topics,
        eval_metrics=measures,
        names=all_names
    )

    print("\n=== Evaluation Results ===")
    print(results)

    # Save results to CSV if requested
    if args.save_results:
        try:
            results_df = pd.DataFrame(results)
        except Exception:
            # fallback: convert printed string into file (less ideal)
            results_df = None
        if results_df is not None:
            results_df.to_csv(args.save_results, index=False)
            print(f"Saved metrics CSV to: {args.save_results}")
        else:
            print("Could not convert results to DataFrame to save.")

if __name__ == "__main__":
    main()


# python3 eval-precise-qpp/ProbFuse.py --train_res_path lucene-msmarco/data/runs/2019/RL_norm --test_res_path lucene-msmarco/data/runs/2020/RL_norm --train_qrels eval-precise-qpp/data/2019.qrels --test_qrels eval-precise-qpp/data/2020.qrels --topics eval-precise-qpp/data/2020.queries --variant judged --x 50 --L 100 --eps 1e-6
# python3 eval-precise-qpp/ProbFuse.py --train_res_path lucene-msmarco/data/runs/2020/RL_norm --test_res_path lucene-msmarco/data/runs/2019/RL_norm --train_qrels eval-precise-qpp/data/2020.qrels --test_qrels eval-precise-qpp/data/2019.qrels --topics eval-precise-qpp/data/2019.queries --variant judged --x 25 --L 100 --eps 1e-6
