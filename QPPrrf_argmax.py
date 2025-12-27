import pyterrier as pt
import pandas as pd
import os
import argparse
from collections import defaultdict

# if not pt.started():
#     pt.init()
model_name_dict={0:"SMV",1:"Sigma_max",2:"Sigma(%)",3:"NQC",4:"UEF",5:"RSD",6:"QPP-PRP", 7:"WIG", 8:"SCNQC", 9:"QV-NQC",10: "DM", 11:"NQA-QPP", 12:"BERTQPP"}

def load_qpp_estimates(qpp_path):
    """
    Load QPP estimates from folder of .qpp files.
    Returns: dict {qid: {ranker_name: [qpp1, qpp2, ...]}}
    """
    qpp_data = defaultdict(dict)
    files = [os.path.join(qpp_path, f) for f in os.listdir(qpp_path) if f.endswith(".mmnorm.qpp")]  #zscore 
    
    for f in files:
        ranker = os.path.basename(f).replace(".res.mmnorm.qpp","")  #.res.znorm.qpp minmax 
        with open(f, "r") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                # print(parts)
                if len(parts) < 2:
                    continue
                qid = parts[0]
                scores = [float(x) for x in parts[1:]]  # all QPP methods in columns
                qpp_data[qid][ranker] = scores
    return qpp_data

def load_runs(res_path):
    """
    Load all ranker runs as DataFrames
    Returns: dict {ranker: df}
    """
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".100.res")]
    for f in files:
        ranker = f.replace(".res","")
        df = pd.read_csv(os.path.join(res_path, f), sep=r"\s+", names=["qid","iter","docno","rank","score","runid"])
        runs[ranker] = df
    return runs

def build_adaptive_run(qpp_data, runs, qpp_method_index=0):
    """
    Build adaptive run using argmax-QPP strategy for a specific QPP method (column index).
    Returns a concatenated DataFrame of selected rankers per query.
    """
    records = []
    for qid, ranker_scores in qpp_data.items():
        if not ranker_scores:
            continue
        # select ranker that maximizes QPP score for this QPP method
        selected_ranker = max(ranker_scores.items(), key=lambda x: x[1][qpp_method_index])[0]
        df = runs[selected_ranker.split('.res')[0]]
        df_q = df[df["qid"] == int(qid)]
        records.append(df_q)

    if records:
        adaptive_df = pd.concat(records)
        # Re-rank by score within each query
        adaptive_df = adaptive_df.sort_values(["qid", "score"], ascending=[True, False])
        adaptive_df["rank"] = adaptive_df.groupby("qid").cumcount() + 1
        return adaptive_df
    else:
        return pd.DataFrame(columns=["qid","iter","docno","rank","score","runid"])
    
def build_rrf_run(runs, k=60, qpp_data=None, qpp_method_index=None):
    records = []
    rankers = list(runs.keys())
    for qid in runs[rankers[0]]["qid"].unique():
        doc_scores = defaultdict(float)
        for ranker in rankers:
            df_q = runs[ranker][runs[ranker]["qid"] == int(qid)]
            if df_q.empty:
                continue
            weight = 1.0
            # print("qpp_data",qpp_data['87181'])
            # print(ranker)
            if qpp_data and qpp_method_index is not None:
                try:
                    weight = qpp_data[str(qid)][ranker][qpp_method_index]
                    # print("weight",weight)
                except KeyError:
                    weight = 0.0
                    # print("null")
            for _, row in df_q.iterrows():
                # print("row",row['score'])
                doc_scores[row["docno"]] += weight * (1.0 / (k + row["rank"]))
        if doc_scores:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (docno, score) in enumerate(sorted_docs, start=1):
                records.append([qid,"Q0",docno,rank,score,"Weighted-RRF" if qpp_data else "RRF"])
    return pd.DataFrame(records, columns=["qid","iter","docno","rank","score","runid"])


def main():
    parser = argparse.ArgumentParser(description="Flexible QPP evaluation")
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--qpp_path", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs='+', default=["map","ndcg@10","recip_rank"])
    parser.add_argument("--strategy", type=str, choices=["adaptive","rrf","wrrf"], required=True,
                        help="Choose fusion strategy: adaptive | rrf | wrrf")
    args = parser.parse_args()

    runs = load_runs(args.res_path)
    qpp_data = load_qpp_estimates(args.qpp_path)

    qrels = pt.io.read_qrels(args.qrels)
    topics = pd.read_csv(args.topics,sep="\t",names=["qid","query"]).astype({'qid':'str'})

    sample_scores = next(iter(next(iter(qpp_data.values())).values()))
    num_qpp_methods = len(sample_scores)
    print(f"Detected {num_qpp_methods} QPP methods (columns) per ranker")

    measures = []
    for m in args.metrics:
        if m.lower() == "map":
            measures.append(pt.measures.AP(rel=2))
        elif "ndcg@10" in m.lower():
            k = int(m.split("@")[1]) if "@" in m else None
            # print("k",k)
            measures.append(pt.measures.nDCG(cutoff=k) if k else pt.measures.nDCG())
        # elif "ndcg@10" in m.lower():
        #     k = int(m.split("@")[1]) if "@" in m else None
        #     measures.append(pt.measures.nDCG(k=k) if k else pt.measures.nDCG())
        elif m.lower() == "recip_rank":
            measures.append(pt.measures.RR())

    all_runs, all_names = [], []

    if args.strategy == "adaptive":
        for i in range(num_qpp_methods):
            df = build_adaptive_run(qpp_data, runs, qpp_method_index=i)
            all_runs.append(df)
            all_names.append(f"Adaptive-QPP-{model_name_dict[i]}")

    elif args.strategy == "wrrf":
        for i in range(num_qpp_methods):
            df = build_rrf_run(runs, qpp_data=qpp_data, qpp_method_index=i)
            all_runs.append(df)
            all_names.append(f"Weighted-RRF-QPP-{model_name_dict[i]}")

    elif args.strategy == "rrf":
        df = build_rrf_run(runs)
        all_runs.append(df)
        all_names.append("RRF")

    # Always add baselines
    all_runs.extend(runs.values())
    all_names.extend(runs.keys())

    results = pt.Experiment(
        all_runs,
        qrels=qrels,
        topics=topics,
        eval_metrics=measures,
        names=all_names
    )
    print(results)

if __name__ == "__main__":
    main()


# --res_path "/content/2020" --qpp_path "/content/2020" --qrels "/content/2020.qrels" --topics "/content/2020.queries"
# python3 eval-precise-qpp/argmax_qpp_evaluation.py --res_path lucene-msmarco/data/runs/2019/norm_qpp --qpp_path lucene-msmarco/data/runs/2019/norm_qpp --qrels eval-precise-qpp/data/2019.qrels --topics eval-precise-qpp/data/2019.queries
# python3 eval-precise-qpp/QPPrrf_argmax.py --res_path lucene-msmarco/data/runs/2019/norm_qpp2 --qpp_path lucene-msmarco/data/runs/2019/norm_qpp2 --qrels eval-precise-qpp/data/2019.qrels --topics eval-precise-qpp/data/2019.queries --strategy wrrf

# python3 eval-precise-qpp/QPPrrf_argmax.py --res_path lucene-msmarco/data/runs/2019/norm_qpp2 --qpp_path lucene-msmarco/data/runs/2019/norm_qpp2 --qrels eval-precise-qpp/data/2019.qrels --topics eval-precise-qpp/data/2019.queries --strategy adaptive