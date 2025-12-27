#!/bin/bash
# Generate RAG jobs for prompt variant B (all Qwen retrievers)

RETRIEVERS=(
  "bm25:BM25:BM25_nqtest_all.res"
  "splade:SPLADE:SPLADE_nqtest_all.res"
  "bm25-monot5:BM25-MonoT5:BM25_MonoT5_nqtest_all.res"
  "bm25-tct:BM25-TCT:BM25_TCT_nqtest_all.res"
  "bm25-bge-reranker:BM25-BGE-Reranker:BM25_BGE_Reranker_nqtest_all.res"
  "tctcolbert:TCT-ColBERT:TCTColBERT_nqtest_all.res"
  "bge:BGE:BGE_nqtest_all.res"
)

for entry in "${RETRIEVERS[@]}"; do
  IFS=':' read -r slug retriever runfile <<< "$entry"
  
  cat > "rag-${slug}-prompt-b-k0to6.job.yaml" <<EOF
# Incoming: shared-datasets + shared-indexes (${runfile}) --- {Dict, JSON/TREC}
# Processing: RAG generation (prompt variant B: k=0 uses parametric knowledge, k≥1 uses context) + QA scoring --- {2 jobs: generation, evaluation}
# Outgoing: shared-rag-results:/mnt/rag/nqtest_prompt_b/${slug}_k0to6 --- {Dict, JSONL}
apiVersion: batch/v1
kind: Job
metadata:
  name: rag-${slug}-prompt-b-k0to6
  namespace: 425krish
spec:
  backoffLimit: 10
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: rag
        image: image-registry.openshift-image-registry.svc:5000/425krish/rag-runner:latest
        imagePullPolicy: Always
        env:
        - name: HOME
          value: /tmp
        - name: TMPDIR
          value: /tmp
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: LM_STUDIO_URL
          value: http://qwen3-4b-vllm-gpu.425krish.svc.cluster.local:8000/v1
        - name: SRC_SKIP_PACKAGE_IMPORTS
          value: "1"
        - name: SRC_SKIP_IR_EVAL
          value: "1"
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euo pipefail
          cd /app
          python scripts/12_rag_runfile_eval.py \\
            --nq_path /mnt/datasets/dpr/nq_test.json \\
            --psgs_path /mnt/datasets/dpr/data/wikipedia_split/psgs_w100/psgs_w100.tsv \\
            --run_path /mnt/index/dpr/runs/${runfile} \\
            --retriever ${retriever} \\
            --dataset nq_test \\
            --output_root /mnt/rag/nqtest_prompt_b/${slug}_k0to6 \\
            --shots 0,1,2,3,4,5,6 \\
            --model Qwen/Qwen3-4B-Instruct-2507 \\
            --prompt_variant variant_b \\
            --rag_variant ${retriever}_PromptB_k0to6 \\
            --shard_size 100 \\
            --resume
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
        volumeMounts:
        - name: datasets
          mountPath: /mnt/datasets
        - name: indexes
          mountPath: /mnt/index
        - name: rag
          mountPath: /mnt/rag
      volumes:
      - name: datasets
        persistentVolumeClaim:
          claimName: shared-datasets
      - name: indexes
        persistentVolumeClaim:
          claimName: shared-indexes
      - name: rag
        persistentVolumeClaim:
          claimName: shared-rag-results
EOF

  echo "Created rag-${slug}-prompt-b-k0to6.job.yaml"
done

echo "All prompt variant B job manifests generated."

