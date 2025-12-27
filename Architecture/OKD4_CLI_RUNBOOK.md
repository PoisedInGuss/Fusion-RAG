# OKD4 / OpenShift CLI Runbook (QPP-Fusion-RAG)

This is the single source of truth for running **repeatable, large** workloads on the IDA OKD4 cluster from this repo using **CLI-first** practices. The UI is for inspection; manifests + `oc` are for control.

## What you have in this repo (local tooling contract)

- **Repo-local binaries**
  - `oc/oc` (OpenShift CLI, **darwin-arm64**) — canonical in this repo
  - `bin/oc` (OpenShift CLI, **darwin-arm64**) — optional/alternate if present
  - `oc/odo-darwin-arm64` / `bin/odo` (developer workflow CLI, **darwin-arm64**)
- **Repo-local kubeconfig**
  - `./.kube/config` (created/updated by login)
- **Repo-local login script**
  - `./oc_login.sh` (reads token from `./.env`, logs in, selects project, writes `./.kube/config`)

## Non-negotiables (operational and security)

- **Never print tokens**. Do not `cat .env` into terminals that are logged/recorded. Treat tokens as compromised if they appear in logs, chat, screenshots, or paste buffers.
- **Do not commit `.env`**. Store it locally only.
- **Prefer repo-local kubeconfig**. This avoids polluting `~/.kube` and makes runs reproducible per repo.
- **Cluster TLS**: this environment currently uses an internal CA chain; the login script uses `--insecure-skip-tls-verify=true`. The correct fix is to install the cluster CA cert and stop skipping verification.

## One-time local setup (per machine)

From repo root:

```bash
export PATH="$PWD/bin:$PATH"
export KUBECONFIG="$PWD/.kube/config"
```

Optional: add those exports to your shell profile so they’re automatic when you work in this repo.

## One-line login (per session / when token expires)

From repo root:

```bash
./oc_login.sh
```

Contract:
- Reads `TOKEN=...` from `./.env`
- Logs into `https://api.os.dcs.gla.ac.uk:6443` (override via `OPENSHIFT_SERVER`)
- Selects project `425krish` (override via `OPENSHIFT_PROJECT`)
- Writes kubeconfig to `./.kube/config`

Useful overrides:

```bash
OPENSHIFT_PROJECT=425krish ./oc_login.sh
OPENSHIFT_SERVER=https://api.os.dcs.gla.ac.uk:6443 ./oc_login.sh
ENV_FILE=/path/to/.env ./oc_login.sh
KUBECONFIG_PATH=$PWD/.kube/config ./oc_login.sh
OC_BIN=$PWD/oc/oc ./oc_login.sh
```

### If login fails: “token provided is invalid or expired”

That error is not a network problem; it means your SSO-issued token has expired. Fix is to refresh the token and rerun login.

Token refresh (SSO):
- Open the web console (`https://console-openshift-console.apps.os.dcs.gla.ac.uk`) while on VPN.
- Use the account menu → **“Copy login command”** (or similar).
- Copy the `--token=...` value into `./.env` as `TOKEN=sha256~...` (do not commit the file).
- Rerun `./oc_login.sh` to recreate `./.kube/config`.

## “Clean CLI” command set (daily drivers)

### Context + permissions

```bash
oc whoami
oc whoami --show-server
oc projects
oc project 425krish
oc auth can-i create pods
oc auth can-i get pods
```

### Discover what the cluster supports (authoritative “API list”)

```bash
oc api-versions
oc api-resources
oc api-resources --namespaced=true
oc explain pod
oc explain job.spec.template.spec
```

### Inspect current state

```bash
oc get all
oc get pods -o wide
oc get job,cronjob
oc get pvc
oc get svc,route
oc get events --sort-by=.lastTimestamp | tail -100
```

### Debug why something is stuck

```bash
oc describe pod <pod>
oc logs -f <pod>
oc logs -f job/<jobname>
oc exec -it <pod> -- /bin/bash
```

### Quotas and limits (must check before big workloads)

```bash
oc describe quota
oc describe limitrange
```

## Large-task pattern (how you should run experiments)

### Use Jobs, not interactive pods

Interactive pods are for debugging. Large experiments must be run as **Jobs**:
- reproducible
- schedulable
- restartable
- auditable

### Always write outputs to persistent storage

Large runs must mount a PVC and write all outputs to it. If you write to container FS, you will lose outputs when pods die.

### Shard work (do not run one mega-pod)

For “many queries / many docs” workloads, shard by query range:
- One shard = one Job (or one index in an Indexed Job)
- Each shard writes separate outputs (e.g., `runs/shard_000.res`)
- Merge downstream

### Use Indexed Jobs when you have many shards

Indexed Jobs reduce operational burden:
- single manifest
- multiple indexed pods
- controlled `parallelism`

### GPU usage rule

Only request GPUs for steps that truly need them (embedding, heavy neural inference). Retrieval over prebuilt sparse/dense indexes is often CPU-bound; over-requesting GPUs makes you “Unschedulable” and wastes quota.

## When to use `odo` vs `oc`

- Use **`odo`** for developer loops (quick app deploys, iterative testing).
- Use **`oc` + manifests** for batch pipelines and reproducible experiments.

## Minimal repository conventions (recommended)

Create these directories (if you commit manifests later):
- `k8s/`: versioned manifests (Jobs, PVC, Secrets, ConfigMaps)
- `logs/`: local logs (do not commit secrets)
- `outputs/`: local outputs (or point to PVC mount)

Naming:
- Jobs: `<dataset>-<stage>-<shard>`
- ConfigMaps: `<dataset>-<stage>-config`
- Secrets: `<dataset>-<stage>-secrets` (never commit secret material)

## Quick sanity checklist before you start a large run

- `oc project 425krish` shows correct namespace
- `oc describe quota` shows enough headroom
- PVC exists and is mounted in manifests
- outputs path is on PVC
- jobs are sharded and `parallelism` is conservative
- token is not printed anywhere


