# ENSy — Plan for producing the real (frozen) results off the local machine

**Goal:** produce the frozen `data/dataset.json` (Qwen + Gemma + BERT) and the 12 main configurations plus ablations, reproducibly, without depending on the local machine's limited resources.

## 1. What the workload actually needs

| Step | Cost today | Notes |
|---|---|---|
| Tokenizers (3 models) | ~10 MB download, seconds | Gemma's tokenizer is also gated |
| Embedding matrices | **Full checkpoint download**: Qwen ~1 GB, BERT ~0.4 GB, Gemma 2B shard ~5 GB; Gemma matrix in float32 ≈ 2 GB RAM | The only heavy step |
| Projections + co-ranking (N ≈ 440) | Seconds to a few minutes on CPU | No GPU needed |

The bottleneck is **downloading and holding whole checkpoints**, not computation.

### Probe result (2026-10-01)

Safetensors files start with a byte-offset header, so HTTP `Range` requests can fetch only the rows we need. Probed on `Qwen/Qwen2.5-0.5B-Instruct`: header + 4 embedding rows = **39 KiB transferred** (vs. 988 MB), and the rows are **bit-identical** to the full download. For Gemma 2B, ~440 rows × 2048 × 2 bytes ≈ **1.8 MB**. That alone makes the pipeline feasible on almost any machine — remote compute then becomes a question of reproducibility, not of capacity.

## Status (2026-10-02)

| Task | Status |
|---|---|
| T1 ranged extraction | **Done** — `ensy.models.embedding_rows`, bit-identical to the full download (tests) |
| T2 Gemma access | **Done** — licences accepted; both variants run (`gemma`, `gemma2`) |
| T3 GitHub Actions | **Running** — first run (commit `009c1ac`) succeeded in 11 min 13 s: gated models accessible, 45/45 validation tests, full pipeline |
| T4 determinism check | **Done** — the CI rebuild reproduces the dataset hash exactly; PCA, PCA-NC and Isomap match the local results exactly; t-SNE and UMAP do **not** match across platforms (Windows local vs Linux CI) despite fixed seeds, with differences of the order of the seed-to-seed spread (e.g. Qwen UMAP $B_{NX}(10)$ +0.019 vs +0.009). Consequence: the CI results are the canonical numbers, and stochastic methods are reported over seeds |
| T5 Colab notebook | **Done** — `notebooks/ensy_colab.ipynb` |

Real results were produced on the limited local machine: the main matrix (16 configurations) took 3 min 17 s, including the ranged downloads; the ablation and sensitivity runs also completed locally.

## 2. Options

| | A. Ranged extraction, run anywhere | B. GitHub Actions (canonical run) | C. Google Colab |
|---|---|---|---|
| Resources | Local, ~1 GB RAM (torch import) | Private repo: 2 vCPU, 7 GB RAM, 14 GB disk; free-plan minutes budget | Free tier ~12 GB RAM, GPU not needed |
| Reproducibility | Local env, dirty trees possible | **Best**: clean checkout of a commit, pinned deps, logs kept, `working_tree_dirty: false` | Weakest: preinstalled packages drift; sessions time out |
| Gemma token | Local env var | Repository secret `HF_TOKEN` | Colab secret `HF_TOKEN` |
| Output | Local files | Artifacts + PR with `data/dataset.json` and `results/` | Manual download / push |
| Best for | Development, quick checks | **Frozen thesis results** | Interactive exploration, multi-seed sensitivity |

**Recommendation:** A + B. Implement ranged extraction first (it shrinks every option); then run the frozen pipeline on GitHub Actions so the thesis numbers come from a clean, pinned, logged run tied to a commit SHA. Keep Colab as an optional notebook for exploration and the sensitivity analysis — never as the source of the reported numbers.

## 3. Investigation tasks (in order, each with an acceptance check)

**T1 — Ranged extraction in `ensy/models.py`**
- Add `embedding_rows(key, ids)`: read the safetensors header via `Range`, compute each row's byte span, merge contiguous spans, fetch concurrently, decode BF16/F16/F32.
- Accept when: a test shows `embedding_rows(k, ids) == embedding_matrix(k)[ids]` **bit-for-bit** for Qwen and BERT; Gemma extraction completes with < 50 MB of traffic.
- Keep the full-download path as the reference implementation used by that test.

**T2 — Gemma access**
- Accept the Gemma licence on Hugging Face; create a read token (fine-grained, "read access to public gated repos").
- Decide `google/gemma-2b` vs `google/gemma-2-2b` (open item in the plan) and pin it.
- Accept when: `python -m ensy.dataset` builds with all three tokenizers locally and reports `frozen: true`.

**T3 — GitHub Actions workflow (`.github/workflows/experiment.yml`)**
- Trigger: `workflow_dispatch` only (manual; inputs: models, whether to run ablations).
- Steps: checkout → Python 3.12 → `pip install` pinned requirements (CPU-only torch wheel to save ~2 GB) → `python -m ensy.dataset` → `python -m ensy.experiment` → `--ablation` → upload `data/` and `results/` as artifacts → open a PR from a bot branch.
- Secret: `HF_TOKEN` (repo secret; not exposed to fork PRs). Cache `~/.cache/huggingface` keyed on model revisions.
- Accept when: a dry run with the provisional models finishes within budget (target < 15 min) and the results file records a clean tree and the commit SHA.

**T4 — Cross-machine determinism check**
- t-SNE/UMAP/Isomap outputs can differ in the last bits across CPUs/BLAS builds even with fixed seeds.
- Run the same commit locally and on the runner; compare metric curves.
- Accept when: differences are below a documented tolerance (e.g. |ΔM_T|, |ΔB_NX| < 1e-3). If they are not, the thesis reports the CI run as canonical and states the tolerance.

**T5 — Colab notebook (optional)**
- `notebooks/ensy_colab.ipynb`: clone the repo at a given commit, `pip install -r requirements.txt`, read `HF_TOKEN` from Colab secrets, run the pipeline, show the curves.
- Use for the multi-seed sensitivity analysis (plan Sec. 7), reported separately from the main results.

## 4. Safety and licensing notes

- Never commit the token; GitHub and Colab secrets only. The workflow must not echo it.
- Only derived artefacts are committed (words, token ids, metrics, projections); the embedding vectors themselves are **not** stored. Check that the Gemma terms of use allow publishing derived metrics (they generally do for research), and cite the model and revision.
- The full PTB is LDC-licensed; the pipeline uses the NLTK sample. Do not upload LDC data to Colab or Actions without checking the licence.

## 5. Decisions needed from the author

1. Gemma variant (`gemma-2b` vs `gemma-2-2b`).
2. Approve adding `HF_TOKEN` as a repository secret.
3. GitHub Actions minutes budget for the private repo (or make the repo public, which gives larger free runners).
