# Runbook: variant-level embeddings on a GPU

The committed embeddings cover the 2,595 reference 200-mers and **none** of the
5,428 variant sequences, so no probe of a variant effect is possible today.
This is the GPU work that unblocks it. Everything downstream is CPU and already
written and tested.

Three runs, ~8,023 sequences each (2,595 reference + 5,428 variant, deduplicated).

## 0. Before you start

```bash
git checkout claude/figures        # or wherever these scripts live
nvidia-smi                          # confirm the GPU and free VRAM
df -h .                             # Evo 2 needs ~15 GB free for weights
```

Two independent setups. Do NTv3 first: it is smaller, faster, and will surface
any data-path problem before you spend time on the 7B model.

## 1. NTv3 100M and 650M

```bash
pip install torch transformers
huggingface-cli login               # the InstaDeepAI checkpoints are gated

python3 embed_variants.py --backend ntv3 --out results/ntv3_100m_variants \
    --checkpoint InstaDeepAI/NTv3_100M_pre --layer 5  --batch-size 16

python3 embed_variants.py --backend ntv3 --out results/ntv3_650m_variants \
    --checkpoint InstaDeepAI/NTv3_650M_pre --layer 11 --batch-size 8
```

Layer 5 is the final block of the 6-block 100M model, layer 11 the final block
of the 12-block 650M, matching the committed reference embeddings exactly.

## 2. Evo 2

```bash
# the evo2 package is not on PyPI; it needs CUDA and builds its own kernels
git clone https://github.com/ArcInstitute/evo2 && cd evo2 && pip install . && cd ..

python3 embed_variants.py --backend evo2 --out results/evo_variants \
    --checkpoint evo2_7b_base --layer blocks.26.mlp.l3 --batch-size 4
```

Weights are ~13 GB and download on first use. If VRAM is tight, drop
`--batch-size` to 2 or 1; the run is resumable so an OOM costs only the current
batch.

## 3. Verify before trusting anything

The new run re-embeds the reference sequences, so its `wt` rows must reproduce
the committed reference embeddings. If they do not, the layer, dtype, revision
or padding has changed and the difference vectors are not comparable.

```bash
python3 - <<'EOF'
import numpy as np, pandas as pd
new, old = "results/ntv3_100m_variants", "results/ntv3_100m_final"   # or evo_variants / evo_probe
n = pd.read_csv(f"{new}/sequences.csv"); o = pd.read_csv(f"{old}/elements.csv")
Xn = np.load(f"{new}/X_mean.npy", mmap_mode="r"); Xo = np.load(f"{old}/X_mean.npy")
pos = {s: i for i, s in enumerate(n.sequence_id)}
rows = [pos[s] for s in o.sequence_id]
d = np.abs(np.asarray(Xn[rows]) - Xo).max()
print(f"max abs difference on the {len(rows)} shared reference sequences: {d:.2e}")
print("consistent" if d < 1e-3 else "MISMATCH: do not form difference vectors across these runs")
EOF
```

## 4. Probe (CPU, minutes)

```bash
python3 variant_probe.py --embeddings results/ntv3_100m_variants --label "NTv3 100M probe"
python3 variant_probe.py --embeddings results/ntv3_650m_variants --label "NTv3 650M probe"
python3 variant_probe.py --embeddings results/evo_variants       --label "Evo 2 probe"
```

Each writes `variant_probe.json` beside the embeddings with the probe's
Spearman, its interval, and its margin over the delta k-mer baseline (+0.177),
which is the number that decides whether the probe found anything.

## What to expect, and the one caveat worth knowing first

The feature is `h(alt) - h(ref)`, pooled, both from the same checkpoint. The
difference cancels the static genomic background, which matters here: an
element-level probe predicts reference activity at about +0.5, so a probe fed
raw embeddings would mostly learn which region it was looking at.

For NTv3 specifically, 7 downsamples put the transformer blocks at a
**two-position bottleneck** for a 256-token input. A single-base change has to
survive 128x downsampling to appear there at all, so a null from NTv3 may say
more about where it was probed than about the model. Evo 2 has no such
bottleneck and is the cleaner test. If NTv3 comes back flat and you want a
second look, probe a full-resolution layer instead of the transformer block.

## Sizes

`X_mean.npy` and `X_last.npy` per run: NTv3 100M ~25 MB each, 650M ~50 MB each,
Evo 2 ~131 MB each. `.gitignore` excludes `results/*_variants/X_*.npy` so a
13 GB model's output does not land in git by accident. Commit the JSON and the
`sequences.csv`; keep the matrices out or put them in LFS.

Add `--include-double` to any run to also embed the AB sequences, which is what
a later interaction probe would need. It costs ~35% more time.
