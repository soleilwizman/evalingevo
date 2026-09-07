# Runbook: variant-level embeddings on a GPU

The committed embeddings cover the 2,595 reference 200-mers and **none** of the
5,428 variant sequences, so no probe of a variant effect is possible today.
This is the GPU work that unblocks it. Everything downstream is CPU and already
written and tested.

Three runs, 5,428 sequences each: the variant sequences only. The references
already exist and get reused, and `variant_probe.py` refuses to subtract two
directories unless their `meta.json` name the same checkpoint, layer and
revision.

| model | variant embeddings | reference embeddings (already committed) |
|---|---|---|
| Evo 2 7B, `blocks.26.mlp.l3` | `results/evo_variants` | `results/evo_probe` |
| NTv3 100M, block 5 | `results/ntv3_100m_variants` | `results/ntv3_100m_final` |
| NTv3 650M, block 11 | `results/ntv3_650m_variants` | `results/ntv3_650m_final` |

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

## 3. Probe (CPU, minutes)

```bash
python3 variant_probe.py --embeddings results/ntv3_100m_variants \
    --reference results/ntv3_100m_final --label "NTv3 100M probe"
python3 variant_probe.py --embeddings results/ntv3_650m_variants \
    --reference results/ntv3_650m_final --label "NTv3 650M probe"
python3 variant_probe.py --embeddings results/evo_variants \
    --reference results/evo_probe       --label "Evo 2 probe"
```

The checkpoint, layer and revision of the two directories are compared before
anything is subtracted, and the run stops with both tuples printed if they
differ. That is the whole reason the references are not re-embedded: the check
is free and the GPU time is not.

If you would rather have a self-contained directory, add `--include-reference`
to the embedding run (5,428 to 8,023 sequences, about 48% more time) and then
drop `--reference` from the probe.

Each writes `variant_probe.json` beside the embeddings with four rows and two
margins:

| row | what it answers |
|---|---|
| difference vector `h(alt) - h(ref)` | the probe itself |
| **reference only, control** | can `h(ref)` alone predict the effect? |
| delta k-mers | the baseline every other bar is judged against |
| difference plus delta k-mers | does the model add to the baseline? |

The two margins are the results. **Difference minus reference-only** says
whether the subtraction bought anything: each reference carries about two
variants, so `h(ref)` alone can only learn how mutable that element is, never
which substitution happened. If the difference vector does not beat it, the
probe is reading the genomic background rather than the variant, and its
correlation is not evidence about variant effects. **Difference minus delta
k-mers** (+0.177 to beat) is whether the model beats letter counting.

`--include-alternate` adds `h(alt)` alone as a third control, at the cost of
another wide ridge fit.

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

`X_mean.npy` and `X_last.npy` per run, at 5,428 rows: NTv3 100M ~17 MB each,
650M ~33 MB each, Evo 2 ~89 MB each. `.gitignore` excludes `results/*_variants/X_*.npy` so a
13 GB model's output does not land in git by accident. Commit the JSON and the
`sequences.csv`; keep the matrices out or put them in LFS.

Add `--include-double` to any run to also embed the AB sequences, which is what
a later interaction probe would need. It costs ~35% more time.
