# Variant embedding runbook

Run commands from the repository root. New runs belong under `results/v2/`; leave
historical artifacts untouched. See [PROTOCOL.md](PROTOCOL.md) before mixing
embeddings or comparing old and new scores.

## Preparation

Use a compatible CUDA environment for Evo 2 and the dependencies documented by
the checkpoint authors. Install CPU dependencies from `requirements.txt` and
authenticate with Hugging Face for gated checkpoints. Never commit access tokens
in commands or put credential values in result metadata.

```bash
nvidia-smi
df -h .
hf auth login
export MODEL_REVISION="<immutable Hugging Face checkpoint commit>"
```

Replace the placeholder. `main` is accepted by some entrypoints but is mutable,
so it is not sufficient to reproduce independent embedding jobs.

## Paired variant workflow

This obtains reference, mean-difference and variant-position features from the
same model load. NTv3 defaults to the final deconvolution stage (`--layer -1`),
not the two-position transformer bottleneck.

```bash
# Offline shape map, no download.
bash scripts/run_ntv3_deconv.sh map 650m

# GPU smoke: separate output, 40 observations.
python3 scripts/variant_probe.py embed --model ntv3 --layer -1 --limit 40 \
  --checkpoint InstaDeepAI/NTv3_650M_pre --revision "$MODEL_REVISION" \
  --out results/v2/smoke_ntv3_variants

# Full run after the smoke passes.
python3 scripts/variant_probe.py embed --model ntv3 --layer -1 \
  --checkpoint InstaDeepAI/NTv3_650M_pre --revision "$MODEL_REVISION" \
  --out results/v2/vp_ntv3_650m

# Evo requires the evo2 package and supported NVIDIA hardware.
python3 scripts/variant_probe.py embed --model evo2 \
  --layer blocks.26.mlp.l3 --out results/v2/vp_evo2

# DNABERT-2: set its own immutable DNABERT_REVISION first.
python3 scripts/variant_probe.py embed --model dnabert2 \
  --revision "$DNABERT_REVISION" --out results/v2/vp_dnabert2

# CPU: choose the target explicitly.
python3 scripts/variant_probe.py probe --embeddings results/v2/vp_ntv3_650m --target signed
python3 scripts/variant_probe.py probe --embeddings results/v2/vp_ntv3_650m --target magnitude
python3 scripts/variant_probe.py compare --target signed \
  --embeddings results/v2/vp_evo2 results/v2/vp_ntv3_650m
```

Compare complete runs on identical observations, not smoke subsets against full
runs. Outputs are `X_d_mean.npy`, `X_d_pos.npy`, `X_wt_mean.npy`, observation and
sequence tables, and metadata. This is not the resumable alternate-only workflow.
Reference-only controls genomic-background predictability; subtraction does not
guarantee that background is removed. Pooling selection stays in training folds.

## Separate reference and alternate jobs

Use this resumable path for a specifically chosen transformer representation.
Both sides must record the same checkpoint, immutable revision, layer and pooling
protocol. Historical metadata lacking that evidence is rejected.

```bash
python3 scripts/ntv3_probe.py embed --checkpoint InstaDeepAI/NTv3_100M_pre \
  --layer 5 --revision "$MODEL_REVISION" --batch-size 8 --out results/v2/ntv3_100m_ref
python3 scripts/embed_variants.py --backend ntv3 --checkpoint InstaDeepAI/NTv3_100M_pre \
  --layer 5 --revision "$MODEL_REVISION" --batch-size 8 --out results/v2/ntv3_100m_var
python3 scripts/variant_probe_fit.py --variants results/v2/ntv3_100m_var \
  --references results/v2/ntv3_100m_ref --label "NTv3 100M transformer probe"
```

For 650M use that checkpoint's revision and block 11 on both sides. These indices
are zero-based transformer-block indices, not `hidden_states` indices. The fitter
reports difference, reference-only, delta-k-mer and GC readouts on the same covered
variants. It prints coverage; a subset is not the full benchmark.

For separate Evo jobs, pass the same local `--weights` file to `evo_probe.py embed`
and `embed_variants.py --backend evo2`; metadata checks its hash. Otherwise use the
paired workflow: the Evo API does not resolve a supplied revision label.

`embed_variants.py --include-reference` and `--include-double` add WT/AB sequences
to its cache. They do not create an interaction probe or replace the fitter's
required `--references` element table. There is no `variant_probe.py --reference`
or `--include-alternate` option.

## Resume and verification

Re-run an interrupted `embed_variants.py` command unchanged. It checks configuration,
input/code hashes and progress before resuming. Changed checkpoint, layer, batch
size, code or inputs require a new output directory. Progress follows flushed
matrix writes. Never edit metadata to force an incompatible resume. An incomplete
paired run should be restarted in a new directory, not used as a complete cache.

Check metadata, finite matrices, row counts and printed unit counts. NTv3 `--layer -1`
should yield one vector per real base. Coarse stages map real-base coverage, not
recovered resolution. Stop on load, shape or non-finite errors; do not bypass checks.

GPU compatibility and biological results require actual checkpoint runs. CPU
tests and mocked forward passes are necessary but not sufficient. Large matrices
can exceed ordinary GitHub file limits; do not commit weights or bulk artifacts
as part of a source-code cleanup.
