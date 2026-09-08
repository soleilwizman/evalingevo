# Which layer every readout uses

For the primary analysis, the authoritative registry is `scripts/benchmark_models.py`:
Evo 2 `blocks.26.mlp.l3`, NTv3 100M and 650M `hidden_states[-1]` after the last
deconvolution stage, and DNABERT-2's last encoder layer. Use `embed_elements.py`
and `regulatory_benchmark.py probes`; all primary probes are whole-element ridge
regressions against the same GC and k-mer controls. The tables below are historical
and exploratory, not substitute primary configurations.

One place to check before quoting any probe number. Every fact here is read from
the `meta.json` beside the matrices, or from the matrices themselves.

## Element-activity probes (2,595 reference 200-mers)

| result directory | checkpoint | layer | width |
|---|---|---|---|
| `results/evo_probe` | `evo2_7b_base` | `blocks.26.mlp.l3` | 4096 |
| `results/ntv3_100m_final` | `InstaDeepAI/NTv3_100M_pre` | `core.transformer_blocks.5.final_layer_norm` | 768 |
| `results/ntv3_650m_final` | `InstaDeepAI/NTv3_650M_pre` | `core.transformer_blocks.11.final_layer_norm` | 1536 |

Block 5 is the last of 6 for NTv3 100M and block 11 the last of 12 for 650M, so
both NTv3 readouts are the final transformer block. Evo 2 is a mid-stack block,
not its last. That asymmetry was a choice made before the results existed and has
never been revisited: no Evo 2 layer sweep has been run.

## Single-variant probes (5,666 substitutions)

The variant embeddings hold alternate sequences only (`include_reference: false`),
so the probe feature is `h(alt) - h(ref)` with the reference vector taken from the
element embeddings above. Both sides come from the same layer within a model.

| result directory | checkpoint | layer | width | n |
|---|---|---|---|---|
| `results/ntv3_100m_variants` | `InstaDeepAI/NTv3_100M_pre` | block 5, as above | 768 | 5,428 sequences, 5,666 variants matched |
| `results/ntv3_650m_variants` | `InstaDeepAI/NTv3_650M_pre` | block 11, as above | 1536 | 5,428 sequences, 5,666 variants matched |
| `results/vp_evo2` | `evo2_7b_base` | `blocks.26.mlp.l3` | 4096 | 5,428 observations |

`vp_evo2` holds no matrices, only derived tables, so it cannot be refitted under
the current protocol. Its number is carried with a flag, not merged in.

## The 650M layer sweep

`results/ntv3_650m_sweep` holds 26 matrices, `X_mean_L0.npy` through `X_mean_L25.npy`.
They are the model's `hidden_states` in order, not the 12 transformer blocks alone:
NTv3 650M has 7 conv stages, 12 transformer blocks and 7 deconv stages, which is 26.
Two facts pin the map. `python3 scripts/ntv3_unet.py list --offline --num-layers 12`
prints it from the architecture, and `X_mean_L25.npy` is bit-identical to
`results/ntv3_650m_deconv/X_mean.npy`, whose `meta.json` names `deconv_7` (max abs
diff 0.0).

| sweep index | stage | positions per 256-token input |
|---|---|---|
| L0 to L6 | `conv_1` to `conv_7` | 256, 128, 64, 32, 16, 8, 4 |
| L7 to L18 | `transformer_1` to `transformer_12` | 2 |
| L19 to L25 | `deconv_1` to `deconv_7` | 4, 8, 16, 32, 64, 128, 256 |

`results/ntv3_650m_sweep/layer_curve.txt` is the probe over that sweep on the same
folds and k-mer baseline (+0.4560) as every other element probe. L18 there is the
same stage as `results/ntv3_650m_final` (`transformer_11` is zero-indexed, block 12
of 12 in the map above).

Layers 4 through 21 are 100% non-finite in float32: `conv_5` to `conv_7`, every
transformer block, and `deconv_1` to `deconv_3`. That is corruption rather than a
model property, because `deconv_4` (L22) is finite while the `deconv_3` it is
computed from is not, and because `results/ntv3_650m_final` captures the last
transformer block through a forward hook with finite values. The usable layers are
L0 to L3 and L22 to L25. The numbers that matter, all from `layer_curve.txt`:

| layer | rho | margin vs k-mers |
|---|---|---|
| L1 `conv_2` | +0.5655 | +0.1094 [+0.0808, +0.1382] |
| L2 `conv_3` | +0.5607 | |
| L0 `conv_1` | +0.5533 | |
| L24 `deconv_6` | +0.4983 | |
| L25 `deconv_7` | +0.4847 | +0.0287 [+0.0007, +0.0564], on the boundary |

L1 was chosen after seeing the curve, so its margin is optimistic, but the lower
bound is far from zero. Re-running `scripts/ntv3_sweep.py` would fill in the 18
missing layers; nothing above the bottleneck has been measured for 100M at all.

The two `hidden_states[-4]` directories that used to sit beside these
(`results/ntv3_probe`, `results/ntv3_650m_probe`) were `deconv_4` reads, 32 positions
at 8x downsampling. Nothing quoted them and they have been removed.
# Current code versus historical sweeps

The tables below describe historical artifacts. Current `ntv3_sweep.py` uses
explicit transformer final-layer-normalization hooks; its `L0` is not the old
U-Net `conv_1`. Read each run's `meta.json`. `ntv3_unet.py list` maps U-Net stages,
and its `embed --representation` selects them explicitly. Pooling and CV changed
in this cleanup; see [PROTOCOL.md](PROTOCOL.md) before interpreting old scores.
