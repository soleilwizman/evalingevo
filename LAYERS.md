# Which layer every readout uses

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

## Two directories that are NOT interchangeable with the above

`results/ntv3_650m_probe` records `layer: "hidden_states[-4]"`, not block 11. Same
checkpoint and width as `ntv3_650m_final`, different depth. Do not treat the two
as the same readout.

`results/ntv3_650m_sweep` is unusable as committed, for two reasons.

Its `meta.json` records `layers: [0..25]` and 26 matrices are present, but the
650M model has 12 transformer blocks. `find_layers` in `ntv3_sweep.py` selects
the *largest* contiguous module stack it can find, not necessarily the blocks,
and it prints the module names to stdout while writing only indices to
`meta.json`. The log from the run that produced these matrices (`sweep650.log`,
recoverable from commit e1d359d) carries no layer-name line, so **what L0 through
L25 are is not recorded anywhere in the repository.** A later run under the
current script did print names and found 12 layers under `core.transformer_blocks`
(`sweep650_named.log`, same commit), but those are not the committed matrices.

18 of the 26 matrices contain non-finite values. Checked directly:

    non-finite: layers 4-21
    usable:     layers 0, 1, 2, 3, 22, 23, 24, 25

`sweep650.log` shows the run scoring layers 0, 1 and 2 at +0.5533, +0.5655 and
+0.5607 against a k-mer baseline of +0.4560, then failing at layer 3 with
"features contain NaN or infinity". Those three numbers are higher than every
readout in the committed benchmark, including the Evo 2 element probe at +0.5051.
They should not be quoted: the layers are unidentified and the run they come from
did not complete.

Re-running `ntv3_sweep.py` under the current script would fix both problems at
once, since it names the layers it captures.
