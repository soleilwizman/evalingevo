#!/usr/bin/env python3
"""Variant probe across model families, on one protocol.

    # GPU, once per model
    python3 variant_probe.py embed --model evo2     --out results/vp_evo2
    python3 variant_probe.py embed --model ntv3     --out results/vp_ntv3_100m \
        --checkpoint InstaDeepAI/NTv3_100M_pre
    python3 variant_probe.py embed --model ntv3     --out results/vp_ntv3_650m \
        --checkpoint InstaDeepAI/NTv3_650M_pre
    python3 variant_probe.py embed --model dnabert2 --out results/vp_dnabert2

    # CPU
    python3 variant_probe.py probe   --embeddings results/vp_evo2
    python3 variant_probe.py compare --embeddings results/vp_evo2 results/vp_ntv3_100m \
        results/vp_ntv3_650m results/vp_dnabert2

Each model is read in its OWN units, not forced into a shared one.

  Evo 2 and NTv3 read one base at a time, so a sequence is 200 vectors and the
  variant has a vector of its own.
  DNABERT-2 uses BPE, so a sequence is about 40 vectors, each covering four or
  five bases, and the variant has no vector of its own. It gets the vector of
  the token containing it.

Sequence pooling is a plain mean over whatever those units are, which is what
each model's own documentation recommends. An earlier version broadcast
DNABERT-2's token vectors out to bases before pooling, which is a
length-weighted mean over tokens and not a representation the model produces.

The consequence to keep in mind when reading the results: the variant-level
feature is a single base for Evo 2 and NTv3 and a four-to-five base chunk for
DNABERT-2. That is each model at its native resolution, which is the fair
comparison, but it is not the same quantity.

Three feature sets per model, and the third decides what a win means:

  d_mean   mutant mean-pooled minus reference mean-pooled, averaged over 200
           positions, so one changed base is heavily diluted
  d_pos    the same difference read only at the base that changed
  wt_mean  the reference's own embedding, which knows nothing about which base
           changed. Some elements are more fragile than others, so a probe can
           score by recognising a sensitive element. If this matches the other
           two, that is what happened.

GC is not a separate row. For a single substitution the GC change is +1, -1 or
0, which is contained in the first four columns of the k-mer difference
baseline, so it is covered rather than omitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_epistasis import load_quartets
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evo_probe import ALPHAS, kmers, paired_interval

AUDIT = "data/audit.csv.gz"

QUARTETS = "data/quartets.csv.gz"
NTV3_MULTIPLE = 128
SEED = 0


def out_of_fold(X, y, groups, folds=5, seed=SEED):
    """Ridge with alpha chosen inside each training fold, never across it.

    Seeded, unlike evo_probe.out_of_fold. Unseeded GroupKFold assigns groups by
    a rule that has changed between scikit-learn versions, so the same data and
    the same code give different numbers on different machines. Every figure
    from this script is reproducible; the element-level numbers from evo_probe
    are not, and the two are therefore not directly comparable.
    """
    X = np.asarray(X, float)
    if not np.isfinite(X).all():
        raise ValueError("features contain NaN or infinity")
    prediction = np.empty(len(y))
    split = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in split.split(X, y, groups):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(X[train], y[train])
        prediction[test] = model.predict(X[test])
    return prediction


def observations(quartets_path=QUARTETS):
    """One row per single-variant measurement: two per quartet, stacked."""
    q = load_quartets(quartets_path)
    frames = []
    for side in ("a", "b"):
        frames.append(pd.DataFrame({
            "wt_id": q.id_wt, "mut_id": q[f"id_{side}"],
            "wt_seq": q.seq_wt, "mut_seq": q[f"seq_{side}"],
            "position": q[f"pos_{side}"], "effect": q[f"y_{side}"],
            "group": q.group_id, "side": side}))
    obs = pd.concat(frames, ignore_index=True)
    # pos_a and pos_b are 1-based, so the changed base sits at position - 1.
    # The check below refuses to run if that is ever untrue, because reading one
    # base off the variant produces a clean and entirely plausible null.
    obs["index"] = obs.position - 1
    bad = [r.wt_seq[r.index] == r.mut_seq[r.index] or
           sum(a != b for a, b in zip(r.wt_seq, r.mut_seq)) != 1
           for r in obs.itertuples(index=False)]
    if any(bad):
        raise SystemExit(f"{sum(bad)} of {len(obs)} rows do not differ by exactly one "
                         "base at the stated position; the position convention is wrong")
    # 207 variants are measured in more than one quartet and those repeats
    # disagree, by up to 0.20 in effect. Keeping whichever came first would be an
    # arbitrary choice between two real measurements, so they are averaged.
    keys = ["wt_id", "mut_id", "index"]
    repeats = obs.groupby(keys).effect.transform("size")
    averaged = (obs.groupby(keys, as_index=False)
                   .agg(wt_seq=("wt_seq", "first"), mut_seq=("mut_seq", "first"),
                        position=("position", "first"), effect=("effect", "mean"),
                        group=("group", "first"), side=("side", "first"),
                        replicates=("effect", "size")))
    if repeats.max() > 1:
        print(f"{int((repeats > 1).sum())} rows are repeated measurements of "
              f"{int((averaged.replicates > 1).sum())} variants; averaged", flush=True)
    return averaged.reset_index(drop=True)


# ------------------------------------------------------------------- adapters

class Evo2Adapter:
    """Autoregressive, one token per base, so positions map straight through."""

    def __init__(self, checkpoint="evo2_7b_base", layer="blocks.26.mlp.l3", weights=None):
        import torch
        from evo2 import Evo2
        self.torch = torch
        self.model = Evo2(checkpoint, local_path=weights) if weights else Evo2(checkpoint)
        names = [n for n, _ in self.model.model.named_modules()]
        if layer not in names:
            near = [n for n in names if n.endswith(layer.split(".")[-1])][:20]
            raise SystemExit(f"no module named {layer}. Candidates:\n  " + "\n  ".join(near))
        self.grabbed = {}
        self.handle = dict(self.model.model.named_modules())[layer].register_forward_hook(
            lambda _m, _i, o: self.grabbed.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o).detach()))
        self.label = f"Evo 2 {checkpoint} {layer}"

    units = "base"

    def encode(self, sequence):
        """One vector per base, and the identity map from base to unit."""
        ids = self.torch.tensor(self.model.tokenizer.tokenize(sequence),
                                dtype=self.torch.int, device="cuda:0").unsqueeze(0)
        self.grabbed.clear()
        with self.torch.inference_mode():
            self.model.model(ids)
        h = self.grabbed["h"].float()
        h = h[0] if h.dim() == 3 else h
        if h.shape[0] != len(sequence):
            raise SystemExit(f"layer gave {h.shape[0]} positions for {len(sequence)} bases")
        return h, np.arange(len(sequence))

    def close(self):
        self.handle.remove()


class NTv3Adapter:
    """Masked LM, one token per base, but the U-Net needs a length divisible by
    128 and the padding must be N rather than the pad token."""

    def __init__(self, checkpoint="InstaDeepAI/NTv3_100M_pre", layer=-4, revision="main"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch, self.layer = torch, layer
        kw = {"trust_remote_code": True, "revision": revision}
        self.tok = AutoTokenizer.from_pretrained(checkpoint, **kw)
        self.model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kw).float()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)
        probe = "ACGT" * (NTV3_MULTIPLE // 4)
        ids = self.tok(probe, add_special_tokens=True)["input_ids"]
        want = self.tok.convert_tokens_to_ids(list("ACGT"))
        self.offset = next(i for i in range(len(ids) - len(probe) + 1)
                           if list(ids[i:i + 4]) == list(want) and len(ids) - i >= len(probe))
        self.label = f"NTv3 {checkpoint.split('/')[-1]} hidden_states[{layer}]"

    units = "base"

    def encode(self, sequence):
        """One vector per base, after stripping the N padding back off."""
        target = -(-len(sequence) // NTV3_MULTIPLE) * NTV3_MULTIPLE
        left = (target - len(sequence)) // 2
        padded = "N" * left + sequence + "N" * (target - len(sequence) - left)
        ids = self.torch.tensor([self.tok(padded, add_special_tokens=True)["input_ids"]],
                                dtype=self.torch.long, device=self.device)
        span = slice(self.offset + left, self.offset + left + len(sequence))
        got = "".join(self.tok.convert_ids_to_tokens(ids[0, span].tolist()))
        if got != sequence:
            raise SystemExit(f"token alignment wrong: {got[:20]} vs {sequence[:20]}")
        with self.torch.inference_mode():
            states = self.model(input_ids=ids, output_hidden_states=True).hidden_states
        return states[self.layer][0, span].float(), np.arange(len(sequence))

    def close(self):
        pass


class DNABERT2Adapter:
    """BPE, so a base has no token of its own. Each base takes the vector of the
    token covering it, located independently in the reference and the mutant."""

    def __init__(self, checkpoint="zhihan1996/DNABERT-2-117M", layer=10):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.layer = torch, layer
        self.tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        try:
            self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        except Exception as first:
            from transformers.models.bert.configuration_bert import BertConfig
            try:
                self.model = AutoModel.from_pretrained(
                    checkpoint, trust_remote_code=True,
                    config=BertConfig.from_pretrained(checkpoint))
            except Exception as second:
                raise SystemExit(f"could not load {checkpoint}\n  direct: {first}\n"
                                 f"  with BertConfig: {second}\n"
                                 "This checkpoint targets transformers 4.28; try "
                                 "`pip install 'transformers<5'` in a separate venv.")
        self.model = self.model.float()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)
        blocks = None
        for path in ("encoder.layer", "bert.encoder.layer"):
            node = self.model
            try:
                for part in path.split("."):
                    node = getattr(node, part)
                blocks = list(node)
                break
            except (AttributeError, TypeError):
                continue
        if blocks is None:
            names = [n for n, _ in self.model.named_modules() if n.count(".") <= 2][:40]
            raise SystemExit("cannot find the blocks. Modules:\n  " + "\n  ".join(names))
        index = layer if layer >= 0 else len(blocks) + layer
        self.grabbed = {}
        self.handle = blocks[index].register_forward_hook(
            lambda _m, _i, o: self.grabbed.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o).detach()))
        self.label = f"DNABERT-2 block {index} of {len(blocks)}"

    def _base_to_token(self, sequence, token_ids):
        """Which token covers each base. Decoding piece by piece rather than
        using offset mappings, which this tokenizer may not provide."""
        pieces = self.tok.convert_ids_to_tokens(token_ids)
        specials = set(self.tok.all_special_tokens)
        covered = sum(len(piece.replace("##", "")) for piece in pieces
                      if piece not in specials)
        if covered != len(sequence):
            raise SystemExit(f"tokens cover {covered} bases for a {len(sequence)} base "
                             "sequence; this vocabulary is not a plain BPE over ACGT")
        mapping, base = np.full(len(sequence), -1), 0
        for position, piece in enumerate(pieces):
            if piece in specials:
                continue
            for _ in range(len(piece.replace("##", ""))):
                mapping[base] = position
                base += 1
        return mapping

    units = "token"

    def encode(self, sequence):
        """One vector per real token, plus which token covers each base.

        The token vectors are returned as the model produced them. Nothing is
        broadcast out to bases: pooling here is a plain mean over tokens, which
        is what this checkpoint's own usage example does.
        """
        token_ids = self.tok(sequence)["input_ids"]
        mapping = self._base_to_token(sequence, token_ids)
        ids = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.device)
        self.grabbed.clear()
        with self.torch.inference_mode():
            self.model(ids)
        h = self.grabbed["h"].float()
        h = h[0] if h.dim() == 3 else h
        if h.shape[0] != len(token_ids):
            raise SystemExit(f"block gave {h.shape[0]} positions for {len(token_ids)} tokens")
        pieces = self.tok.convert_ids_to_tokens(token_ids)
        specials = set(self.tok.all_special_tokens)
        real = [i for i, piece in enumerate(pieces) if piece not in specials]
        # mapping indexes into the full token list, so shift it onto the real ones
        shift = {full: position for position, full in enumerate(real)}
        return h[real], np.array([shift[int(i)] for i in mapping])

    def close(self):
        self.handle.remove()


ADAPTERS = {"evo2": Evo2Adapter, "ntv3": NTv3Adapter, "dnabert2": DNABERT2Adapter}
DEFAULT_CHECKPOINT = {"evo2": "evo2_7b_base", "ntv3": "InstaDeepAI/NTv3_100M_pre",
                      "dnabert2": "zhihan1996/DNABERT-2-117M"}
DEFAULT_LAYER = {"evo2": "blocks.26.mlp.l3", "ntv3": -4, "dnabert2": 10}


def embed(out, model="evo2", checkpoint=None, layer=None, weights=None,
          quartets=QUARTETS, limit=0):
    obs = observations(quartets)
    if limit:
        obs = obs.head(limit)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)

    needed = {}
    for row in obs.itertuples(index=False):
        needed.setdefault(row.wt_id, {"seq": row.wt_seq, "pos": set()})["pos"].add(row.index)
        needed.setdefault(row.mut_id, {"seq": row.mut_seq, "pos": set()})["pos"].add(row.index)

    checkpoint = checkpoint or DEFAULT_CHECKPOINT[model]
    layer = DEFAULT_LAYER[model] if layer is None else layer
    kwargs = {"checkpoint": checkpoint, "layer": layer}
    if model == "evo2" and weights:
        kwargs["weights"] = weights
    adapter = ADAPTERS[model](**kwargs)
    print(f"{adapter.label}: {len(obs)} observations, {len(needed)} distinct sequences",
          flush=True)

    pooled, at_variant, unit_counts = {}, {}, []
    try:
        for n, (sequence_id, item) in enumerate(needed.items()):
            h, base_to_unit = adapter.encode(item["seq"])
            # Plain mean over the model's own units. For DNABERT-2 that is a mean
            # over ~40 tokens, not over 200 bases, so no token is weighted by how
            # many bases it happens to cover.
            pooled[sequence_id] = h.mean(0).cpu().numpy()
            unit_counts.append(int(h.shape[0]))
            for position in item["pos"]:
                at_variant[(sequence_id, position)] = h[base_to_unit[position]].cpu().numpy()
            if n % 250 == 0:
                print(f"  {n}/{len(needed)}", flush=True)
    finally:
        adapter.close()
    print(f"pooled over {adapter.units}s: "
          f"{np.mean(unit_counts):.1f} per sequence on average")

    rows = list(obs.itertuples(index=False))
    matrices = {
        "d_mean": np.stack([pooled[r.mut_id] - pooled[r.wt_id] for r in rows]),
        "d_pos": np.stack([at_variant[(r.mut_id, r.index)] - at_variant[(r.wt_id, r.index)]
                           for r in rows]),
        "wt_mean": np.stack([pooled[r.wt_id] for r in rows]),
    }
    for name, matrix in matrices.items():
        np.save(out / f"X_{name}.npy", matrix)
    obs.drop(columns=["wt_seq", "mut_seq"]).to_csv(out / "observations.csv", index=False)
    obs[["wt_id", "wt_seq", "mut_id", "mut_seq", "position", "index"]].to_csv(
        out / "sequences.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"model": model, "checkpoint": checkpoint, "layer": str(layer),
         "label": adapter.label, "units": adapter.units,
         "units_per_sequence": float(np.mean(unit_counts)),
         "n_observations": len(obs), "n_sequences": len(needed),
         "width": int(matrices["d_mean"].shape[1])},
        indent=2) + "\n")
    print(f"\nwrote three matrices to {out}, width {matrices['d_mean'].shape[1]}")


def noise_ceiling(obs, quartets_path=QUARTETS, audit_path=AUDIT):
    """How high can any predictor of the SIGNED effect reach here?

    Each single-variant measurement carries a standard error, so part of the
    spread in the target is assay noise that nothing can predict. Reliability is
    (Var(y) - mean SE^2) / Var(y), and its square root bounds the correlation.

    Returns None if the audit table is missing, since it is optional.

    Only defined for the signed target: attenuation is a linear-model result and
    taking absolute values is not a linear operation, so the magnitude target has
    no ceiling from this formula.
    """
    if not Path(audit_path).exists():
        return None
    q = load_quartets(quartets_path)
    audit = pd.read_csv(audit_path)
    key = ["v1", "v2", "center_variant", "window", "library"]
    if any(c not in audit.columns for c in key):
        return None
    joined = audit[key[0]].astype(str)
    for column in key[1:]:
        joined = joined + ";" + audit[column].astype(str)
    audit = audit.assign(_k=joined)
    wanted = ["_k", "altref_log2SkewSE", "refalt_log2SkewSE"]
    if any(c not in audit.columns for c in wanted):
        return None
    q = q.assign(_k=q.pair_id.str.split("|").str[0]).merge(
        audit.drop_duplicates("_k")[wanted], on="_k", how="left")
    se = pd.concat([
        pd.DataFrame({"wt_id": q.id_wt, "mut_id": q.id_a, "index": q.pos_a - 1,
                      "se": q.altref_log2SkewSE}),
        pd.DataFrame({"wt_id": q.id_wt, "mut_id": q.id_b, "index": q.pos_b - 1,
                      "se": q.refalt_log2SkewSE})]).groupby(
        ["wt_id", "mut_id", "index"], as_index=False).se.mean()
    merged = obs.merge(se, on=["wt_id", "mut_id", "index"], how="left")
    if merged.se.isna().any():
        return None
    variance = float(np.var(merged.effect, ddof=0))
    noise = float(np.mean(merged.se ** 2))
    reliability = (variance - noise) / variance
    return {"variance": variance, "mean_se_squared": noise,
            "reliability": reliability, "ceiling": float(np.sqrt(max(reliability, 0.0)))}


def within_element_accuracy(prediction, y, groups):
    """Can it rank the variants INSIDE an element?

    The overall Spearman mixes two things: telling a fragile element from a
    robust one, and telling which base inside an element matters. Only the
    second is variant interpretation, and it is the smaller of the two here,
    since roughly half the variance in effect size sits between elements.

    Every within-element pair is scored on whether the predicted ordering
    matches the measured one. Chance is 0.500 by construction: no amount of
    knowing the element can help, because both members share it.
    """
    frame = pd.DataFrame({"p": prediction, "y": y, "g": groups})
    correct = total = 0
    for _, block in frame.groupby("g"):
        if len(block) < 2:
            continue
        p_values, y_values = block.p.to_numpy(), block.y.to_numpy()
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if y_values[i] == y_values[j]:
                    continue
                total += 1
                correct += (p_values[i] > p_values[j]) == (y_values[i] > y_values[j])
    return (correct / total if total else float("nan")), total


def build_features(directory, obs):
    d = Path(directory)
    seqs = pd.read_csv(d / "sequences.csv")
    position = np.column_stack([obs["index"].values,
                                (obs.side == "a").astype(int).values])
    return {
        "variant position + side (2)": position,
        "k-mer difference (84)": kmers(seqs.mut_seq.values) - kmers(seqs.wt_seq.values),
        "reference embedding only (control)": np.load(d / "X_wt_mean.npy"),
        "difference, mean pooled": np.load(d / "X_d_mean.npy"),
        "difference, at the variant unit": np.load(d / "X_d_pos.npy"),
    }


def score_one(embeddings, target="magnitude", folds=5, seed=SEED,
              n_permutations=20, quiet=False):
    d = Path(embeddings)
    obs = pd.read_csv(d / "observations.csv")
    meta = json.loads((d / "meta.json").read_text())
    y = np.abs(obs.effect.values) if target == "magnitude" else obs.effect.values
    g = obs.group.values
    features = build_features(d, obs)

    preds, rows = {}, []
    for name, X in features.items():
        preds[name] = out_of_fold(X, y, g, folds, seed)
        accuracy, n_pairs = within_element_accuracy(preds[name], y, g)
        rows.append((name, float(spearmanr(preds[name], y).statistic),
                     float(np.sqrt(np.mean((preds[name] - y) ** 2))), accuracy))
    rows.append(("predict the mean", 0.0, float(np.sqrt(np.mean((y - y.mean()) ** 2))), 0.5))
    named = dict((r[0], r) for r in rows)
    best = max((named["difference, mean pooled"],
                named["difference, at the variant unit"]), key=lambda r: r[1])

    ceiling = noise_ceiling(obs) if target == "signed" else None
    if not quiet:
        print(f"n={len(obs)}  {meta['label']}  width {meta['width']}  target {target}")
        print(f"pooled over {meta.get('units', 'base')}s, "
              f"{meta.get('units_per_sequence', 200):.1f} per sequence, "
              f"grouped {folds}-fold seed {seed}")
        if ceiling:
            print(f"measurement reliability {ceiling['reliability']:.4f}, so no predictor "
                  f"of the signed effect can exceed {ceiling['ceiling']:.4f}")
        elif target == "magnitude":
            print("no ceiling for the magnitude target: attenuation is a linear-model "
                  "result and |y| is not linear. Run with --target signed for it.")
        print()
        _, n_pairs = within_element_accuracy(preds[best[0]], y, g)
        width = max(len(r[0]) for r in rows)
        print(f"{'':{width}}   Spearman     RMSE   within-element")
        for name, rho, rmse, accuracy in rows:
            print(f"{name:{width}}   {rho:+.4f}   {rmse:.4f}   {accuracy:.3f}")
        print(f"\nwithin-element column: {n_pairs} pairs sharing an element, ranked "
              "on which variant matters more. Chance is 0.500 and knowing the "
              "element cannot help, so this is the variant question on its own.")
        rng = np.random.default_rng(0)
        null = [spearmanr(out_of_fold(features[best[0]], rng.permutation(y), g, folds, seed),
                          y).statistic for _ in range(n_permutations)]
        print(f"\nnoise floor from {n_permutations} label permutations: "
              f"{np.mean(null):+.4f} +/- {np.std(null):.4f}")

    rivals = [named["variant position + side (2)"], named["k-mer difference (84)"],
              named["reference embedding only (control)"]]
    intervals, beaten = {}, []
    for rival in rivals:
        low, high = paired_interval(preds[best[0]], preds[rival[0]], y, g)
        intervals[rival[0]] = (best[1] - rival[1], low, high)
        beaten.append(low > 0)
        if not quiet:
            print(f"\n{best[0]} minus {rival[0]}: {best[1] - rival[1]:+.4f}  "
                  f"95% interval [{low:+.4f}, {high:+.4f}]")

    control_low = intervals["reference embedding only (control)"][1]
    verdict = ("locates the variant" if all(beaten)
               else "no better than the element control" if control_low <= 0
               else "beats the control but not every baseline")
    if not quiet:
        print(f"\n{verdict}.")
    return {"label": meta["label"], "units": meta.get("units", "base"),
            "ceiling": ceiling, "rows": rows,
            "best": best[0], "spearman": best[1], "intervals": intervals,
            "verdict": verdict}


def probe(embeddings, target="magnitude", folds=5, seed=SEED):
    score_one(embeddings, target, folds, seed)


def compare(embeddings, target="magnitude", folds=5, seed=SEED):
    # The baselines are printed once, from the first run, so every run must be
    # over the same rows. Different n means different observations, and the
    # shared baseline line would silently describe only one of them.
    # Count the rows on disk rather than trusting meta.json, which records what
    # the embed run intended rather than what is actually there.
    counts = {d: len(pd.read_csv(Path(d) / "observations.csv")) for d in embeddings}
    if len(set(counts.values())) > 1:
        raise SystemExit("these runs cover different numbers of observations, so "
                         "their numbers are not comparable:\n  " +
                         "\n  ".join(f"{d}: {n}" for d, n in counts.items()) +
                         "\nRe-embed them from the same quartets file without --limit.")
    results = [score_one(d, target, folds, seed, quiet=True) for d in embeddings]
    print(f"target {target}, grouped {folds}-fold seed {seed}, "
          f"n={next(iter(counts.values()))}\n")
    print("baselines, identical for every model:")
    for name, rho, _ in results[0]["rows"]:
        if name in ("variant position + side (2)", "k-mer difference (84)",
                    "predict the mean"):
            print(f"  {name:<36} {rho:+.4f}")
    width = max(len(r["label"]) for r in results)
    print(f"\n{'model':{width}}  units  feature  probe    within   control   probe minus control")
    for r in results:
        by_name = dict((x[0], x) for x in r["rows"])
        control = by_name["reference embedding only (control)"][1]
        within = by_name[r["best"]][3]
        gap, low, high = r["intervals"]["reference embedding only (control)"]
        short = "d_unit" if "unit" in r["best"] else "d_mean"
        print(f"{r['label']:{width}}  {r['units']:<5}  {short}  {r['spearman']:+.4f}  "
              f"{within:.3f}   {control:+.4f}   {gap:+.4f} [{low:+.4f}, {high:+.4f}]  "
              f"{r['verdict']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="GPU. One forward pass per distinct sequence.")
    e.add_argument("--model", choices=tuple(ADAPTERS), default="evo2")
    e.add_argument("--out", required=True)
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--layer", default=None,
                   help="module name for evo2, integer index for ntv3 and dnabert2")
    e.add_argument("--weights", default=None, help="evo2 only")
    e.add_argument("--quartets", default=QUARTETS)
    e.add_argument("--limit", type=int, default=0, help="first N observations, smoke run")

    p = sub.add_parser("probe", help="CPU. One model.")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--target", choices=("magnitude", "signed"), default="magnitude")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED,
                   help="fold assignment; vary it to see how much the split matters")

    c = sub.add_parser("compare", help="CPU. Several models side by side.")
    c.add_argument("--embeddings", nargs="+", required=True)
    c.add_argument("--target", choices=("magnitude", "signed"), default="magnitude")
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--seed", type=int, default=SEED,
                   help="fold assignment; vary it to see how much the split matters")

    args = vars(parser.parse_args())
    cmd = args.pop("cmd")
    if cmd == "embed" and args["layer"] is not None and args["model"] != "evo2":
        args["layer"] = int(args["layer"])
    {"embed": embed, "probe": probe, "compare": compare}[cmd](**args)


if __name__ == "__main__":
    main()
