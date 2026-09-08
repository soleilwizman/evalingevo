"""Sequence validation and reconstruction of the measured quartet dataset."""

import hashlib
import itertools
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("wt", "a", "b", "ab")
SOURCE = "https://zenodo.org/records/15297965"
CONTRAST = "A+B-WT-AB (expected additive minus observed double)"
KMER_VOCAB = tuple(
    "".join(chars) for k in (1, 2, 3) for chars in itertools.product("ACGT", repeat=k)
)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seq_id(seq):
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def differences(ref, seq):
    if len(ref) != len(seq):
        raise ValueError("Only equal-length substitutions are supported")
    return [(i, r, a) for i, (r, a) in enumerate(zip(ref, seq)) if r != a]


def mutate(ref, changes):
    out = list(ref)
    for i, old, new in changes:
        if ref[i] != old:
            raise ValueError("Reference allele mismatch")
        out[i] = new
    return "".join(out)


def load_quartets(path):
    """Load and strictly validate one K562 row per WT/A/B/AB quartet."""
    df = pd.read_csv(path)
    required = {
        "pair_id",
        "group_id",
        "condition",
        "pos_a",
        "pos_b",
        "distance",
        "epsilon",
    }
    required |= {f"{kind}_{state}" for kind in ("seq", "id", "y") for state in STATES}
    if missing := required - set(df):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df.pair_id.duplicated().any() or df[list(required)].isna().any().any():
        raise ValueError("Quartets must be nonempty, unique, and complete")
    if df.condition.nunique() != 1:
        raise ValueError("Evaluate one experimental condition at a time")
    for state in STATES:
        seq = df[f"seq_{state}"].astype(str)
        if not seq.str.fullmatch("[ACGT]+", na=False).all():
            raise ValueError("Sequences must be uppercase A/C/G/T")
        if not all(seq_id(s) == i for s, i in zip(seq, df[f"id_{state}"])):
            raise ValueError("Sequence hash mismatch")
        df[f"y_{state}"] = pd.to_numeric(df[f"y_{state}"], errors="raise")
    numeric = [f"y_{s}" for s in STATES] + ["epsilon", "distance", "pos_a", "pos_b"]
    if not np.isfinite(df[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite measurement or coordinate")
    if "epsilon_se" in df:
        df["epsilon_se"] = pd.to_numeric(df["epsilon_se"], errors="raise")
        if not np.isfinite(df.epsilon_se.to_numpy(float)).all() or (df.epsilon_se < 0).any():
            raise ValueError("epsilon_se must be finite and nonnegative")
    expected = df.y_a + df.y_b - df.y_wt - df.y_ab
    if not np.allclose(df.epsilon, expected, atol=1e-8, rtol=1e-8):
        raise ValueError(f"epsilon must use {CONTRAST}")
    for row in df.itertuples():
        da, db, dab = (
            differences(row.seq_wt, row.seq_a),
            differences(row.seq_wt, row.seq_b),
            differences(row.seq_wt, row.seq_ab),
        )
        if len(da) != 1 or len(db) != 1 or sorted(da + db) != dab:
            raise ValueError(f"{row.pair_id}: sequences are not a complete two-SNV quartet")
        if (da[0][0] + 1, db[0][0] + 1, db[0][0] - da[0][0]) != (
            row.pos_a,
            row.pos_b,
            row.distance,
        ):
            raise ValueError(f"{row.pair_id}: mutation coordinates do not match sequences")
    return df


def parse_variant(value):
    value = value.replace("chr:", "chr")
    match = re.fullmatch(r"(chr[0-9XY]+):(\d+):([ACGT]):([ACGT])", value)
    if not match or match.group(3) == match.group(4):
        raise ValueError("not_biallelic_SNV")
    chrom, pos, ref, alt = match.groups()
    return value, chrom, int(pos), ref, alt


def read_oligos(code_zip, wanted):
    found = {}

    def add(key, value):
        if key in found and found[key] != value:
            raise ValueError(f"Conflicting FASTA records for {key}")
        found[key] = value

    with zipfile.ZipFile(code_zip) as archive:
        for name in archive.namelist():
            if "/dockerfiles/mpra_chr" not in name or not name.endswith(".fasta"):
                continue
            key, parts = None, []
            with archive.open(name) as handle:
                for raw in handle:
                    line = raw.decode().strip()
                    if line.startswith(">"):
                        if key in wanted:
                            add(key, "".join(parts))
                        key, parts = line[1:].split()[0], []
                    elif key in wanted:
                        parts.append(line)
                if key in wanted:
                    add(key, "".join(parts))
    return found


def reconstruct(row, fasta):
    v1, chrom1, p1, r1, a1 = parse_variant(row.v1)
    v2, chrom2, p2, r2, a2 = parse_variant(row.v2)
    if chrom1 != chrom2 or p1 == p2:
        raise ValueError("different_chromosome_or_same_site")
    ref, single1 = fasta.get(v1 + "_allele1_oligo"), fasta.get(v1 + "_allele2_oligo")
    ref2, alt2 = fasta.get(v2 + "_allele1_oligo"), fasta.get(v2 + "_allele2_oligo")
    if None in (ref, single1, ref2, alt2):
        raise ValueError("missing_oligo")
    if any(len(s) != 200 or not re.fullmatch("[ACGT]+", s) for s in (ref, single1, ref2, alt2)):
        raise ValueError("not_200nt_unambiguous_DNA")
    if differences(ref, single1) != [(99, r1, a1)] or differences(ref2, alt2) != [(99, r2, a2)]:
        raise ValueError("central_allele_or_coordinate_mismatch")
    index2 = 99 + p2 - p1
    if not 10 <= index2 < 190 or ref[index2] != r2:
        raise ValueError("second_variant_outside_window_or_reference_mismatch")
    start1, start2 = p1 - 99, p2 - 99
    lo, hi = max(start1, start2), min(start1 + 200, start2 + 200)
    if ref[lo - start1 : hi - start1] != ref2[lo - start2 : hi - start2]:
        raise ValueError("reference_oligo_overlap_mismatch")
    single2 = mutate(ref, [(index2, r2, a2)])
    double = mutate(single1, [(index2, r2, a2)])
    return [ref, single1, single2, double], chrom1, start1, start1 + 199


def prepare_siraj(windows_path, code_zip, out, cell="K562", min_dna=20.0, max_se=0.5):
    """Reconstruct the fixed 200-nt K562 middle-window benchmark and its audit trail."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(windows_path, sep="\t")
    source_rows = len(source)
    source = source[
        (source.cell_type == cell) & (source.window == "middle") & (source.center_variant == "var1")
    ].copy()
    source[["v1", "v2"]] = source[["v1", "v2"]].replace("chr:", "chr", regex=True)
    source = source.sort_values(["v1", "v2", "library", "v1v2_construct"])
    if source.duplicated(["v1", "v2", "library"]).any():
        raise ValueError("Unexpected duplicate pair within a library")
    duplicate = source.duplicated(["v1", "v2"])
    audit = [
        dict(status="not_selected", reason="lexicographic_library_rule", **r)
        for r in source[duplicate].to_dict("records")
    ]
    source = source[~duplicate].copy()
    wanted = {
        v + suffix
        for v in set(source.v1) | set(source.v2)
        for suffix in ("_allele1_oligo", "_allele2_oligo")
    }
    fasta, accepted = read_oligos(code_zip, wanted), []
    for row in source.itertuples(index=False):
        pair_id = f"{row.v1};{row.v2};var1;middle;{row.library}"
        try:
            seqs, chrom, start, end = reconstruct(row, fasta)
            names = ("refref", "altref", "refalt", "altalt")
            y = np.array(
                [0.0, row.altref_log2Skew, row.refalt_log2Skew, row.altalt_log2Skew],
                float,
            )
            se = np.array([getattr(row, f"{x}_Log2FC_SE") for x in names], float)
            dna = np.array([getattr(row, f"mean_Plasmid_{x}") for x in names], float)
            if not np.isfinite(np.r_[y, se, dna]).all() or (se < 0).any() or (dna < 0).any():
                raise ValueError("nonfinite_or_invalid_measurement")
            if (dna < min_dna).any():
                raise ValueError("mean_DNA_below_threshold")
            if (se > max_se).any():
                raise ValueError("activity_SE_above_threshold")
            epsilon = y[1] + y[2] - y[0] - y[3]
            if not np.isfinite(row.int_log2Skew) or not np.isclose(
                epsilon, -row.int_log2Skew, atol=1e-6, rtol=1e-6
            ):
                raise ValueError("interaction_contrast_mismatch")
            accepted.append((pair_id, seqs, y, row.int_log2SkewSE, chrom, start, end))
            audit.append(dict(status="accepted", reason="", **row._asdict()))
        except ValueError as error:
            audit.append(
                dict(
                    status="excluded",
                    reason=str(error),
                    pair_id=pair_id,
                    **row._asdict(),
                )
            )
    rows, active_chrom, active_end, group = [], None, -1, -1
    for pair_id, seqs, y, epsilon_se, chrom, start, end in sorted(
        accepted, key=lambda x: (x[4], x[5], x[6])
    ):
        if chrom != active_chrom or start > active_end:
            group += 1
            active_chrom, active_end = chrom, end
        active_end = max(active_end, end)
        singles = sorted([(differences(seqs[0], seqs[i])[0][0], seqs[i], y[i]) for i in (1, 2)])
        ordered_seq = [seqs[0], singles[0][1], singles[1][1], seqs[3]]
        ordered_y = [y[0], singles[0][2], singles[1][2], y[3]]
        pos_a, pos_b = singles[0][0] + 1, singles[1][0] + 1
        row = {
            "pair_id": f"{pair_id}|{cell}|{seq_id(ordered_seq[3])[:16]}",
            "background_id": pair_id,
            "condition": cell,
            "group_id": f"{cell}_region_{group:05d}",
            "pos_a": pos_a,
            "pos_b": pos_b,
            "distance": pos_b - pos_a,
        }
        for state, seq, value in zip(STATES, ordered_seq, ordered_y):
            row.update({f"seq_{state}": seq, f"id_{state}": seq_id(seq), f"y_{state}": value})
        row.update(
            epsilon=ordered_y[1] + ordered_y[2] - ordered_y[0] - ordered_y[3],
            epsilon_se=epsilon_se,
        )
        rows.append(row)
    quartets = pd.DataFrame(rows)
    if quartets.empty:
        raise ValueError("No accepted quartets")
    qpath, apath = out / "quartets.csv.gz", out / "audit.csv.gz"
    quartets.to_csv(qpath, index=False)
    pd.DataFrame(audit).to_csv(apath, index=False)
    load_quartets(qpath)
    report = {
        "source": SOURCE,
        "source_rows": source_rows,
        "cell": cell,
        "window": "middle",
        "center_variant": "var1",
        "contrast": CONTRAST,
        "source_int_log2Skew_relation": "epsilon = -int_log2Skew",
        "library_rule": "lexicographic_first_before_QC",
        "candidate_pairs": len(source),
        "duplicate_libraries_not_selected": int(duplicate.sum()),
        "accepted_pairs": len(quartets),
        "excluded_pairs": int(sum(x["status"] == "excluded" for x in audit)),
        "overlap_groups": int(quartets.group_id.nunique()),
        "min_mean_DNA": min_dna,
        "max_activity_SE_log2": max_se,
        "all_windows_sha256": file_hash(windows_path),
        "code_zip_sha256": file_hash(code_zip),
        "original_haplotype_design_independently_verified": False,
    }
    (out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
