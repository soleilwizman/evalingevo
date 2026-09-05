import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evo_epistasis import (add_predictions, differences, load_quartets, metrics,
                            noise_ceiling, rank_feature_contrasts, reconstruct,
                            reverse_complement, seq_id, sequence_only_features)


def fixture(n=15):
    rows = []
    for i in range(n):
        wt, a, b, ab = "AACCGGTT", "ATCCGGTT", "AACCGATT", "ATCCGATT"
        row = {"pair_id": str(i), "group_id": str(i), "condition": "test",
               "pos_a": 2, "pos_b": 6, "distance": 4,
               "y_wt": 0.0, "y_a": -1.0, "y_b": -2.0,
               "y_ab": -2.5 - i / 100, "epsilon": -0.5 + i / 100}
        for state, seq in zip(("wt", "a", "b", "ab"), (wt, a, b, ab)):
            row[f"seq_{state}"], row[f"id_{state}"] = seq, seq_id(seq)
        rows.append(row)
    return pd.DataFrame(rows)


class MVPTests(unittest.TestCase):
    def test_sign_and_sequence_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.csv"
            fixture().to_csv(path, index=False)
            df = load_quartets(path)
            self.assertAlmostEqual(df.epsilon.iloc[0], -0.5)  # expected - observed
            bad = fixture(); bad.loc[0, "epsilon"] *= -1; bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_quartets(path)

    def test_grouped_prediction_and_additive_null(self):
        q = fixture()
        scores = pd.DataFrame({"sequence_id": [seq_id(x) for x in ("AACCGGTT", "ATCCGGTT", "AACCGATT", "ATCCGATT")],
                               "score": [0.0, 1.0, 2.0, 3.0]})
        out, _ = add_predictions(q, scores)
        np.testing.assert_allclose(out.model_interaction, 0)
        self.assertTrue((out.groupby("group_id").fold.nunique() == 1).all())
        self.assertIn("sequence_only_kmer_ridge", out)
        self.assertEqual(sequence_only_features(q).shape, (len(q), 4 * 84 + 3))
        altered = q.copy()
        altered[["y_wt", "y_a", "y_b", "y_ab", "epsilon"]] += 7.0
        np.testing.assert_allclose(sequence_only_features(q), sequence_only_features(altered))
        self.assertIsNone(metrics([1, 2, 3], [0, 0, 0])["spearman"])

    def test_reverse_complement_and_sae_sign(self):
        self.assertEqual(reverse_complement("AACCGT"), "ACGGTT")
        x = np.array([[1, 2], [2, 4], [3, 8], [7, 20]])  # WT,A,B,AB
        ranked = rank_feature_contrasts(x, top_k=1)
        self.assertEqual(int(ranked.feature.iloc[0]), 1)
        self.assertAlmostEqual(ranked.contrast.iloc[0], -10)

    def test_noise_ceiling(self):
        summary = noise_ceiling(np.array([-2.0, 0.0, 2.0, 4.0]),
                                np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["epsilon_variance_population"], 5.0)
        self.assertAlmostEqual(summary["mean_epsilon_se_squared"], 0.075)
        self.assertAlmostEqual(summary["reliability"], 0.985)
        self.assertAlmostEqual(summary["perfect_predictor_observed_correlation_ceiling"], np.sqrt(0.985))
        self.assertEqual(summary["distinguishable_n_abs_epsilon_ge_z_se"], 3)

    def test_source_oligo_reconstruction(self):
        genome = list("A" * 210); genome[109] = "G"; genome = "".join(genome)
        ref, ref2 = genome[:200], genome[10:]
        fasta = {"chr1:100:A:C_allele1_oligo": ref,
                 "chr1:100:A:C_allele2_oligo": ref[:99] + "C" + ref[100:],
                 "chr1:110:G:T_allele1_oligo": ref2,
                 "chr1:110:G:T_allele2_oligo": ref2[:99] + "T" + ref2[100:]}
        seqs, chrom, start, end = reconstruct(
            SimpleNamespace(v1="chr1:100:A:C", v2="chr1:110:G:T"), fasta)
        self.assertEqual((chrom, start, end), ("chr1", 1, 200))
        self.assertEqual([len(differences(ref, s)) for s in seqs], [0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
