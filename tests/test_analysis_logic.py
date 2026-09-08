import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import embed_elements
import numpy as np
import pandas as pd
import torch
from artifact_io import atomic_json
from benchmark_models import MODELS
from model_runtime import POOLING_PROTOCOL
from regulatory_benchmark import baseline_features, metrics, primary_embedding, probe_analysis
from validation import out_of_fold_linear
from variant_probe import DEFAULT_LAYER


class AnalysisLogicTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame(
            {
                "sequence_id": ["a", "b", "c"],
                "activity": [1.0, 2.0, 3.0],
                "group": [0, 1, 2],
                "seq": ["ACGT", "AAAA", "GCGC"],
                "s_wt": [1.0, 2.0, 3.0],
            }
        )

    def test_primary_layers_are_prespecified_for_all_four_models(self):
        self.assertEqual(list(MODELS), ["evo2", "ntv3_100m", "ntv3_650m", "dnabert2"])
        self.assertEqual(MODELS["evo2"].layer, "blocks.26.mlp.l3")
        for key in ("ntv3_100m", "ntv3_650m"):
            self.assertEqual(MODELS[key].layer, -1)
            self.assertEqual(MODELS[key].representation, "final_deconvolution")
        self.assertEqual(MODELS["dnabert2"].layer, -1)
        self.assertEqual(DEFAULT_LAYER["dnabert2"], -1)

    def test_raw_likelihood_has_no_cross_unit_rmse(self):
        y = np.linspace(-2, 3, 24)
        likelihood = 17 * y - 1000
        groups = np.repeat(np.arange(12), 2)
        raw = metrics(likelihood, y, groups, 10, 0, activity_scale=False)
        self.assertAlmostEqual(raw["spearman"], 1)
        self.assertIsNone(raw["rmse"])
        calibrated = out_of_fold_linear(likelihood, y, groups, folds=3)
        fitted = metrics(calibrated, y, groups, 10, 0, activity_scale=True)
        self.assertLess(fitted["rmse"], 1e-12)

    def test_variant_kmers_are_first_differences_not_element_features(self):
        table = self.table().assign(seq_ref=["AAAA", "AAAA", "AAAA"])
        element, variant = baseline_features(table, "element"), baseline_features(table, "variant")
        self.assertEqual(element["kmer"].shape, (3, 84))
        np.testing.assert_array_equal(variant["kmer"][1], np.zeros(84))
        self.assertGreater(element["kmer"][1].sum(), 0)
        np.testing.assert_array_equal(element["gc"][:, 0], [0.5, 0, 1])

    def test_primary_probe_reorders_rows_but_rejects_wrong_layer_or_subset(self):
        table, spec = self.table(), MODELS["ntv3_100m"]
        meta = {
            "checkpoint": spec.checkpoint,
            "representation": spec.representation,
            "layer": "-1",
            "primary_pooling": "mean",
            "pooling_protocol": POOLING_PROTOCOL,
            "revision": "a" * 40,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            atomic_json(path / "meta.json", meta)
            table.iloc[::-1].to_csv(path / "elements.csv", index=False)
            np.save(path / "X_mean.npy", np.array([[3.0], [2.0], [1.0]]))
            matrix, _ = primary_embedding(path, table, spec)
            np.testing.assert_array_equal(matrix[:, 0], [1, 2, 3])
            atomic_json(path / "meta.json", {**meta, "representation": "transformer_bottleneck"})
            with self.assertRaisesRegex(ValueError, "substitute"):
                primary_embedding(path, table, spec)
            atomic_json(path / "meta.json", meta)
            table.iloc[:2].to_csv(path / "elements.csv", index=False)
            np.save(path / "X_mean.npy", np.ones((2, 1)))
            with self.assertRaisesRegex(ValueError, "complete shared"):
                primary_embedding(path, table, spec)

    def test_element_extraction_uses_registry_and_keeps_metadata(self):
        table, observed = self.table(), []

        class Adapter:
            label, units = "DNABERT last encoder", "token"

            def __init__(self, **kwargs):
                observed.append(kwargs)

            def encode(self, sequence):
                return torch.tensor([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1, 1, 1])

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "elements"
            with (
                patch.object(embed_elements, "elements", return_value=table),
                patch.object(embed_elements, "file_hash", return_value="hash"),
                patch.dict(embed_elements.ADAPTERS, {"dnabert2": Adapter}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                embed_elements.embed("dnabert2", out, revision="a" * 40)
                matrix, meta = primary_embedding(out, table, MODELS["dnabert2"])
                np.testing.assert_array_equal(matrix, np.tile([2.0, 3.0], (3, 1)))
                self.assertEqual(observed[0]["layer"], -1)
                self.assertEqual(meta["units"], "token")
                self.assertIn("source_sha256", meta)
                with self.assertRaisesRegex(ValueError, "not empty"):
                    embed_elements.embed("dnabert2", out, revision="a" * 40)

    def test_missing_primary_models_are_gaps_not_legacy_substitutes(self):
        table = self.table()
        fake_predictions = {
            key: table.activity.to_numpy() for key in ("gc", "kmer", "training_mean")
        }
        fake_rows = {key: {"rmse": 0} for key in fake_predictions}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("regulatory_benchmark.elements", return_value=table),
            patch(
                "regulatory_benchmark.fitted_readouts", return_value=(fake_predictions, fake_rows)
            ),
        ):
            result = probe_analysis(directory, n_boot=10)
        self.assertEqual(len(result["models"]), 4)
        self.assertTrue(all(row["status"] == "not_computed" for row in result["models"].values()))


if __name__ == "__main__":
    unittest.main()
