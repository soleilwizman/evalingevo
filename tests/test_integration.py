import contextlib
import importlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import embed_variants
import numpy as np
import pandas as pd
from benchmark_data import load_quartets
from epistasis_evaluation import evaluate
from model_runtime import load_hf

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    def test_all_entrypoints_and_compatibility_exports_import(self):
        for path in sorted((ROOT / "scripts").glob("*.py")):
            importlib.import_module(path.stem)
        from evo_epistasis import load_quartets as compatibility_loader
        from model_adapters import NTv3Adapter as adapter
        from variant_evaluation import score_one as evaluator
        from variant_probe import NTv3Adapter, score_one

        self.assertIs(compatibility_loader, load_quartets)
        self.assertIs(NTv3Adapter, adapter)
        self.assertIs(score_one, evaluator)

    def test_cached_scores_produce_complete_evaluation(self):
        quartets = load_quartets(ROOT / "data/quartets.csv.gz")
        groups = quartets.group_id.drop_duplicates().iloc[:15]
        sample = quartets[quartets.group_id.isin(groups)].groupby("group_id", sort=False).head(2)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "quartets.csv"
            sample.to_csv(source, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(
                    source,
                    ROOT / "results/evo2_7b_base/evo_scores.csv",
                    directory / "evaluation",
                    folds=3,
                    n_boot=20,
                )
            self.assertEqual(result["pairs"], len(sample))
            self.assertEqual(result["protocol"], "nested-genomic-group-cv-v2")
            out = directory / "evaluation"
            for name in (
                "predictions.csv",
                "metrics.json",
                "results_table.csv",
                "cases.csv",
                "plots.png",
            ):
                self.assertTrue((out / name).is_file(), name)
            prediction = pd.read_csv(out / "predictions.csv")
            self.assertEqual(prediction.groupby("group_id").fold.nunique().max(), 1)
            np.testing.assert_allclose(
                prediction.model_interaction,
                prediction.flip
                * (prediction.s_a + prediction.s_b - prediction.s_wt - prediction.s_ab),
            )

    def test_model_load_failure_is_not_retried_with_different_configuration(self):
        loader = Mock(side_effect=OSError("network failure"))
        fake = types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=loader),
            AutoModelForMaskedLM=types.SimpleNamespace(from_pretrained=loader),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock()),
        )
        with patch.dict("sys.modules", {"transformers": fake}):
            with self.assertRaisesRegex(OSError, "network failure"):
                load_hf("org/model", "abc", "cpu", family="dnabert2")
        self.assertEqual(loader.call_count, 1)

    def test_missing_or_mismatched_weights_are_rejected(self):
        for info in (
            {"missing_keys": ["cls.predictions.weight"]},
            {"missing_keys": ["encoder.layer.0.weight"]},
            {"mismatched_keys": ["encoder.layer.0.weight"]},
            {"error_msgs": ["incompatible tensors"]},
        ):
            loader = Mock(return_value=(Mock(), info))
            fake = types.SimpleNamespace(
                AutoModel=types.SimpleNamespace(from_pretrained=loader),
                AutoModelForMaskedLM=types.SimpleNamespace(from_pretrained=loader),
                AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock()),
            )
            with self.subTest(info=info), patch.dict("sys.modules", {"transformers": fake}):
                with self.assertRaisesRegex(ValueError, "required weights"):
                    load_hf("org/model", "abc", "cpu", family="dnabert2")

    def test_embedding_resume_preserves_rows_and_rejects_config_changes(self):
        table = pd.DataFrame(
            {
                "sequence_id": ["a", "b", "c"],
                "seq": ["AAAA", "CCCC", "GGGG"],
                "role": ["a", "a", "a"],
                "group_id": [0, 1, 2],
                "pair_id": [0, 1, 2],
            }
        )

        class Embedder:
            representation = "layer"

            def __init__(self, *args):
                self.calls = 0

            def __call__(self, sequences):
                self.calls += 1
                if failing[0] and self.calls == 2:
                    raise RuntimeError("interrupted")
                value = np.full((len(sequences), 2), ord(sequences[0][0]), dtype=np.float32)
                return value, value

        failing = [True]
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                out=directory,
                backend="evo2",
                checkpoint="evo",
                layer="layer",
                revision="main",
                quartets=str(ROOT / "data/quartets.csv.gz"),
                include_reference=False,
                include_double=False,
                weights=None,
                batch_size=1,
            )
            with (
                patch.object(embed_variants, "sequence_table", return_value=table),
                patch.object(embed_variants, "Evo2Embedder", Embedder),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    embed_variants.run(args)
                meta = json.loads((Path(directory) / "meta.json").read_text())
                self.assertIn("code_sha256", meta)
                self.assertIn("source_sha256", meta)
                args.checkpoint = "different"
                with self.assertRaises(SystemExit):
                    embed_variants.run(args)
                args.checkpoint = "evo"
                failing[0] = False
                embed_variants.run(args)
                matrix = np.load(Path(directory) / "X_mean.npy")
                np.testing.assert_array_equal(matrix[:, 0], [65, 67, 71])
                embed_variants.run(args)
                np.testing.assert_array_equal(np.load(Path(directory) / "X_mean.npy"), matrix)


if __name__ == "__main__":
    unittest.main()
