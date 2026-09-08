import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import validation
from artifact_io import aligned_tables, atomic_json, load_matrix, matched_embedding_metadata
from benchmark_stats import element_kmers
from evo_probe import kmers
from model_runtime import capture_layers, pool_real_bases, real_base_states, tokenize_ntv3
from variant_data import within_element_accuracy


class PoolingTests(unittest.TestCase):
    def test_all_ntv3_resolutions_match_real_base_reference(self):
        for positions in (2, 4, 8, 16, 32, 64, 128, 256):
            for length in (1, 127, 199, 200, 201, 255, 256):
                with self.subTest(positions=positions, length=length):
                    h = torch.arange(positions * 3, dtype=torch.float32).reshape(1, positions, 3)
                    left = (256 - length) // 2
                    repeated = h.repeat_interleave(256 // positions, dim=1)[:, left : left + length]
                    mean, last = pool_real_bases(h, left, length, 256)
                    np.testing.assert_allclose(mean, repeated.mean(1).numpy(), rtol=1e-6)
                    np.testing.assert_array_equal(last, repeated[:, -1].numpy())

    def test_padding_and_special_positions_have_zero_weight(self):
        h = torch.ones(2, 256, 4)
        h[:, :28] = float("nan")
        h[:, 228:] = 1e20
        mean, last = pool_real_bases(h, 28, 200, 256)
        np.testing.assert_array_equal(mean, np.ones((2, 4)))
        np.testing.assert_array_equal(last, np.ones((2, 4)))
        h[:, 100] = float("nan")
        with self.assertRaises(ValueError):
            pool_real_bases(h, 28, 200, 256)

    def test_invalid_resolution_does_not_round_or_fallback(self):
        with self.assertRaises(ValueError):
            real_base_states(torch.ones(1, 3, 2), 28, 200, 256)
        with self.assertRaises(ValueError):
            real_base_states(torch.ones(1, 256, 2), 100, 200, 256)

    def test_symmetric_bottleneck_preserves_existing_200mer_mean(self):
        h = torch.tensor([[[2.0, 4.0], [8.0, 10.0]]])
        mean, _ = pool_real_bases(h, 28, 200, 256)
        np.testing.assert_array_equal(mean, h.mean(1).numpy())

    def test_every_tokenized_row_is_checked(self):
        class Tokenizer:
            def __call__(self, sequence, add_special_tokens):
                self_test.assertFalse(add_special_tokens)
                return {"input_ids": list(sequence)}

            def convert_ids_to_tokens(self, ids):
                return ["A" if base == "C" else base for base in ids]

        self_test = self
        with self.assertRaisesRegex(ValueError, "every padded input"):
            tokenize_ntv3(Tokenizer(), ["AAAA", "CCCC"], "cpu")

    def test_hooks_removed_after_failure(self):
        layer = torch.nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with capture_layers([("h", layer)]) as captured:
                layer(torch.ones(1, 4, 2))
                self.assertEqual(tuple(captured["h"].shape), (1, 4, 2))
                raise RuntimeError("inference failed")
        self.assertFalse(layer._forward_hooks)

    def test_partial_hook_registration_is_cleaned(self):
        layer = torch.nn.Identity()

        class BadLayer:
            def register_forward_hook(self, hook):
                raise RuntimeError("registration failed")

        with self.assertRaisesRegex(RuntimeError, "registration"):
            with capture_layers([("h", layer), ("bad", BadLayer())]):
                self.fail("registration should fail first")
        self.assertFalse(layer._forward_hooks)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.groups = np.repeat(np.arange(12), 2)
        self.X = rng.normal(size=(24, 3))
        self.y = 2 * self.X[:, 0] + rng.normal(size=24)

    def test_group_split_is_seeded_complete_and_disjoint(self):
        splits = validation.group_splits(self.groups, 3, 19)
        self.assertEqual(sorted(np.concatenate([test for _, test in splits])), list(range(24)))
        for (train, test), (again_train, again_test) in zip(
            splits, validation.group_splits(self.groups, 3, 19)
        ):
            self.assertFalse(set(self.groups[train]) & set(self.groups[test]))
            np.testing.assert_array_equal(train, again_train)
            np.testing.assert_array_equal(test, again_test)

    def test_inner_cv_groups_and_scaler_fit_scope(self):
        model = validation.fit_ridge(self.X, self.y, self.groups, folds=3)
        for train, test in model.cv:
            self.assertFalse(set(self.groups[train]) & set(self.groups[test]))
        self.assertEqual(list(model.estimator.named_steps), ["standardscaler", "ridge"])
        np.testing.assert_allclose(model.best_estimator_[0].mean_, self.X.mean(0))

    def test_outer_predictions_ignore_heldout_labels(self):
        _, test = validation.group_splits(self.groups, 3, 0)[0]
        changed = self.y.copy()
        changed[test] += 1e6
        before = validation.out_of_fold(self.X, self.y, self.groups, 3)
        after = validation.out_of_fold(self.X, changed, self.groups, 3)
        np.testing.assert_array_equal(before[test], after[test])

    def test_selection_ignores_heldout_labels(self):
        _, test = validation.group_splits(self.groups, 3, 0)[0]
        changed = self.y.copy()
        changed[test] *= -1e6
        candidates = {"a": self.X[:, :1], "b": self.X[:, 1:]}
        before, choices = validation.selected_predictions(candidates, self.y, self.groups, 3)
        after, changed_choices = validation.selected_predictions(
            candidates, changed, self.groups, 3
        )
        self.assertEqual(choices[0], changed_choices[0])
        np.testing.assert_array_equal(before[test], after[test])

    def test_residual_training_does_not_depend_on_outer_test_labels(self):
        _, test = validation.group_splits(self.groups, 3, 0)[0]
        changed = self.y.copy()
        changed[test] += 500
        before, target = validation.residual_predictions(
            self.X[:, 1:], self.X[:, :1], self.y, self.groups, 3
        )
        after, changed_target = validation.residual_predictions(
            self.X[:, 1:], self.X[:, :1], changed, self.groups, 3
        )
        np.testing.assert_array_equal(before[test], after[test])
        np.testing.assert_allclose(changed_target[test] - target[test], 500)

    def test_invalid_data_never_silently_changes_protocol(self):
        with self.assertRaises(ValueError):
            validation.group_splits(["a", "a"], 2)
        with self.assertRaises(ValueError):
            validation.out_of_fold([[np.nan], [1]], [0, 1], [0, 1], 2)
        with self.assertRaises(ValueError):
            validation.out_of_fold(self.X, self.y[:-1], self.groups, 3)

    def test_mean_baseline_uses_only_training_labels(self):
        means = validation.mean_prediction(self.y, self.groups, 3)
        for train, test in validation.group_splits(self.groups, 3):
            np.testing.assert_array_equal(means[test], np.repeat(self.y[train].mean(), len(test)))

    def test_permutation_scores_the_permuted_target(self):
        with patch.object(validation, "out_of_fold", side_effect=lambda X, y, *args: y.copy()):
            null = validation.permutation_null(self.X, self.y, self.groups, 3, draws=3)
        np.testing.assert_allclose(null, np.ones(3))


class ArtifactTests(unittest.TestCase):
    def test_paired_metadata_requires_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            variant, reference = Path(directory) / "variant", Path(directory) / "reference"
            common = {
                "checkpoint": "model",
                "pooling_protocol": "real-base-overlap-v2",
                "layer": "layer",
            }
            atomic_json(variant / "meta.json", common)
            atomic_json(reference / "meta.json", common)
            with self.assertRaisesRegex(ValueError, "revision"):
                matched_embedding_metadata(variant, reference)
            common["weights_sha256"] = "identical-local-weights"
            atomic_json(variant / "meta.json", common)
            atomic_json(reference / "meta.json", common)
            matched_embedding_metadata(variant, reference)
            atomic_json(reference / "meta.json", {**common, "weights_sha256": "different"})
            with self.assertRaisesRegex(ValueError, "hashes"):
                matched_embedding_metadata(variant, reference)

    def test_shared_kmers_preserve_overlap_and_column_order(self):
        from itertools import product

        sequences = ["AAAACGTN", "N", "", "ACGTACGT"]
        for maximum in (0, 1, 2, 3, 4):
            words = [
                "".join(word) for k in range(1, maximum + 1) for word in product("ACGT", repeat=k)
            ]
            expected = np.array(
                [
                    [
                        sum(
                            sequence[i : i + len(word)] == word
                            for i in range(len(sequence) - len(word) + 1)
                        )
                        for word in words
                    ]
                    for sequence in sequences
                ],
                dtype=float,
            )
            np.testing.assert_array_equal(kmers(sequences, maximum), expected)
            np.testing.assert_array_equal(element_kmers(sequences, maximum), expected)

    def test_only_measurements_allow_csv_round_trip_noise(self):
        table = pd.DataFrame({"id": [0.0, 1.0], "activity": [1.0, 2.0]})
        changed = table.copy()
        changed.loc[0, "activity"] += 2e-16
        aligned_tables(table, changed, ["id", "activity"], numeric_columns=("activity",))
        changed.loc[0, "id"] += 2e-16
        with self.assertRaises(ValueError):
            aligned_tables(table, changed, ["id", "activity"], numeric_columns=("activity",))
        changed = table.copy()
        changed.loc[0, "activity"] += 1e-6
        with self.assertRaises(ValueError):
            aligned_tables(table, changed, ["id", "activity"], numeric_columns=("activity",))

    def test_equal_row_counts_do_not_allow_misaligned_targets(self):
        table = pd.DataFrame({"id": ["a", "b"], "y": [1, 2], "group": [0, 1]})
        with self.assertRaises(ValueError):
            aligned_tables(table, table.iloc[::-1], ["id", "y", "group"])
        changed = table.copy()
        changed.loc[1, "y"] = 3
        with self.assertRaises(ValueError):
            aligned_tables(table, changed, ["id", "y", "group"])

    def test_matrix_rejects_corruption_duplicates_and_wrong_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "X.npy"
            table = pd.DataFrame({"id": ["a", "b"]})
            for matrix in (np.ones((3, 2)), np.array([[1.0], [np.nan]]), np.ones(2)):
                np.save(path, matrix)
                with self.assertRaises(ValueError):
                    load_matrix(path, table, ["id"])
            np.save(path, np.ones((2, 2)))
            with self.assertRaises(ValueError):
                load_matrix(path, pd.DataFrame({"id": ["a", "a"]}), ["id"])

    def test_atomic_json_failure_preserves_previous_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.json"
            atomic_json(path, {"done": 10})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                atomic_json(path, {"done": float("nan")})
            self.assertEqual(before, path.read_bytes())

    def test_tied_variant_predictions_receive_half_credit(self):
        self.assertEqual(within_element_accuracy([1, 1, 1], [1, 2, 3], [0, 0, 0]), (0.5, 3))


if __name__ == "__main__":
    unittest.main()
