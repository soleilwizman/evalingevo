import types
import unittest
from unittest.mock import patch

import model_adapters
import model_embeddings
import numpy as np
import torch
from model_runtime import load_evo
from ntv3_score import NTv3Scorer


class CharacterTokenizer:
    alphabet = "ACGTN?"
    mask_token_id = 5
    eod_id = 99

    def __call__(self, sequence, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("unexpected special tokens")
        return {"input_ids": self.tokenize(sequence)}

    def tokenize(self, sequence):
        return [self.alphabet.index(base) for base in sequence]

    def convert_ids_to_tokens(self, ids):
        return [self.alphabet[index] for index in ids]


class ModelPathTests(unittest.TestCase):
    def test_dnabert_pooling_keeps_native_tokens_and_excludes_specials(self):
        class Tokenizer:
            all_special_tokens = ["[CLS]", "[SEP]"]

            def __call__(self, sequence, add_special_tokens):
                return {"input_ids": [0, 1, 2, 3]}

            def convert_ids_to_tokens(self, ids):
                return [["[CLS]", "A", "CGT", "[SEP]"][i] for i in ids]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = types.SimpleNamespace(layer=[torch.nn.Identity()])

            def forward(self, input_ids):
                return self.encoder.layer[0](input_ids.float().unsqueeze(-1))

        model = Model()
        with patch.object(model_adapters, "load_hf", return_value=(Tokenizer(), model, "cpu")):
            adapter = model_adapters.DNABERT2Adapter(layer=0)
            hidden, mapping = adapter.encode("ACGT")
            np.testing.assert_array_equal(hidden[:, 0], [1, 2])
            np.testing.assert_array_equal(mapping, [0, 1, 1, 1])
            self.assertEqual(float(hidden.mean()), 1.5)
            with self.assertRaisesRegex(ValueError, "reproduce"):
                adapter.encode("TGCA")
        self.assertFalse(model.encoder.layer[0]._forward_hooks)

    def test_evo_batch_pooling_matches_single_adapter_and_removes_bos(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()

            def forward(self, ids):
                return self.layer(ids.float().unsqueeze(-1))

        model = Model()
        evo = types.SimpleNamespace(model=model, tokenizer=CharacterTokenizer())
        with (
            patch.object(model_embeddings, "load_evo", return_value=(evo, "cpu")),
            patch.object(model_adapters, "load_evo", return_value=(evo, "cpu")),
        ):
            embedder = model_embeddings.Evo2Embedder("evo", "layer")
            adapter = model_adapters.Evo2Adapter("evo", "layer")
            mean, last = embedder(["ACGT", "TGCA"])
            np.testing.assert_array_equal(mean[:, 0], [1.5, 1.5])
            np.testing.assert_array_equal(last[:, 0], [3, 0])
            for row, sequence in enumerate(["ACGT", "TGCA"]):
                hidden, indices = adapter.encode(sequence)
                np.testing.assert_array_equal(indices, np.arange(4))
                np.testing.assert_array_equal(mean[row], hidden.mean(0).numpy())
        self.assertFalse(model.layer._forward_hooks)

    def test_ntv3_batch_and_single_adapter_use_real_base_overlap(self):
        layer = torch.nn.Identity()

        class Model:
            core = types.SimpleNamespace(
                transformer_blocks=[types.SimpleNamespace(final_layer_norm=layer)]
            )

            def __call__(self, input_ids):
                n, length = input_ids.shape
                state = torch.arange(length // 2).float().reshape(1, -1, 1).repeat(n, 1, 1)
                layer(state)

        tokenizer, model = CharacterTokenizer(), Model()
        with patch.object(model_embeddings, "load_hf", return_value=(tokenizer, model, "cpu")):
            embedder = model_embeddings.NTv3Embedder("ntv3", 0)
            mean, last = embedder(["A" * 199, "C" * 199])
        expected = torch.arange(128).repeat_interleave(2)[28:227].float()
        np.testing.assert_allclose(mean[:, 0], expected.mean().repeat(2).numpy())
        np.testing.assert_array_equal(last[:, 0], expected[-1].repeat(2).numpy())
        self.assertFalse(layer._forward_hooks)

        class Core:
            def __call__(self, input_ids, output_hidden_states):
                return {
                    "hidden_states": [torch.arange(input_ids.shape[1]).reshape(1, -1, 1).float()]
                }

        model.core = Core()
        with patch.object(model_adapters, "load_hf", return_value=(tokenizer, model, "cpu")):
            adapter = model_adapters.NTv3Adapter("ntv3")
            state, indices = adapter.encode("A" * 199)
        np.testing.assert_array_equal(state[:, 0], np.arange(28, 227))
        np.testing.assert_array_equal(indices, np.arange(199))

    def test_pseudolikelihood_masks_and_scores_only_real_bases(self):
        masks = []

        class Model:
            def __call__(self, input_ids):
                positions = (input_ids == 5).nonzero()
                masks.extend(positions[:, 1].tolist())
                self_test.assertEqual(len(positions), len(input_ids))
                return types.SimpleNamespace(logits=torch.zeros(*input_ids.shape, 6))

        self_test = self
        with patch("model_runtime.load_hf", return_value=(CharacterTokenizer(), Model(), "cpu")):
            scorer = NTv3Scorer(device="cpu")
            scorer.POSITION_CHUNK = 3
            score = scorer.forward(["ACGTACG"])[0]
        self.assertEqual(masks, list(range(60, 67)))
        self.assertAlmostEqual(score, -7 * np.log(6), places=5)

    def test_evo_does_not_claim_to_move_an_auto_sharded_model(self):
        fake = types.SimpleNamespace(
            Evo2=lambda *args, **kwargs: self.fail("must reject before loading")
        )
        with (
            patch.dict("sys.modules", {"evo2": fake}),
            patch("torch.cuda.is_available", return_value=True),
        ):
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                load_evo("evo", device="cuda:1")


if __name__ == "__main__":
    unittest.main()
