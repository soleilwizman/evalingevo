"""Batch embedders shared by reference and alternate-sequence jobs."""

from model_runtime import (
    capture_layers,
    load_evo,
    load_hf,
    pool_real_bases,
    tokenize_ntv3,
)


class Evo2Embedder:
    def __init__(self, checkpoint, layer, weights=None, device="cuda:0"):
        self.model, self.device = load_evo(checkpoint, weights, device)
        self.layer = self.model.model.get_submodule(layer)
        self.representation = layer

    def __call__(self, sequences):
        import torch

        tokenizer = self.model.tokenizer
        tokens = [tokenizer.tokenize(s) for s in sequences]
        if not sequences or len({len(s) for s in sequences}) != 1:
            raise ValueError("Evo batches must contain equal-length sequences")
        if any(len(t) != len(s) for t, s in zip(tokens, sequences)):
            raise ValueError("Evo tokenizer must produce one token per nucleotide")
        ids = torch.tensor(
            [[tokenizer.eod_id] + t for t in tokens],
            dtype=torch.long,
            device=self.device,
        )
        with capture_layers([("hidden", self.layer)]) as captured:
            with torch.inference_mode():
                self.model.model(ids)
            hidden = captured["hidden"]
            if hidden.shape[:2] != ids.shape:
                raise ValueError("Evo embedding is not aligned to input tokens")
            return pool_real_bases(hidden, 1, len(sequences[0]), ids.shape[1])


class NTv3Embedder:
    def __init__(self, checkpoint, layer, revision="main", device="auto"):
        self.tokenizer, self.model, self.device = load_hf(checkpoint, revision, device)
        blocks = self.model.core.transformer_blocks
        if not 0 <= layer < len(blocks):
            raise ValueError(f"layer must be between 0 and {len(blocks) - 1}")
        self.layer = blocks[layer].final_layer_norm
        self.representation = f"core.transformer_blocks.{layer}.final_layer_norm"

    def __call__(self, sequences):
        import torch

        ids, left, length = tokenize_ntv3(self.tokenizer, sequences, self.device)
        with capture_layers([("hidden", self.layer)]) as captured:
            with torch.inference_mode():
                self.model(input_ids=ids)
            hidden = captured["hidden"]
            if hidden.shape[0] != len(sequences):
                raise ValueError("NTv3 hidden states are not batch-first")
            return pool_real_bases(hidden, left, len(sequences[0]), length)
