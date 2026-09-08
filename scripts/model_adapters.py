"""Per-sequence model representations with explicit positional alignment."""

import numpy as np
from model_runtime import (
    capture_layers,
    load_evo,
    load_hf,
    real_base_states,
    tokenize_ntv3,
)


class Evo2Adapter:
    units = "base"

    def __init__(
        self,
        checkpoint="evo2_7b_base",
        layer="blocks.26.mlp.l3",
        weights=None,
        device="cuda:0",
    ):
        self.model, self.device = load_evo(checkpoint, weights, device)
        self.layer = self.model.model.get_submodule(layer)
        self.label = f"Evo 2 {checkpoint} {layer}"

    def encode(self, sequence):
        import torch

        tokenizer = self.model.tokenizer
        tokens = tokenizer.tokenize(sequence)
        if len(tokens) != len(sequence):
            raise ValueError("Evo tokenizer must produce one token per base")
        ids = torch.tensor([[tokenizer.eod_id] + tokens], dtype=torch.long, device=self.device)
        with capture_layers([("hidden", self.layer)]) as captured:
            with torch.inference_mode():
                self.model.model(ids)
            h = captured["hidden"]
            if h.shape[:2] != ids.shape:
                raise ValueError("Evo hidden states do not align to input tokens")
            h = h[0, 1:].float()
        if not torch.isfinite(h).all():
            raise ValueError("non-finite Evo activations")
        return h, np.arange(len(sequence))


class NTv3Adapter:
    units = "base"

    def __init__(
        self,
        checkpoint="InstaDeepAI/NTv3_100M_pre",
        layer=-1,
        revision="main",
        device="auto",
    ):
        self.tok, self.model, self.device = load_hf(checkpoint, revision, device)
        self.layer = layer
        self.label = f"NTv3 {checkpoint.split('/')[-1]} hidden_states[{layer}]"

    def encode(self, sequence):
        import torch

        ids, left, padded_length = tokenize_ntv3(self.tok, [sequence], self.device)
        with torch.inference_mode():
            states = self.model.core(input_ids=ids, output_hidden_states=True)["hidden_states"]
        if not -len(states) <= self.layer < len(states):
            raise ValueError(f"hidden state index {self.layer} outside {len(states)} states")
        real = real_base_states(states[self.layer], left, len(sequence), padded_length)[0]
        return real, np.arange(len(sequence))


class DNABERT2Adapter:
    units = "token"

    def __init__(
        self,
        checkpoint="zhihan1996/DNABERT-2-117M",
        layer=-1,
        revision="main",
        device="auto",
    ):
        self.tok, self.model, self.device = load_hf(
            checkpoint, revision, device, family="dnabert2", masked_lm=False
        )
        encoder = self.model.encoder if hasattr(self.model, "encoder") else self.model.bert.encoder
        blocks = encoder.layer
        if not -len(blocks) <= layer < len(blocks):
            raise ValueError(f"block index {layer} outside {len(blocks)} blocks")
        index = layer % len(blocks)
        self.layer = blocks[index]
        self.label = f"DNABERT-2 block {index} of {len(blocks)}"

    def _base_to_token(self, sequence, token_ids):
        pieces = self.tok.convert_ids_to_tokens(token_ids)
        specials = set(self.tok.all_special_tokens)
        real = [
            (i, piece.replace("##", "")) for i, piece in enumerate(pieces) if piece not in specials
        ]
        if "".join(piece for _, piece in real) != sequence:
            raise ValueError("DNABERT-2 tokens do not reproduce the input sequence exactly")
        return np.concatenate([np.full(len(piece), i, dtype=int) for i, piece in real])

    def encode(self, sequence):
        import torch

        token_ids = self.tok(sequence, add_special_tokens=True)["input_ids"]
        mapping = self._base_to_token(sequence, token_ids)
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with capture_layers([("hidden", self.layer)]) as captured:
            with torch.inference_mode():
                self.model(input_ids=ids)
            h = captured["hidden"]
            if h.shape[:2] != ids.shape:
                raise ValueError("DNABERT-2 hidden states do not align to input tokens")
            real, inverse = np.unique(mapping, return_inverse=True)
            h = h[0, real.tolist()].float()
        if not torch.isfinite(h).all():
            raise ValueError("non-finite DNABERT-2 activations")
        return h, inverse
