"""Model loading, exact sequence alignment, and bounded activation capture."""

from contextlib import contextmanager

MULTIPLE = 128
POOLING_PROTOCOL = "real-base-overlap-v2"


def pick_device(requested="auto"):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf(checkpoint, revision=None, device="auto", family="ntv3", masked_lm=True):
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    if family == "ntv3" and "/" not in checkpoint:
        checkpoint = f"InstaDeepAI/{checkpoint}"
    kwargs = {"trust_remote_code": True}
    if revision and revision != "UNRECORDED":
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    loader = AutoModelForMaskedLM if masked_lm else AutoModel
    model, info = loader.from_pretrained(checkpoint, output_loading_info=True, **kwargs)
    # DNABERT's pooler is unused when extracting encoder states. Every used
    # weight, including a requested MLM head, must actually come from the checkpoint.
    missing = [
        key
        for key in info.get("missing_keys", [])
        if not (family == "dnabert2" and not masked_lm and key.startswith("pooler."))
    ]
    if missing or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(f"checkpoint did not load all required weights: {info}")
    device = pick_device(device)
    model = model.float().eval().to(device)
    if family == "ntv3" and not hasattr(model, "core"):
        raise ValueError("checkpoint does not expose the NTv3 core")
    return tokenizer, model, device


def load_evo(checkpoint, weights=None, device="cuda:0"):
    import torch
    from evo2 import Evo2

    device = pick_device(device)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Evo 2 requires a supported NVIDIA GPU")
    if device not in ("cuda", "cuda:0"):
        raise ValueError("Evo manages device placement; select GPUs with CUDA_VISIBLE_DEVICES")
    model = Evo2(checkpoint, local_path=weights)
    model.model.eval()
    return model, "cuda:0"


def pad_to_multiple(sequence, multiple=MULTIPLE):
    if not sequence or multiple < 1:
        raise ValueError("sequence and padding multiple must be nonempty/positive")
    target = -(-len(sequence) // multiple) * multiple
    left = (target - len(sequence)) // 2
    return "N" * left + sequence + "N" * (target - len(sequence) - left), left


def tokenize_ntv3(tokenizer, sequences, device):
    import torch

    if not sequences or len({len(s) for s in sequences}) != 1:
        raise ValueError("NTv3 batches must contain equal-length nonempty sequences")
    padded, lefts = zip(*(pad_to_multiple(s) for s in sequences))
    rows = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in padded]
    for row, sequence in zip(rows, padded):
        if len(row) != len(sequence) or "".join(tokenizer.convert_ids_to_tokens(row)) != sequence:
            raise ValueError("NTv3 tokens do not reproduce every padded input base exactly")
    return torch.tensor(rows, dtype=torch.long, device=device), lefts[0], len(padded[0])


def real_base_states(state, left, real_length, padded_length):
    """Align an NTv3 state to real bases using its exact integer downsample factor.

    Repetition defines positional coverage, not new information. Padding can
    influence activations through the model, but receives zero pooling weight.
    """
    import torch

    if state.ndim != 3 or state.shape[1] < 1 or padded_length % state.shape[1]:
        raise ValueError("hidden state must be batch-first with an exact integer resolution")
    if real_length < 1 or left < 0 or left + real_length > padded_length:
        raise ValueError("real sequence span is outside the tokenized input")
    factor = padded_length // state.shape[1]
    if factor < 1:
        raise ValueError("hidden state has more positions than input bases")
    positions = torch.arange(left, left + real_length, device=state.device) // factor
    real = state.index_select(1, positions).float()
    if not torch.isfinite(real).all():
        raise ValueError("non-finite activations in the real sequence")
    return real


def pool_real_bases(state, left, real_length, padded_length):
    real = real_base_states(state, left, real_length, padded_length)
    return real.mean(1).cpu().numpy(), real[:, -1].cpu().numpy()


def hidden_tensor(output, name="layer"):
    import torch

    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        tensor = output[0]
    else:
        raise ValueError(f"{name} returned an unsupported output; refusing to guess a tensor")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must return a batch-first rank-three hidden state")
    return tensor.detach()


@contextmanager
def capture_layers(layers):
    """Remove every hook on success, inference failure, or partial registration."""
    captured, handles = {}, []
    try:
        for name, module in layers:

            def capture(_module, _inputs, output, name=name):
                captured[name] = hidden_tensor(output, name)

            handles.append(module.register_forward_hook(capture))
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
