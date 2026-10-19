"""CPU FP32 q/v LoRA and an explicit prefix-only binary next-token objective.

Torch is imported only when functions are called. No base model, tokenizer,
network, training loop or dataset is loaded by this module.
"""

import math


def install_lora(model, rank=4, alpha=8, dropout=0.05):
    """Freeze every base parameter, then adapt every q_proj/v_proj linear layer.

    Return the same model. SmolLM2-135M has30layers,576wide q and192wide v;
    rank4 therefore adds30*(4*(576+576)+4*(576+192))=230400parameters.
    """
    import torch
    from torch import nn

    if (
        type(rank) is not int
        or rank < 1
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 0
        or type(dropout) not in (int, float)
        or not math.isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise ValueError("finite positive rank/alpha and bounded dropout required")
    if any(getattr(module, "_sentry_lora_v10", False) for module in model.modules()):
        raise ValueError("LoRA already installed")
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    ]
    if not targets or any(not isinstance(module, nn.Linear) for _, module in targets):
        raise ValueError("q_proj/v_proj linear targets required")
    model.to(device="cpu", dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class ProjectionLora(nn.Module):
        def __init__(self, base):
            super().__init__()
            self._sentry_lora_v10 = True
            self.base = base
            self.lora_A = nn.Parameter(
                torch.empty(rank, base.in_features, dtype=torch.float32)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(base.out_features, rank, dtype=torch.float32)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.dropout = nn.Dropout(float(dropout))
            self.scaling = float(alpha) / rank

        def forward(self, x):
            return (
                self.base(x)
                + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            )

    for name, base in targets:
        path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(path) if path else model
        setattr(parent, attribute, ProjectionLora(base).train(base.training))
    return model


def pair_logits(model, input_ids, attention_mask, label_token_ids):
    """Return A/B logits predicting the token AFTER each right-padded prefix.

    Inputs contain prefixes only: no answer token is appended and no shift is
    applied. Only the last real hidden state reaches lm_head, avoiding a full
    batch-by-sequence-by-vocabulary activation tensor.
    """
    import torch

    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.dtype != torch.long
        or input_ids.ndim != 2
        or input_ids.shape[0] < 1
        or input_ids.shape[1] < 1
        or input_ids.device.type != "cpu"
        or torch.any(input_ids < 0)
        or not isinstance(attention_mask, torch.Tensor)
        or attention_mask.shape != input_ids.shape
        or attention_mask.device.type != "cpu"
        or not torch.all((attention_mask == 0) | (attention_mask == 1))
    ):
        raise ValueError("aligned nonempty CPU prefix and binary mask required")
    if (
        type(label_token_ids) not in (list, tuple)
        or len(label_token_ids) != 2
        or any(type(token) is not int or token < 0 for token in label_token_ids)
        or label_token_ids[0] == label_token_ids[1]
    ):
        raise ValueError("two distinct single-token label IDs required")
    lengths = attention_mask.long().sum(dim=1)
    if torch.any(lengths < 1) or torch.any(
        attention_mask[:, 1:] > attention_mask[:, :-1]
    ):
        raise ValueError("nonempty right-padded prefixes required")
    output = model.model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = output.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
        raise ValueError("aligned last hidden states required")
    selected = hidden[torch.arange(input_ids.shape[0]), lengths - 1]
    logits = model.lm_head(selected)
    if (
        logits.ndim != 2
        or logits.shape[0] != input_ids.shape[0]
        or max(label_token_ids) >= logits.shape[1]
    ):
        raise ValueError("label token outside vocabulary")
    result = logits[:, label_token_ids]
    if result.dtype != torch.float32 or not torch.isfinite(result).all():
        raise ValueError("finite FP32 pair logits required")
    return result


def binary_loss(logits, labels):
    """Direct class0/1 cross entropy for already gathered next-token pair logits."""
    import torch

    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[1] != 2
        or logits.shape[0] < 1
        or not torch.isfinite(logits).all()
        or logits.dtype != torch.float32
        or logits.device.type != "cpu"
        or not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.long
        or labels.shape != (logits.shape[0],)
        or labels.device.type != "cpu"
        or not torch.all((labels == 0) | (labels == 1))
    ):
        raise ValueError(
            "finite CPU FP32 pair logits and aligned binary class labels required"
        )
    return torch.nn.functional.cross_entropy(logits, labels)


def _adapter_parameters(model):
    result = {}
    for name, module in model.named_modules():
        if getattr(module, "_sentry_lora_v10", False):
            for leaf in ("lora_A", "lora_B"):
                result[name + "." + leaf] = getattr(module, leaf)
    if not result:
        raise ValueError("installed genuine LoRA adapters required")
    return result


def adapter_state(model):
    """Return detached copies of A/B tensors only; no frozen base state."""
    import torch

    result = {}
    for name, value in _adapter_parameters(model).items():
        if (
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finite CPU FP32 adapter state required")
        result[name] = value.detach().clone()
    return result


def load_adapter(model, state):
    """Validate every tensor before copying any; never load missing/base keys."""
    import torch

    expected = _adapter_parameters(model)
    if type(state) is not dict or set(state) != set(expected):
        raise ValueError("exact LoRA A/B state keys required")
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[name].shape
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError("adapter tensor shape/dtype/finiteness mismatch: " + name)
    with torch.no_grad():
        for name, value in state.items():
            expected[name].copy_(value)
    return model
