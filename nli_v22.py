"""Task-adapted entailment encoder; canonical identities and raw labels stay fixed.

The hypothesis is fixed before inference. No request keyword, gold reference,
order status or controller result is used to construct or repair predictions.
"""
import json
import math
import re
from pathlib import Path

from pair_v10 import messages_for_pair
from worker_v10 import _contents, validate_training

HYPOTHESIS = 'The customer wants to cancel Order A now.'
MAX_TOKENS = 384
CHECKPOINTS = (32, 64, 128)


def nli_pair(content):
    messages_for_pair(content)
    value = json.loads(content)
    request = re.sub(r'OTHER_ORDER_(\d+)', lambda m: 'Order ' + chr(66 + int(m[1])), value['request'])
    request = request.replace('THIS_ORDER', 'Order A')
    return f'Customer message: {request}\nNumber of listed orders: {value["snapshot_count"]}.', HYPOTHESIS


def validate_payload(stage, value):
    if stage == 'train':
        if type(value) is not dict or set(value) != {'training_pairs', 'training_families', 'dev_contents', 'training_plan'}:
            raise ValueError('exact training-only payload required')
        rows, contents = validate_training({k: value[k] for k in ['training_pairs', 'dev_contents']})
        plan = value['training_plan']
        if type(plan) is not list or len(plan) != 128 or len(value['training_families']) != len(rows):
            raise ValueError('128 steps and aligned training families required')
        for step in plan:
            if type(step) is not list or len(step) != 4:
                raise ValueError('four accumulation microbatches required')
            for batch in step:
                if type(batch) is not list or len(batch) != 2 or any(type(i) is not int or not 0 <= i < len(rows) for i in batch):
                    raise ValueError('two bounded training indices required')
                if sorted(rows[i]['label'] for i in batch) != [0, 1]:
                    raise ValueError('balanced binary training labels required')
        if len(contents) != 195:
            raise ValueError('unchanged V19 development prefix coverage required')
    elif stage == 'score':
        if type(value) is not dict or set(value) != {'contents', 'adapter_file', 'parity_contents', 'expected_probabilities'}:
            raise ValueError('exact unlabeled score payload required')
        contents = _contents(value['contents'], 500)
        name = value['adapter_file']
        if type(name) is not str or Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('adapter basename required')
        parity = value['parity_contents']; expected = value['expected_probabilities']
        if type(parity) is not list or type(expected) is not list or len(parity) != len(expected) or len(parity) not in (0, 4):
            raise ValueError('zero or four aligned parity sentinels required')
        if parity:
            _contents(parity, 4)
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in expected):
            raise ValueError('bounded parity probabilities required')
    else:
        raise ValueError('known worker stage required')
    for content in contents:
        nli_pair(content)
    return value


def encode(tokenizer, content):
    premise, hypothesis = nli_pair(content)
    result = tokenizer(premise, hypothesis, truncation=False, padding=False)
    if not 1 <= len(result['input_ids']) <= MAX_TOKENS:
        raise ValueError('NLI pair exceeds384 tokens; truncation prohibited')
    return result


def adapt(model):
    import torch
    from torch import nn
    targets = [(n, m) for n, m in model.named_modules() if n.rsplit('.', 1)[-1] in {'query_proj', 'value_proj'}]
    if not targets or any(not isinstance(m, nn.Linear) for _, m in targets):
        raise ValueError('unadapted DeBERTa query/value linear projections required')
    for p in model.parameters():
        p.requires_grad_(False)

    class LoRA(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.empty(8, base.in_features, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, 8, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, value):
            return self.base(value) + (value @ self.lora_A.T @ self.lora_B.T) * 2

    for name, base in targets:
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent), leaf, LoRA(base))
    for module in [model.pooler, model.classifier]:
        for p in module.parameters():
            p.requires_grad_(True)
    return model


def binary_logits(logits):
    import torch
    if logits.ndim != 2 or logits.shape[1] != 3 or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise ValueError('finite FP32 contradiction/entailment/neutral logits required')
    # Preserve neutral versus contradiction freedom; the task labels only entailment.
    return torch.stack((logits[:, 1], torch.logsumexp(logits[:, [0, 2]], dim=1)), dim=1)


def state(model):
    import torch
    result = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    if not result or any(p.dtype != torch.float32 or p.device.type != 'cpu' or not torch.isfinite(p).all() for p in result.values()):
        raise ValueError('finite CPU FP32 trainable state required')
    return result


def restore(model, values):
    import torch
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if type(values) is not dict or set(values) != set(expected):
        raise ValueError('exact adapter and classifier keys required')
    for n, p in values.items():
        if not isinstance(p, torch.Tensor) or p.shape != expected[n].shape or p.dtype != torch.float32 or p.device.type != 'cpu' or not torch.isfinite(p).all():
            raise ValueError('adapter/classifier tensor identity mismatch')
    with torch.no_grad():
        for n, p in expected.items():
            p.copy_(values[n])
