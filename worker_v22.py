"""Bounded offline NLI adaptation. No final references enter this process."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from nli_v22 import CHECKPOINTS, adapt, binary_logits, encode, restore, state, validate_payload
from worker_v10 import peak_rss, new_adapter_directory


def run(args, value, output):
    validate_payload(args.stage, value)
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(1019)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32)
    if model.config.label2id != {'contradiction': 0, 'entailment': 1, 'neutral': 2}:
        raise ValueError('frozen NLI class mapping mismatch')
    adapt(model)
    metadata = {'python': sys.version.split()[0], 'torch': torch.__version__,
                'dtype': 'float32', 'device': 'cpu', 'threads': 4, 'seed': 1019,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'machine': platform.machine(), 'processor': platform.processor()}
    for package in ['transformers', 'numpy', 'safetensors', 'tokenizers', 'huggingface-hub']:
        metadata[package] = importlib.metadata.version(package)
    output['metadata'] = metadata
    if metadata['trainable_parameters'] != 740355:
        raise ValueError('NLI adapter plus pooler/classifier parameter count mismatch')
    destination = Path(args.output)

    def persist():
        output['rss_peak_bytes'] = peak_rss()
        output['wall_seconds'] = time.monotonic() - started
        temporary = destination.with_suffix('.pending')
        temporary.write_text(json.dumps(output, allow_nan=False))
        temporary.replace(destination)

    def guard():
        if peak_rss() > 8 * 1024**3 or time.monotonic() - started > 3000:
            raise ValueError('NLI worker memory/wall limit exceeded')

    def score(contents, records):
        model.eval()
        with torch.inference_mode():
            for content in contents:
                guard()
                begun = time.monotonic()
                inputs = tokenizer.pad([encode(tokenizer, content)], padding=True, return_tensors='pt')
                logits = binary_logits(model(**inputs).logits)
                probability = float(torch.softmax(logits, dim=1)[0, 0])
                elapsed = time.monotonic() - begun
                records.append({'prefix_content': content, 'positive_probability': probability, 'duration_seconds': elapsed})
                output['scoring_requests'] += 1
                if elapsed > 10:
                    persist()
                    raise ValueError('individual scoring request exceeded10 seconds')
                if len(records) % 16 == 0:
                    persist()
                guard()

    try:
        if args.stage == 'train':
            directory = new_adapter_directory(args.adapter_dir)
            rows = value['training_pairs']
            encoded = [encode(tokenizer, r['prefix_content']) for r in rows]
            output['checkpoints'] = {}

            def checkpoint(step):
                name = 'base' if step == 0 else f'step{step}'
                path = directory / (name + '.safetensors')
                save_file(state(model), str(path))
                item = {'step': step, 'adapter_file': path.name,
                        'adapter_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'records': []}
                output['checkpoints'][name] = item
                persist()
                score(value['dev_contents'], item['records'])
                persist()
                print('NLI_CHECKPOINT ' + json.dumps({'step': step, 'scoring_requests': output['scoring_requests'], 'wall_seconds': output['wall_seconds']}), flush=True)

            checkpoint(0)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001, weight_decay=.01)
            output['distinct_presented_pairs'] = len({i for step in value['training_plan'] for batch in step for i in batch})
            training_started = time.monotonic()
            for step, batches in enumerate(value['training_plan'], 1):
                guard()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for indices in batches:
                    features = tokenizer.pad([encoded[i] for i in indices], padding=True, return_tensors='pt')
                    labels = torch.tensor([rows[i]['label'] for i in indices], dtype=torch.long)
                    loss = torch.nn.functional.cross_entropy(binary_logits(model(**features).logits), labels) / 4
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite NLI training loss')
                    loss.backward(); loss_sum += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                output['completed_optimizer_steps'] = step
                output.setdefault('losses', []).append({'step': step, 'loss': loss_sum, 'gradient_norm': float(norm)})
                persist()
                if step == 16 and time.monotonic() - training_started > 600:
                    raise ValueError('first16-step training admission exceeded600 seconds')
                if step in CHECKPOINTS:
                    checkpoint(step)
        else:
            path = Path(args.adapter_dir) / value['adapter_file']
            if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 4_000_000:
                raise ValueError('bounded retained adapter/classifier required')
            restore(model, load_file(str(path), device='cpu'))
            output['adapter_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            output['parity'] = {'records': [], 'expected_probabilities': value['expected_probabilities'], 'passed': False}
            score(value['parity_contents'], output['parity']['records'])
            errors = [abs(r['positive_probability'] - p) for r, p in zip(output['parity']['records'], value['expected_probabilities'])]
            output['parity']['max_absolute_error'] = max(errors, default=0.0)
            output['parity']['passed'] = bool(errors) and max(errors) <= 1e-6
            persist()
            if errors and not output['parity']['passed']:
                raise ValueError('fresh NLI process parity failed at1e-6 tolerance')
            output['records'] = []
            score(value['contents'], output['records'])
        guard()
        output['passed'] = True
    finally:
        persist()


def main():
    parser = argparse.ArgumentParser()
    for name in ['stage', 'model', 'input', 'output', 'adapter-dir']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = {'passed': False, 'stage': args.stage, 'completed_optimizer_steps': 0, 'scoring_requests': 0}
    try:
        path = Path(args.input)
        if path.is_symlink() or path.stat().st_size > 8 * 1024**2:
            raise ValueError('bounded regular worker input required')
        from run_v19 import checked_json
        run(args, checked_json(path.read_bytes()), output)
    except BaseException as error:
        output.update(passed=False, failure_type=type(error).__name__, failure=str(error)[:2000])
    Path(args.output).write_text(json.dumps(output, allow_nan=False))
    return 0 if output['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
