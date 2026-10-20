"""Freeze model-free probes for a prospective training-fit versus transfer study.

This module does not load weights, run inference or authorize a hosted job.
"""
import hashlib
import json
from collections import defaultdict

from data_v17 import training_curriculum, build_split
from pair_v10 import pair_inputs
from sampling_v14 import stratified_indices


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def select_probes(rows, families, plan, per_stratum=4):
    """Sample fit probes by label, semantic family and exposure at step32.

    These are training-corpus diagnostics, never held-out generalization data.
    """
    if type(per_stratum) is not int or not 1 <= per_stratum <= 8:
        raise ValueError('bounded probe count required')
    if len(rows) != len(families) or len(plan) < 32:
        raise ValueError('aligned corpus and at least32 planned steps required')
    seen = {i for step in plan[:32] for batch in step for i in batch}
    pools = defaultdict(list)
    for i, (row, family) in enumerate(zip(rows, families)):
        pools[(row['label'], family, i in seen)].append(i)
    probes = []
    for (label, family, presented), indices in sorted(pools.items()):
        # Hash order avoids hand-picking easy or known failing training requests.
        indices.sort(key=lambda i: digest(rows[i]['prefix_content']))
        for i in indices[:per_stratum]:
            probes.append({'training_index': i, 'label': label,
                'family': family, 'presented_in_v17_first32': presented,
                'prefix_content': rows[i]['prefix_content']})
    return probes


def build_study():
    rows, families = training_curriculum()
    plan = stratified_indices(rows, families)
    probes = select_probes(rows, families, plan)
    development = build_split('dev')
    dev_contents = sorted({r['prefix_content'] for c in development
        for r in pair_inputs(c['request'], [o['id'] for o in c['orders']])})
    if {p['prefix_content'] for p in probes} & set(dev_contents):
        raise ValueError('training-fit probe overlaps development')
    return {'study': 'training-fit-and-transfer-v18-preparation',
        'status': 'prepared_not_executed_not_a_run_grant',
        'AI_authored_references': True, 'independent_final_qualified': False,
        'training_corpus_sha256': digest(rows), 'training_families_sha256': digest(families),
        'sampling_plan_sha256': digest(plan), 'fit_probes': probes,
        'fit_probe_count': len(probes), 'development_prefix_count': len(dev_contents),
        'development_sha256': digest(development),
        'scoring_inputs_sha256': digest(sorted(set(dev_contents) | {p['prefix_content'] for p in probes})),
        'final_inputs_included': False,
        'exposure_limit': 'Not presented in first32 means this V17 schedule only; earlier warm-start exposure is unknown.',
        'next_execution': {
            'first_stage': 'Score retained initial, step32 and step128 adapters on frozen training-fit probes only; reuse retained development outputs.',
            'maximum_physical_scoring_requests': 3 * (len(probes) + 4),
            'training_steps': 0, 'automatic_promotion': False,
            'threshold': 0.8, 'threshold_tuning_allowed': False,
            'seed': 1019, 'threads': 4, 'rss_limit_bytes': 8 * 1024**3,
            'request_deadline_seconds': 10, 'worker_deadline_seconds': 1200,
            'requirements_before_dispatch': ['Exact source, adapter and protocol hashes',
                'One consumed run-specific grant on a free standard public CPU runner',
                'Worker isolation, retained partial errors and verified cleanup'],
            'decision_rule': 'Compare per-family training-fit errors with retained development errors. Do not infer full-corpus fit from this stratified sample. No final inference or automatic extra training.'}}


if __name__ == '__main__':
    from pathlib import Path
    value = build_study()
    target = Path(__file__).resolve().parent.parent / 'sentry-trained-v17' / 'STUDY_V18_PREPARATION.json'
    target.write_text(json.dumps(value, indent=2) + '\n')
    print(target)
    print('fit_probes', value['fit_probe_count'], 'maximum_new_scores', value['next_execution']['maximum_physical_scoring_requests'])
