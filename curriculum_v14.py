"""Replay varied earlier training with shared style and explicit family provenance."""
import hashlib
import json

from data_v10 import TRAIN
from data_v11 import NEUTRAL_SUFFIXES, _views
from data_v13 import build_split as previous_split
from data_v13 import training_cases as previous_training
from pair_v10 import pair_inputs
from sampling_v14 import stratified_indices, validate_plan


def training_cases():
    cases = previous_training()
    t, o, u = 'order-red', 'order-blue', 'order-missing'
    for kind, templates in TRAIN.items():
        for index, template in enumerate(templates):
            target = u if kind == 'absent' else t if kind in ['direct', 'distractor'] else None
            for variant, suffix in enumerate(NEUTRAL_SUFFIXES):
                cases.extend(_views(f'train14-replay-{kind}-{index}-{variant}', kind,
                                    template.format(t=t, o=o, u=u) + suffix,
                                    target, [t, o, 'order-green']))
    # Capitalization augmentation is label-preserving across every family and class.
    # It changes training text only; inference normalization stays byte-identical.
    originals = list(cases)
    for case in originals:
        request = case['request']
        if request[0].islower():
            cases.append({**case, 'id': case['id'] + '-capital',
                          'group_id': case['group_id'] + '-capital',
                          'request': request[0].upper() + request[1:]})
    return cases


def training_curriculum():
    labels, provenance = {}, {}
    for case in training_cases():
        for row in pair_inputs(case['request'], [o['id'] for o in case['orders']]):
            content = row['prefix_content']
            label = int(row['candidate'] != case['expected_choice'])
            if content in labels and labels[content] != label:
                raise ValueError('contradictory replay supervision')
            labels[content] = label
            provenance.setdefault(content, set()).add(case['kind'])
    contents = sorted(labels)
    def family(content):
        kinds = provenance[content]
        return 'absent' if 'absent' in kinds else 'distractor' if 'distractor' in kinds else sorted(kinds)[0]
    return ([{'prefix_content': c, 'label': labels[c]} for c in contents],
            [family(c) for c in contents])


def training_pairs():
    return training_curriculum()[0]


def build_split(split):
    return training_cases() if split == 'train' else previous_split(split)


def build_manifest():
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    rows, families = training_curriculum()
    plan = stratified_indices(rows, families)
    return {'AI_authored': True, 'independent_final_qualified': False,
            'training_pairs': len(rows), 'positive_pairs': sum(r['label'] == 0 for r in rows),
            'training_sha256': sha(rows), 'families_sha256': sha(families),
            'sampling_plan_sha256': sha(plan),
            'development': 'Exposed v10-v13 development; used for error analysis',
            'curriculum': 'All v13 pairs plus original v10 raw training templates with shared neutral styles; class/family round-robin and within-family shuffled cycles',
            'dev_cases_sha256': sha(build_split('dev')),
            'final_cases_sha256': sha(build_split('final')),
            'final_source': 'Same unused v11 prospective final, no prior final model inference',
            'final_scenarios': 36, 'final_views': 216}
