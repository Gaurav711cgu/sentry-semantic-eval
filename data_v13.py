"""Retain prior supervision while contrasting display context and conditions."""
import hashlib
import json

from data_v11 import ACTIONS, NEUTRAL_SUFFIXES, _views
from data_v12 import build_split as previous_split
from data_v12 import training_cases as previous_training
from pair_v10 import pair_inputs


def training_cases():
    cases = previous_training()
    t, o = 'order-red', 'order-blue'
    for index, action in enumerate((*ACTIONS, 'end {t}')):
        a = action.format(t=t)
        patterns = [
            ('direct', 'Please ' + a + '; the display must not substitute a different reference.', t),
            ('negated', 'Please do not ' + a + '; the display must not substitute a different reference.', None),
            ('direct', 'I request that you ' + a + ', regardless of whether the reference appears in the list.', t),
            ('negated', 'I do not request that you ' + a + ', regardless of whether the reference appears in the list.', None),
            ('direct', 'Please ' + a + '; do not replace the specified identifier with a displayed one.', t),
            ('direct', 'Please ' + a + '. A missing row does not revoke this instruction.', t),
            ('direct', 'Please ' + a + ', even though it is absent from the screen.', t),
            ('negated', 'I refuse authorization to ' + a + ', even though it is absent from the screen.', None),
            ('conditional', 'After receiving a second approval from me, ' + a + '; until then, take no action.', None),
            ('conditional', 'My confirmation is still pending; do not ' + a + ' yet.', None),
            ('conditional', 'When my later confirmation arrives, you can ' + a + '.', None),
            ('conditional', 'Before you ' + a + ', ask me for fresh consent and wait.', None),
            ('conditional', 'Once I send the follow-up approval, you may ' + a + '.', None),
            ('direct', 'This is my confirmation: ' + a + ' now.', t),
            ('direct', 'I confirm now that you should ' + a + '.', t),
            ('direct', 'I authorize you to ' + a + ' without waiting for another message from me.', t),
            ('negated', 'I do not authorize you to ' + a + '.', None),
            ('distractor', 'Keep ' + o + ' unchanged; please ' + a + ' today.', t),
        ]
        for number, (kind, request, target) in enumerate(patterns):
            for variant, suffix in enumerate(NEUTRAL_SUFFIXES):
                cases.extend(_views(f'train13-{index}-{number}-{variant}', kind,
                                    request + suffix, target, [t, o, 'order-green']))
    return cases


def training_pairs():
    labels = {}
    for case in training_cases():
        for row in pair_inputs(case['request'], [o['id'] for o in case['orders']]):
            content = row['prefix_content']
            label = int(row['candidate'] != case['expected_choice'])
            if content in labels and labels[content] != label:
                raise ValueError('contradictory curriculum label')
            labels[content] = label
    return [{'prefix_content': content, 'label': label} for content, label in sorted(labels.items())]


def build_split(split):
    return training_cases() if split == 'train' else previous_split(split)


def build_manifest():
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pairs = training_pairs()
    return {
        'AI_authored': True, 'independent_final_qualified': False,
        'training_pairs': len(pairs), 'positive_pairs': sum(r['label'] == 0 for r in pairs),
        'training_sha256': sha(pairs),
        'development': 'Exposed v10-v12 development, intentionally used for error analysis',
        'curriculum': 'Retain all v12 pairs; bare end, unrelated display commentary and pending/current confirmation contrasts across all action verbs',
        'dev_cases_sha256': sha(build_split('dev')),
        'final_cases_sha256': sha(build_split('final')),
        'final_source': 'Same unused v11 prospective final; no prior final model inference',
        'final_views': 216, 'final_scenarios': 36,
    }
