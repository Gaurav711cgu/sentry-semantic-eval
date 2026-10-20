"""Reject cross-partition utterance-family and exact canonical-prefix leakage.

Family identities are declared provenance, not proof that independently named
sources are semantically independent. This guard cannot certify human labels,
pretraining independence, source secrecy or real customer distributions.
"""
from pair_v10 import pair_inputs


def validate_partitions(partitions):
    if type(partitions) is not dict or not partitions:
        raise ValueError('nonempty named partitions required')
    owners,groups,prefix_owners,group_reference={}, {}, {}, {}
    for split,cases in partitions.items():
        if type(split) is not str or not split or type(cases) is not list or not cases:
            raise ValueError('nonempty named case arrays required')
        groups[split]=set()
        for case in cases:
            group=case['group_id'];family=case['utterance_family_id']
            if any(type(x) is not str or not x for x in [group,family]):
                raise ValueError('group and utterance-family provenance required')
            if family in owners and owners[family]!=split:
                raise ValueError('utterance family crosses partitions: '+family)
            owners[family]=split
            key=(split,group)
            reference=(case['request'],case['expected_choice'],tuple(sorted((o['id'],o['status']) for o in case['orders'])))
            if key in group_reference and group_reference[key]!=reference:
                raise ValueError('group reference or request changes across views')
            group_reference[key]=reference;groups[split].add(group)
            for row in pair_inputs(case['request'],[o['id'] for o in case['orders']]):
                content=row['prefix_content']
                if content in prefix_owners and prefix_owners[content]!=split:
                    raise ValueError('canonical prefix crosses partitions')
                prefix_owners[content]=split
    return {'groups':{split:len(values) for split,values in groups.items()},
            'declared_utterance_families':len(owners),'canonical_prefixes':len(prefix_owners),
            'independent_validation_established':False}
