import unittest

from data_v11 import ACTIONS
from data_v13 import build_split, training_pairs
from pair_v10 import canonical_pair, pair_inputs


class ContextAndConditionTests(unittest.TestCase):
    def test_missing_screen_context_does_not_change_authorization(self):
        rows = {r['prefix_content']: r['label'] for r in training_pairs()}
        ids = ['order-red', 'order-blue', 'order-green']
        for action in (*ACTIONS, 'end {t}'):
            a = action.format(t='order-red')
            for text, label in [
                ('Please ' + a + '; the display must not substitute a different reference.', 0),
                ('Please do not ' + a + '; the display must not substitute a different reference.', 1),
                ('I request that you ' + a + ', regardless of whether the reference appears in the list.', 0),
                ('I do not request that you ' + a + ', regardless of whether the reference appears in the list.', 1),
            ]:
                key = canonical_pair(text, ids, 'order-red')
                self.assertTrue(key in rows, text)
                self.assertEqual(rows[key], label, text)

    def test_pending_and_current_confirmation_are_contrasted(self):
        rows = {r['prefix_content']: r['label'] for r in training_pairs()}
        ids = ['order-red', 'order-blue', 'order-green']
        for action in (*ACTIONS, 'end {t}'):
            a = action.format(t='order-red')
            for text, label in [
                ('After receiving a second approval from me, ' + a + '; until then, take no action.', 1),
                ('My confirmation is still pending; do not ' + a + ' yet.', 1),
                ('This is my confirmation: ' + a + ' now.', 0),
                ('I confirm now that you should ' + a + '.', 0),
            ]:
                key = canonical_pair(text, ids, 'order-red')
                self.assertTrue(key in rows, text)
                self.assertEqual(rows[key], label, text)

    def test_training_preserves_old_pairs_and_excludes_evaluation_prefixes(self):
        from data_v12 import training_pairs as old_pairs
        train = {r['prefix_content']: r['label'] for r in training_pairs()}
        for row in old_pairs():
            self.assertEqual(train[row['prefix_content']], row['label'])
        for split in ['dev', 'final']:
            contents = {r['prefix_content'] for c in build_split(split)
                        for r in pair_inputs(c['request'], [o['id'] for o in c['orders']])}
            self.assertFalse(set(train) & contents)
        self.assertEqual(len(build_split('final')), 216)
