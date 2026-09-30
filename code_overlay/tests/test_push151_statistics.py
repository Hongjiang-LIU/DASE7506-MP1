import copy
import unittest
from unittest import mock
import numpy as np
import torch

from push145_mkn_build import effective_counts, make_tables
import push151_statistics as subject


class StatisticsDriverTests(unittest.TestCase):
    def setUp(self):
        self.tables, _ = make_tables(effective_counts(np.array([1,2,3,4,1,2,3,5,1,2,3,4,1,2,3,5], dtype=np.int64)), scale=.7)

    def test_valid_control_payload(self):
        subject.validate_tables(self.tables, self.tables, subject.tensor_bytes(self.tables))

    def test_exact_budget_required(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            subject.validate_tables(self.tables, self.tables, subject.tensor_bytes(self.tables)-1)

    def test_bigram_cannot_change(self):
        altered = copy.deepcopy(self.tables)
        altered['unigram'][0] += .001
        with self.assertRaisesRegex(ValueError, 'unigram'):
            subject.validate_tables(self.tables, altered, subject.tensor_bytes(altered))

    def test_non_normalized_context_rejected(self):
        altered = copy.deepcopy(self.tables)
        altered['backoffs_3'][0] *= .5
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            subject.validate_tables(self.tables, altered, subject.tensor_bytes(altered))

    def test_cannot_change_non_statistical_weights(self):
        parent = dict(implementation='push150_expand', protocol='test', train_tokens=638976000,
                      config=dict(temperature=1.3,mkn_shapes={}), model={'token.weight':torch.ones(2,2)})
        other = copy.deepcopy(parent)
        other['model']['token.weight'][0,0] = 2
        with self.assertRaisesRegex(ValueError, 'Non-statistical'):
            subject.assert_neural_unchanged(parent, other)

    def test_cannot_change_temperature(self):
        parent = dict(implementation='push150_expand', protocol='test', train_tokens=638976000,
                      config=dict(temperature=1.3,mkn_shapes={}), model={})
        other = copy.deepcopy(parent)
        other['config']['temperature'] = 1.25
        with self.assertRaisesRegex(ValueError, 'Only table shapes'):
            subject.assert_neural_unchanged(parent, other)

    def test_cannot_erase_ancestry(self):
        parent = dict(implementation='push150_expand', protocol='test', train_tokens=638976000,
                      config={}, model={})
        other = copy.deepcopy(parent)
        other['train_tokens'] = 0
        with self.assertRaisesRegex(ValueError, 'ancestry'):
            subject.assert_neural_unchanged(parent, other)

    def test_corpus_metadata_reader_forbidden_before_io(self):
        with mock.patch('pathlib.Path.read_text', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'corpus'):
                subject.read('wikitext_test.txt')

    def test_unsafe_child_path(self):
        for name in ('../test', 'C:/test', '/test', 'runs//test'):
            with self.assertRaises(ValueError):
                subject.child(name)


if __name__ == '__main__':
    unittest.main()
