"""Synthetic CPU checks only: no course corpus, checkpoint or scoring."""
import copy
from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
import torch

from push145_mkn import MKNTables
from push145_mkn_build import effective_counts, make_tables
import push151_prune as prune


def fixture():
    ids = np.array(([1,2,3,1,2,4,1,2,3,5,6,1,2,7,1,2,3,5]*3)+[8,9,10,11,12],dtype=np.int64)
    effective = effective_counts(ids)
    parent,_ = make_tables(effective,scale=.7,minimum=2,trigram_minimum=1)
    raw = prune.raw_context_counts_from_ids(ids)
    return ids,effective,raw,parent


def packed(values):
    result = 0
    for value in values:
        result = (result << 11) | int(value)
    return result


class PruningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_raw_context_counts_exact_successor_denominator(self):
        ids = np.arange(1,9,dtype=np.int64)
        for order,(contexts,counts) in prune.raw_context_counts_from_ids(ids).items():
            expected = Counter(packed(ids[start:start+order-1]) for start in range(len(ids)-order+1))
            self.assertEqual(dict(zip(contexts.tolist(),counts.tolist())),dict(expected))
            self.assertEqual(int(counts.sum()),len(ids)-order+1)
            self.assertNotIn(packed(ids[-(order-1):]),contexts)

    def test_raw_counts_are_not_continuation_totals(self):
        _,effective,raw,_ = fixture()
        self.assertNotEqual(int(raw[3][1].sum()),int(effective[3][1].sum()))

    def test_proxy_matches_isolated_full_distribution_kl(self):
        q = np.array([.2,.3,.5]); mass = .4; context = .25
        p = (1-mass)*q; p[1] += mass
        expected = context*np.sum(p*np.log(p/q))
        self.assertAlmostEqual(float(prune.isolated_kl_proxy(mass,q[1],context)),expected,places=14)

    def test_proxy_zero_and_fixed_context_mass_monotonicity(self):
        mass = np.array([0.,.01,.1,.4,.9])
        value = prune.isolated_kl_proxy(mass,.07,.02)
        self.assertEqual(value[0],0.)
        self.assertTrue(np.all(np.diff(value)>0))
        self.assertEqual(float(prune.isolated_kl_proxy(.4,.2,0.)),0.)
        # For fixed normalizer and discount, increasing count increases mass.
        self.assertTrue(np.all(np.diff(prune.isolated_kl_proxy((np.arange(1,5)-.3)/20,.2,.1))>0))

    def test_proxy_rejects_invalid_or_nonfinite_input(self):
        for mass,q,context in [(-.1,.2,.5),(1.,.2,.5),(.2,0.,.5),(.2,1.,.5),(.2,.5,1.1),(np.nan,.5,.5)]:
            with self.subTest(values=(mass,q,context)),self.assertRaises(ValueError):
                prune.isolated_kl_proxy(mass,q,context)

    def test_budget_charges_context_once_and_keeps_later_affordable_entry(self):
        keys = np.array([packed([1,2,3]),packed([4,5,6]),packed([1,2,7])])
        keep,report = prune.deterministic_budget_mask([3,2,1],[3,2,1],[1,1,1],[3,3,3],keys,
            fixed_bytes=12,budget_bytes=12+16+2*6)
        np.testing.assert_array_equal(keep,[True,False,True])
        self.assertEqual(report['used_tensor_bytes'],40)
        self.assertEqual(report['selected_contexts'],1)
        self.assertEqual(report['selected_entries'],2)

    def test_context_identity_includes_order(self):
        keys = np.array([packed([0,1,2]),packed([0,0,1,3])])
        keep,report = prune.deterministic_budget_mask([2,1],[2,1],[1,1],[3,4],keys,fixed_bytes=12,budget_bytes=12+28)
        np.testing.assert_array_equal(keep,[True,False])
        self.assertEqual(report['used_tensor_bytes'],34)

    def test_declared_deterministic_tie_break(self):
        priority = [1,1,1,1,1]
        score = [1,2,2,2,2]
        counts = [3,1,2,2,2]
        orders = [3,3,4,3,3]
        keys = [2049,2050,2051,2052,2051]
        keep,_ = prune.deterministic_budget_mask(priority,score,counts,orders,keys,fixed_bytes=12,budget_bytes=34)
        np.testing.assert_array_equal(keep,[False,False,False,False,True])
        permutation = [4,2,0,3,1]
        again,_ = prune.deterministic_budget_mask(np.array(priority)[permutation],np.array(score)[permutation],
            np.array(counts)[permutation],np.array(orders)[permutation],np.array(keys)[permutation],fixed_bytes=12,budget_bytes=34)
        self.assertEqual(np.array(keys)[permutation][again].tolist(),[2051])
        self.assertEqual(np.array(orders)[permutation][again].tolist(),[3])

    def test_insufficient_budget_and_key_overflow_refused(self):
        with self.assertRaises(ValueError):
            prune.deterministic_budget_mask([1],[1],[1],[3],[1],fixed_bytes=12,budget_bytes=11)
        with self.assertRaises(ValueError):
            prune.deterministic_budget_mask([1],[1],[1],[5],[1<<55],fixed_bytes=12,budget_bytes=40)
        keep,_ = prune.deterministic_budget_mask([1],[1],[1],[5],[(1<<55)-1],fixed_bytes=12,budget_bytes=34)
        self.assertTrue(keep[0])

    def test_sparse_lower_queries_match_full_mkn_distribution(self):
        ids,effective,_,_ = fixture()
        tables,_ = make_tables(effective,scale=.7,minimum=1,trigram_minimum=1)
        model = MKNTables({name:list(value.shape) for name,value in tables.items()})
        model.load_state_dict(tables)
        x = torch.from_numpy(ids[:12].reshape(2,6))
        dense = model.probabilities(x,4)
        contexts,words,expected = [],[],[]
        for row in range(2):
            for position in range(2,6):
                word = (row+position)%8
                contexts.append(packed(x[row,position-2:position+1]))
                words.append(word); expected.append(dense[row,position,word].item())
        actual = prune.lower_target_probabilities(tables,contexts,words,4)
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)

    def test_reference_keeps_fixed_base_and_constructs_upper_orders_float64(self):
        _,effective,_,parent = fixture()
        reference = prune.unpruned_reference_tables(effective,parent)
        for name in prune.FIXED_NAMES:
            self.assertEqual(reference[name].dtype,parent[name].dtype)
            self.assertTrue(torch.equal(reference[name],parent[name]))
        self.assertFalse(any(name.endswith('_5') for name in reference))
        for order in (3,4):
            self.assertEqual(reference[f'masses_{order}'].dtype,torch.float64)
            self.assertEqual(reference[f'backoffs_{order}'].dtype,torch.float64)
            keys,counts = effective[order]
            _,starts,lengths = np.unique(keys >> 11,return_index=True,return_counts=True)
            base,_ = prune.discounts(counts)
            discounts = np.clip(base*.7,.01,[.99,1.99,2.99])
            expected = (counts-discounts[np.minimum(counts,3)-1])/np.repeat(np.add.reduceat(counts.astype(np.float64),starts),lengths)
            np.testing.assert_array_equal(reference[f'masses_{order}'].numpy(),expected)
            # There really are values that would change if rounded via FP32.
            self.assertTrue(np.any(expected != expected.astype(np.float32).astype(np.float64)))
            totals = np.add.reduceat(reference[f'masses_{order}'].numpy(),starts)+reference[f'backoffs_{order}'].numpy()
            np.testing.assert_allclose(totals,1.,atol=2e-16,rtol=0.)

    def test_fp64_reference_sparse_recursion_matches_manual_mixture(self):
        _,effective,_,parent = fixture()
        reference = prune.unpruned_reference_tables(effective,parent)
        context,word = packed([1,2,3]),1
        probability = float(reference['unigram'][word])
        for order in (2,3,4):
            suffix = context & ((1 << (11*(order-1)))-1)
            contexts = reference[f'contexts_{order}'].tolist()
            if suffix not in contexts:
                continue
            index = contexts.index(suffix)
            start,end = reference[f'offsets_{order}'][index:index+2].tolist()
            probability *= float(reference[f'backoffs_{order}'][index])
            words = reference[f'words_{order}'][start:end].tolist()
            if word in words:
                probability += float(reference[f'masses_{order}'][start+words.index(word)])
        result = prune.lower_target_probabilities(reference,[context],[word],4)
        self.assertEqual(result.dtype,np.float64)
        self.assertAlmostEqual(result[0],probability,places=16)

    def test_parent_rebuild_and_two_exact_budget_caps(self):
        _,effective,raw,parent = fixture()
        variants = prune.build_pruned_candidates(effective,raw,parent)
        self.assertEqual(len(variants),2)
        for (tables,report),extra in zip(variants,prune.DEFAULT_EXTRA_BUDGETS):
            self.assertEqual(set(tables),prune.TABLE_NAMES)
            self.assertEqual(report['table_buffers'],21)
            self.assertEqual(prune.tensor_bytes(tables),report['used_tensor_bytes'])
            self.assertLessEqual(report['used_tensor_bytes'],prune.tensor_bytes(parent)+extra)
            self.assertEqual(report['budget_tensor_bytes'],prune.tensor_bytes(parent)+extra)
            self.assertTrue(report['original_parent_tables_rebuilt_bitwise'])
            self.assertTrue(report['not_exact_complete_model_deletion_kl'])
            self.assertTrue(report['not_optimal_knapsack'])
            for name in prune.FIXED_NAMES:
                self.assertTrue(torch.equal(tables[name],parent[name]))
                self.assertEqual(tables[name].dtype,parent[name].dtype)
                self.assertNotEqual(tables[name].data_ptr(),parent[name].data_ptr())

    def test_large_budget_matches_unpruned_original_builder_bitwise(self):
        _,effective,raw,parent = fixture()
        variants = prune.build_pruned_candidates(effective,raw,parent,extra_budgets=(8*1024*1024,))
        expected,_ = make_tables(effective,scale=.7,minimum=1,trigram_minimum=1)
        for name,value in expected.items():
            self.assertTrue(torch.equal(value,variants[0][0][name]),name)

    def test_selected_masses_preserve_unpruned_denominators(self):
        _,effective,raw,parent = fixture()
        tables,_ = prune.build_pruned_candidates(effective,raw,parent,extra_budgets=(0,))[0]
        complete,_ = make_tables(effective,scale=.7,minimum=1,trigram_minimum=1)
        for order in prune.ORDERS:
            def entry_map(source):
                contexts = source[f'contexts_{order}'].numpy()
                keys = (np.repeat(contexts,np.diff(source[f'offsets_{order}'].numpy())) << 11) | source[f'words_{order}'].numpy().astype(np.int64)
                return dict(zip(keys,source[f'masses_{order}'].numpy()))
            all_entries = entry_map(complete)
            for key,value in entry_map(tables).items():
                self.assertEqual(value,all_entries[key])

    def test_backoff_mass_and_full_probabilities_remain_normalized_and_causal(self):
        ids,effective,raw,parent = fixture()
        for tables,_ in prune.build_pruned_candidates(effective,raw,parent):
            for order in range(2,6):
                offsets = tables[f'offsets_{order}'].numpy()
                mass = tables[f'masses_{order}'].numpy()
                for index,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
                    self.assertAlmostEqual(float(mass[start:end].astype(np.float64).sum())+tables[f'backoffs_{order}'][index].item(),1.,places=6)
            model = MKNTables({name:list(value.shape) for name,value in tables.items()})
            model.load_state_dict(tables)
            x = torch.from_numpy(ids[:16].reshape(2,8))
            full = model.probabilities(x,5)
            self.assertTrue(torch.isfinite(full).all())
            torch.testing.assert_close(full.sum(-1),torch.ones(2,8),atol=1e-6,rtol=0.)
            self.assertTrue(torch.equal(full[:,:4],model.probabilities(x[:,:4],5)))

    def test_scores_prepared_once_for_both_budgets(self):
        _,effective,raw,parent = fixture()
        with patch.object(prune,'isolated_kl_proxy',wraps=prune.isolated_kl_proxy) as score:
            prune.build_pruned_candidates(effective,raw,parent)
        self.assertEqual(score.call_count,3)

    def test_parent_counts_and_raw_inputs_not_mutated(self):
        _,effective,raw,parent = fixture()
        before_effective,before_raw,before_parent = copy.deepcopy((effective,raw,parent))
        first = prune.build_pruned_candidates(effective,raw,parent)
        second = prune.build_pruned_candidates(effective,raw,parent)
        for group,old in ((effective,before_effective),(raw,before_raw)):
            for order in group:
                for value,reference in zip(group[order],old[order]):
                    np.testing.assert_array_equal(value,reference)
        for name in parent:
            self.assertTrue(torch.equal(parent[name],before_parent[name]))
        for (left,left_report),(right,right_report) in zip(first,second):
            self.assertEqual(left_report,right_report)
            for name in left:
                self.assertTrue(torch.equal(left[name],right[name]))

    def test_changed_parent_and_wrong_raw_denominator_refused(self):
        _,effective,raw,parent = fixture()
        parent['unigram'][0] += .001
        with self.assertRaisesRegex(ValueError,'rebuild mismatch'):
            prune.build_pruned_candidates(effective,raw,parent)
        _,effective,raw,parent = fixture()
        raw[3][1][0] += 1
        with self.assertRaisesRegex(ValueError,'denominators'):
            prune.build_pruned_candidates(effective,raw,parent)

    def test_fixed_dtypes_and_maximum_token_encoding(self):
        ids = np.array([2047,2046,2047,2047,2045,2047,2046,2047,2047,2045]*2,dtype=np.int64)
        effective = effective_counts(ids)
        parent,_ = make_tables(effective,scale=.7)
        raw = prune.raw_context_counts_from_ids(ids)
        tables,_ = prune.build_pruned_candidates(effective,raw,parent,extra_budgets=(1024,))[0]
        for name,value in tables.items():
            expected = torch.int64 if name.startswith('contexts_') else torch.int32 if name.startswith('offsets_') else torch.int16 if name.startswith('words_') else torch.float32
            self.assertEqual(value.dtype,expected,name)
            if name.startswith('words_') and value.numel():
                self.assertLessEqual(value.max().item(),2047)


if __name__=='__main__':
    unittest.main()
