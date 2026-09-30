"""Synthetic-only Phase-D preparation tests; no real data, GPU or checkpoints."""
import ast
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from push145_mkn import MKNTables
from push145_mkn_build import effective_counts, make_tables
from push145_mkn_select import target_probabilities
from push150_crossfit import contiguous_folds, effective_counts_segments
import push150_oof_prepare as prepare


class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ids = np.random.default_rng(211).integers(0, 9, 1700, dtype=np.int64)
        cls.buffers, _ = make_tables(effective_counts(cls.ids), scale=.55)
        cls.arrays = {name: value.numpy() for name,value in cls.buffers.items()}

    def test_four_columns_match_existing_target_only_helper(self):
        ids = self.ids[:311]
        actual = prepare.fold_target_probabilities(self.arrays, ids)
        self.assertEqual(actual.shape, (311,4))
        self.assertEqual(actual.dtype, np.float32)
        for order in prepare.MAXIMUM_ORDERS:
            expected = target_probabilities(self.arrays, ids[:-1][None,:], ids[1:][None,:], maximum=order)
            np.testing.assert_array_equal(actual[:-1,order-2], expected[0])
        np.testing.assert_array_equal(actual[-1], np.zeros(4,np.float32))

    def test_all_orders_and_random_window_offsets_equal_full_distribution(self):
        values = prepare.fold_target_probabilities(self.arrays, self.ids[:700])
        tables = MKNTables({name:list(value.shape) for name,value in self.buffers.items()})
        tables.load_state_dict(self.buffers)
        for start in (0,1,37,190):
            for length in (1,2,3,11,256):
                x = torch.from_numpy(self.ids[start:start+length].copy())[None,:]
                y = torch.from_numpy(self.ids[start+1:start+length+1].copy())[None,:]
                for order in prepare.MAXIMUM_ORDERS:
                    expected = tables.probabilities(x,max_order=order).gather(-1,y.unsqueeze(-1)).squeeze(-1).numpy()
                    actual = prepare.gather_window_probabilities(values,np.asarray([start]),length,order)
                    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-5)

    def test_first_three_columns_reset_not_global_context(self):
        values = np.broadcast_to(np.asarray([.1,.2,.3,.4],np.float32),(100,4)).copy()
        got = prepare.gather_window_probabilities(values,np.asarray([17,40]),7)
        np.testing.assert_array_equal(got,np.asarray([[.1,.2,.3,.4,.4,.4,.4]]*2,np.float32))
        np.testing.assert_array_equal(prepare.window_columns(7),[0,1,2,3,3,3,3])

    def test_fold_end_sentinel_rejects_cross_boundary(self):
        values = np.full((30,4),.2,np.float32); values[[9,19,29]]=0
        prepare.gather_window_probabilities(values,np.asarray([0,10,20]),9)
        for start in (1,6,9,11,21):
            with self.assertRaises(ValueError):
                prepare.gather_window_probabilities(values,np.asarray([start]),9)

    def test_valid_ranges_include_last_legal_start_and_exclude_last_illegal(self):
        folds = [dict(start=0,end=300),dict(start=300,end=650)]
        np.testing.assert_array_equal(prepare.valid_start_ranges(folds),[[0,44],[300,394]])
        with self.assertRaises(ValueError):
            prepare.valid_start_ranges([dict(start=0,end=256)])
        with self.assertRaises(ValueError):
            prepare.valid_start_ranges([dict(start=0,end=300),dict(start=301,end=650)])

    def test_sampler_uniform_over_starts_not_folds_and_reproducible(self):
        folds = [dict(start=0,end=257),dict(start=257,end=517)]
        one = prepare.sample_window_starts(np.random.default_rng(211),4000,folds)
        two = prepare.sample_window_starts(np.random.default_rng(211),4000,folds)
        np.testing.assert_array_equal(one,two)
        self.assertEqual(set(one),{0,257,258,259,260})
        # Each legal start, rather than each unequal-size region, has probability 1/5.
        self.assertTrue(all(650 < np.sum(one==start) < 950 for start in set(one)))
        for start in one:
            self.assertTrue(any(row['start']<=start and start+256<row['end'] for row in folds))

    def test_memmap_gather_has_same_behavior(self):
        values = prepare.fold_target_probabilities(self.arrays,self.ids[:400])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train_oof_probabilities.npy'
            with path.open('xb') as stream:
                np.save(stream,values,allow_pickle=False)
            mapped=np.load(path,mmap_mode='r',allow_pickle=False)
            np.testing.assert_array_equal(prepare.gather_window_probabilities(mapped,np.asarray([10]),256),
                                          prepare.gather_window_probabilities(values,np.asarray([10]),256))
            del mapped

    def test_empty_higher_order_tables_back_off_without_invalid_index(self):
        buffers,_=make_tables(effective_counts(np.arange(20,dtype=np.int64)),scale=.7)
        self.assertEqual(len(buffers['contexts_4']),0)
        values=prepare.fold_target_probabilities({name:value.numpy() for name,value in buffers.items()},np.arange(12,dtype=np.int64))
        np.testing.assert_array_equal(values[:,1],values[:,2])
        np.testing.assert_array_equal(values[:,2],values[:,3])

    def test_held_out_fold_tables_no_artificial_join_and_probabilities_parity(self):
        first=np.asarray([1,2,1,2,1,2,1,2],np.int64)
        held=np.asarray([8,7,6,8,7,6,8,7],np.int64)
        last=np.asarray([3,4,3,4,3,4,3,4],np.int64)
        effective=effective_counts_segments([first,last])
        # A (2,3) join must not be introduced by the omitted middle region.
        self.assertNotIn((2<<11)|3,set(effective[2][0]))
        buffers,_=make_tables(effective,.55)
        arrays={name:value.numpy() for name,value in buffers.items()}
        values=prepare.fold_target_probabilities(arrays,held)
        tables=MKNTables({name:list(value.shape) for name,value in buffers.items()});tables.load_state_dict(buffers)
        expected=tables.probabilities(torch.from_numpy(held[:-1].copy())[None,:]).gather(
            -1,torch.from_numpy(held[1:].copy())[None,:,None]).squeeze(-1).numpy()
        np.testing.assert_allclose(prepare.gather_window_probabilities(values,np.asarray([0]),7),expected,atol=1e-7,rtol=1e-5)

    def test_fixed_parity_records_forward_label_cost(self):
        ids=self.ids[:320];values=prepare.fold_target_probabilities(self.arrays,ids)
        progress=dict(parity_forward_target_occurrences_started=0,parity_forward_target_occurrences_completed=0)
        record=prepare.parity_check(self.buffers,ids,values,progress)
        self.assertTrue(record['passes'])
        self.assertEqual(record['comparisons'],3*4*4)
        expected=3*4*(1+2+3+256)
        self.assertEqual(record['full_distribution_forward_target_occurrences'],expected)
        self.assertEqual(progress['parity_forward_target_occurrences_completed'],expected)
        self.assertEqual(progress['parity_forward_target_occurrences_started'],expected)

    def test_recursive_source_graph_and_safe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            # Tiny fixtures are test assets, not edits to any real project source.
            (root/'a.py').write_text('import b\nimport numpy\n')
            (root/'b.py').write_text('from c import value\n')
            (root/'c.py').write_text('value=1\n')
            graph=prepare.source_graph(root,['a.py'])
            self.assertEqual(set(graph),{'a.py','b.py','c.py'})
            for name,value in graph.items():
                self.assertEqual(value,prepare.sha_file(root/name))
            with self.assertRaises(ValueError):
                prepare.child(root,'../escape.json')
            with self.assertRaises(ValueError):
                prepare.sha_file(root/'wikitext_test.txt')

    def test_no_test_or_validation_loader_gpu_or_training_call(self):
        tree=ast.parse(inspect.getsource(prepare))
        split_calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                     and node.func.id=='load_one_split']
        self.assertEqual(len(split_calls),1)
        self.assertEqual(split_calls[0].args[0].value,'train')
        self.assertNotIn('load_data(',inspect.getsource(prepare))
        self.assertNotIn('.cuda(',inspect.getsource(prepare))
        self.assertNotIn('.backward(',inspect.getsource(prepare))
        self.assertEqual(prepare.PLAN_SHA256,'65800e86fac769df299f3794914a287189274adbb3442e56fa9e7e24367ec25a')

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            prepare.fold_target_probabilities(self.arrays,np.asarray([1,2048]))
        with self.assertRaises(ValueError):
            prepare.gather_window_probabilities(np.ones((40,4),np.float64),np.asarray([0]),3)
        with self.assertRaises(ValueError):
            prepare.gather_window_probabilities(np.ones((40,4),np.float32),np.asarray([-1]),3)
        with self.assertRaises(ValueError):
            prepare.gather_window_probabilities(np.ones((40,4),np.float32),np.asarray([1.5]),3)

    def test_failure_is_preserved_and_directory_cannot_be_reused(self):
        import argparse
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(run_dir=Path(directory)/'new-run',parent=Path('parent.pt'),expected_parent_sha256='0'*64)
            with mock.patch.object(prepare,'check_inputs',side_effect=ValueError('synthetic failure')):
                with self.assertRaisesRegex(ValueError,'synthetic failure'):
                    prepare.prepare(args)
            self.assertEqual(prepare.read_json(args.run_dir/'prepare_failure.json')['status'],'failed')
            self.assertFalse((args.run_dir/'metadata.json').exists())
            with self.assertRaises(FileExistsError):
                prepare.prepare(args)


if __name__=='__main__':
    unittest.main()
