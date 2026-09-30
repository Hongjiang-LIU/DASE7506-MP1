"""Synthetic checkpoint/metadata fixtures only; no data, GPU or real weights."""
import ast
import copy
import inspect
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

import push151_average as average


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(3, 2)
        self.head = nn.Linear(2, 3, bias=False)
        self.head.weight = self.token.weight
        self.linear = nn.Linear(2, 2)
        self.register_buffer('float_buffer', torch.tensor([.1, .2]))
        self.register_buffer('integer_buffer', torch.tensor([1, 2], dtype=torch.int64))
        self.register_buffer('rope_constant', torch.tensor([.5]), persistent=False)


def state_copy(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def states():
    model = TinyTiedModel()
    parent = state_copy(model)
    names = {name for group in average.parameter_groups(model) for name in group}
    inputs = [copy.deepcopy(parent), copy.deepcopy(parent)]
    for index, item in enumerate(inputs):
        for name in names:
            item[name].fill_(index*2.)
    return model, parent, inputs


def plan_fixture():
    snapshots = {name: dict(record, path='runs/fake/'+name+'.pt') for name, record in average.SNAPSHOTS.items()}
    return dict(parent=dict(path='runs/fake/parent.pt',sha256=average.PARENT_SHA,
        implementation=average.IMPLEMENTATION,selected_lineage_targets=average.PARENT_TARGETS),
        phase_a=dict(snapshots=snapshots,candidates=copy.deepcopy(average.CANDIDATES),control_name='control',
            fixed_config=copy.deepcopy(average.FIXED_CONFIG),allowed_snapshot_config_differences=['temperature']))


def checkpoint_fixture():
    model, state, inputs = states()
    parent = dict(protocol=average.PROTOCOL,implementation=average.IMPLEMENTATION,train_tokens=average.PARENT_TARGETS,
        config=dict(average.FIXED_CONFIG,width=2,mkn_shapes={'unigram':[3]}),model=copy.deepcopy(state),
        ancestry={'already_shared':123},seed=211)
    snapshots = {}
    for name, item in average.SNAPSHOTS.items():
        value=copy.deepcopy(parent)
        value.update(selected_step=item['step'],weights=item['weights'],parent_sha256='c'*64,
            train_tokens=average.SHARED_BASE_TARGETS+item['step']*8192,training_recipe=dict(seed=211,steps=12000))
        value['config']['temperature']=1.25
        snapshots[name]=value
    return model,parent,snapshots


class AveragingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_mean_only_parameters_and_keep_tied_alias(self):
        model,parent,inputs=states(); before=copy.deepcopy(parent)
        result=average.average_state(parent,inputs,model)
        for group in average.parameter_groups(model):
            for name in group:
                self.assertTrue(torch.equal(result[name],torch.ones_like(result[name])))
                self.assertEqual(result[name].dtype,torch.float32)
        self.assertIs(result['token.weight'],result['head.weight'])
        self.assertTrue(model.token.weight is model.head.weight)
        for name in ('float_buffer','integer_buffer'):
            self.assertTrue(torch.equal(result[name],before[name]))
        self.assertTrue(all(torch.equal(parent[name],value) for name,value in before.items()))
        self.assertTrue(torch.equal(model.rope_constant,torch.tensor([.5])))

    def test_buffer_difference_rejected_even_if_floating(self):
        for name in ('float_buffer','integer_buffer'):
            model,parent,inputs=states();inputs[0][name][0]+=1
            with self.assertRaises(ValueError):average.average_state(parent,inputs,model)

    def test_missing_or_extra_state_key_rejected(self):
        model,parent,inputs=states();inputs[0].pop('linear.bias')
        with self.assertRaises(ValueError):average.average_state(parent,inputs,model)
        model,parent,inputs=states();inputs[0]['undeclared']=torch.zeros(1)
        with self.assertRaises(ValueError):average.average_state(parent,inputs,model)

    def test_shape_dtype_nonfinite_rejected(self):
        for replacement in (torch.zeros(1),torch.zeros(2,dtype=torch.float64),torch.full((2,),float('nan'))):
            model,parent,inputs=states();inputs[0]['linear.bias']=replacement
            with self.assertRaises(ValueError):average.average_state(parent,inputs,model)

    def test_tied_value_mismatch_rejected_before_loading(self):
        model,parent,inputs=states();inputs[0]['head.weight'][0,0]=3.
        with self.assertRaises(ValueError):average.average_state(parent,inputs,model)

    def test_untied_model_rejected(self):
        model,parent,inputs=states();model.head.weight=nn.Parameter(model.head.weight.detach().clone())
        with self.assertRaises(ValueError):average.average_state(parent,inputs,model)

    def test_average_does_not_overflow_float32_intermediate(self):
        model,parent,inputs=states()
        for item in inputs:
            for group in average.parameter_groups(model):
                for name in group:item[name].fill_(3e38)
        result=average.average_state(parent,inputs,model)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in result.values()))
        self.assertEqual(result['linear.bias'][0].item(),torch.tensor(3e38).item())

    def test_fixed_two_candidate_menu_and_source_hashes(self):
        value=plan_fixture();average.validate_plan(value)
        value['phase_a']['candidates'][0]['sources'][0]['weight']=.75
        with self.assertRaises(ValueError):average.validate_plan(value)
        value=plan_fixture();value['phase_a']['snapshots']['ema_10000']['sha256']='d'*64
        with self.assertRaises(ValueError):average.validate_plan(value)

    def test_no_temperature_or_weight_grid(self):
        model,parent,inputs=states()
        with self.assertRaises(ValueError):average.average_state(parent,inputs,model,weights=(.25,.75))
        value=plan_fixture();value['phase_a']['fixed_config']['temperature']=1.25
        with self.assertRaises(ValueError):average.validate_plan(value)

    def test_shared_trajectory_identity_and_parent_last_ema(self):
        model,parent,snapshots=checkpoint_fixture();average.verify_checkpoints(parent,snapshots)
        snapshots['ema_10000']['parent_sha256']='d'*64
        with self.assertRaises(ValueError):average.verify_checkpoints(parent,snapshots)
        model,parent,snapshots=checkpoint_fixture();snapshots['ema_12000']['model']['linear.bias'][0]+=1.
        with self.assertRaises(ValueError):average.verify_checkpoints(parent,snapshots)

    def test_only_declared_config_difference_allowed(self):
        model,parent,snapshots=checkpoint_fixture();snapshots['ema_10000']['config']['width']=4
        with self.assertRaises(ValueError):average.verify_checkpoints(parent,snapshots)
        model,parent,snapshots=checkpoint_fixture();snapshots['ema_10000']['config']['temperature']=1.2
        with self.assertRaises(ValueError):average.verify_checkpoints(parent,snapshots)

    def test_lineage_is_union_not_sum_and_metadata_separate(self):
        model,parent,snapshots=checkpoint_fixture();before=copy.deepcopy(parent)
        value=average.make_average(parent,snapshots,model,average.CANDIDATES[0],{'plan_sha256':'e'*64})
        self.assertEqual(value['train_tokens'],638976000)
        self.assertEqual(value['new_training_targets'],0)
        self.assertEqual(value['config'],parent['config'])
        self.assertEqual(value['ancestry']['already_shared'],123)
        self.assertTrue(value['averaging']['ancestor_targets_not_summed'])
        self.assertIsNone(value['selected_step'])
        self.assertEqual(parent['ancestry'],before['ancestry'])

    def test_exclusive_artifact_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json';average.write_json(path,{'test':True})
            with self.assertRaises(FileExistsError):average.write_json(path,{'test':False})
            path=Path(directory)/'synthetic.pt';average.save_checkpoint(path,{'x':torch.ones(1)})
            with self.assertRaises(FileExistsError):average.save_checkpoint(path,{'x':torch.zeros(1)})

    def test_test_split_and_corpus_hash_rejected_before_io(self):
        with self.assertRaises(ValueError):average.load_one_split('test',root=Path('nonexistent'))
        with self.assertRaises(ValueError):average.sha('wikitext_test.txt')
        with self.assertRaises(ValueError):average.read('wikitext_test.txt')

    def test_source_has_no_training_or_generic_data_loading_calls(self):
        tree=ast.parse(inspect.getsource(average))
        calls={ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)}
        self.assertFalse({'load_data','common.load_data','optimizer.step','loss.backward'} & calls)
        self.assertEqual(average.PARENT_TARGETS,638976000)
        self.assertEqual(len(average.CANDIDATES),2)


if __name__=='__main__':unittest.main()
