"""Final A/B/C/D VALIDATION selection and occurrence accounting, never scoring.

Rechecks immutable evidence and CPU checkpoint metadata/tensors only. Corpus
text is neither read nor hashed. The OOF training array is only hashed/header-
checked, never decoded. Missing or failed work refuses automatic selection.
Actual final CPU/resource audit, freeze, and any test authority remain separate.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import push150_phase_a_report as a
import push150_phase_bc_report as bc


ROOT = Path(__file__).resolve().parent
require, natural, finite = a.require, a.natural, a.finite
read, child, checkpoint, same_tensors = a.read, a.child, a.checkpoint, a.same_tensors
# Corrupt/partial checkpoints may raise EOFError or pickle errors rather than
# RuntimeError. Preserve them as integrity failures, never skip the report.
ERRORS = (Exception,)
D_PLAN_SHA = '65800e86fac769df299f3794914a287189274adbb3442e56fa9e7e24367ec25a'
D_SOURCES = {
    'push150_oof_prepare.py': '8bb9780d38c2b457ae2da73c22e1337950bf9e6a47b8d7a3141f0a20aea7a34c',
    'push150_oof_train.py': '01036a323891999578dd0f26ee8bec9552c6be4867270fac1a5d5c0ed68aa394',
    'push150_phase_a_report.py': 'd2d123a1cf895a4e94648ae74719a22b066d7fd0039e020ddae50caad5913d88',
    'push150_phase_bc_report.py': '251d1226acdc39b4482053139017709b3d35e103f7a1d6329a11a31bf83efb88',
}
FROZEN_SOURCES = dict(bc.FROZEN_SOURCES, **D_SOURCES)
RUN_NAMES = ('oof-control','oof-joint')
HISTORICAL_METADATA_EVIDENCE = {
    'push150_selection_refused_metadata.json': 'Preserved first refused selection; not a selection candidate',
    'push150_metadata_repair.json': 'Pre-test metadata-report repair disclosure; no predictor change',
    'push150_unit_tests_final.json': 'Historical pre-repair test run; not current-source unit evidence',
}
SNAPSHOT_KEYS = {'protocol','implementation','config','model','seed','parent_sha256','train_tokens',
                 'selected_step','weights','training_config','training_recipe','ancestry','selection',
                 'inference_architecture_unchanged_from_parent'}
EVIDENCE_DIRECTORIES = (
    'push150-init-control','push150-init-expanded','push150-init-dense','push150-feasibility','push150-mkn',
    'push150-control-cosine','push150-control-wsd','push150-expanded-cosine',
    'push150-control-cosine-calibrated','push150-control-wsd-calibrated','push150-expanded-cosine-calibrated',
    'push150-gate','push150-fast-dense','push150-suffix-init','push150-suffix-control','push150-suffix-joint',
    'push150-suffix-control-calibrated','push150-suffix-joint-calibrated','push150-oof-data',
    'push150-oof-control','push150-oof-joint','push150-oof-control-calibrated','push150-oof-joint-calibrated',
    'push150_followon_launches')


def sha(path):
    path = Path(path)
    if ((path.name == 'train_oof_probabilities.npy' and path.parent.name == 'push150-oof-data') or
            (path.name.endswith('.output.log') and path.parent.name == 'push150_followon_launches')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        return digest.hexdigest()
    return bc.sha(path)


def relative(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = Path(root)/path
    require(path.resolve().is_relative_to(Path(root).resolve()), 'Evidence is outside the code directory')
    return path.resolve().relative_to(Path(root).resolve()).as_posix()


def graph(root, roots):
    pending, found = list(roots), {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        require(name in FROZEN_SOURCES, 'Unpinned dependency: '+name)
        path = child(root,name); found[name] = sha(path)
        require(found[name] == FROZEN_SOURCES[name], 'Frozen source changed: '+name)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                       [entry.name for entry in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                filename = module.replace('.','/')+'.py'
                if child(root,filename).is_file():
                    pending.append(filename)
    return dict(sorted(found.items()))


def verify_latest_units(root, paths):
    """Freeze consumes ONE complete latest record; history counts stay separate."""
    paths = [child(root,relative(root,path)) for path in paths]
    evidence = bc.verify_units(root,paths)
    latest = evidence['evidence'][-1]
    sources = {p.name:sha(p) for p in sorted(root.glob('push150*.py'))}
    tests = {p.name:sha(p) for p in sorted((root/'tests').glob('test_push150*.py'))}
    require(bool(sources) and bool(tests), 'No push150 sources/tests found')
    require(latest['record']['source_sha256'] == sources and latest['record']['unit_source_sha256'] == tests,
            'Latest complete unit record must cover all current push150 Python files and tests')
    require('push150_select.py' in sources and 'test_push150_select.py' in tests,
            'Selector and its tests must be included in the latest full unit run')
    unit = dict(path=relative(root,latest['path']),sha256=latest['sha256'],executions=latest['record']['tests_run'])
    return unit,evidence


def verify_bc(root, row):
    path = root/'runs/push150_phase_bc_selection.json'; saved = read(path)
    row.update(path=relative(root,path),sha256=sha(path),record=saved)
    require(saved['status'] == 'phase_abc_choice_complete' and not saved['evidence_errors'] and
            saved['test_used'] is False and saved['cost_summary']['accounting_verified'] is True,
            'A/B/C evidence is incomplete or failed')
    require(saved['report_script_sha256'] == D_SOURCES['push150_phase_bc_report.py'], 'BC selector source differs')
    units = [child(root,relative(root,item['path'])) for item in saved['units']['evidence']]
    checked = bc.collect_report(root,units,root/'runs/push150-gate')
    expected, actual = copy.deepcopy(saved), copy.deepcopy(checked)
    for value in (expected,actual):
        value.pop('recorded_at_utc',None)
        for unit in value['units']['evidence']:
            unit['path'] = relative(root,unit['path'])
    require(actual == expected, 'A/B/C artifacts no longer reproduce its read-only evidence report')
    row['revalidated_without_scoring'] = True
    selected_a = saved['phase_a']['record']['winner']
    parent_path = child(root/'runs',selected_a['selected_run'])/'checkpoint.pt'
    require(sha(parent_path) == selected_a['checkpoint_sha256'], 'Phase A parent changed')
    parent = checkpoint(parent_path)
    require(parent['train_tokens'] == selected_a['selected_lineage_targets'], 'Phase A inherited cost differs')
    return saved,parent,parent_path,selected_a


def verify_prepared_schema(metadata, parent, parent_sha, selection_sha, build):
    require(metadata['status'] == 'completed' and metadata['test_used'] is False and metadata['plan_sha256'] == D_PLAN_SHA,
            'D preparation incomplete or wrong plan')
    require(metadata['parent_sha256'] == parent_sha and metadata['parent_lineage_targets'] == parent['train_tokens'] and
            metadata['parent_implementation'] == parent['implementation'] and metadata['parent_config'] == parent['config'] and
            metadata['phase_a_selection_sha256'] == selection_sha, 'D cache does not belong to the selected Phase A parent')
    bc.verify_data(metadata['train'],'train')
    require(metadata['train'] == build['train'] and metadata['full_train_table'] == build['full_table'] and
            metadata['fold_table_sha256'] == [row['sha256'] for row in build['folds']], 'Reused B table evidence differs')
    require(metadata['training_intermediate_only'] is True and metadata['include_in_inference_assets'] is False and
            metadata['no_validation_or_test_arrays'] is True and metadata['only_count_component_is_out_of_fold'] is True and
            metadata['neural_parent_saw_all_train'] is True, 'OOF cache training-only/bias disclosure differs')
    array = metadata['array']
    require(array['path'] == 'train_oof_probabilities.npy' and array['shape'] == [bc.TRAIN_TOKENS,4] and
            array['dtype'] == 'float32' and array['maximum_orders'] == [2,3,4,5], 'OOF array schema differs')
    require(metadata['window_context'] == 256 and metadata['window_target_span'] == 257 and
            metadata['window_columns'] == 'column=min(within_window_position,3); maximum_order=min(5,t+2)', 'Window reset differs')
    expected_folds = bc.fold_specifications()
    require(len(metadata['folds']) == 10, 'OOF preparation needs ten folds')
    for fold, spec, table in zip(metadata['folds'],expected_folds,build['folds']):
        start,end = spec['excluded_token_interval']; index = spec['fold']
        expected = dict(fold=index,start=start,end=end,valid_random_starts=end-start-256,
            sentinel_input_position=end-1,table_path=table['path'],table_sha256=table['sha256'],table_bytes=table['bytes'],
            target_probability_queries=4*(end-start-1),queried_label_positions=end-start-1)
        require(all(fold[key] == value for key,value in expected.items()), 'D fold/query/sentinel/table metadata differs')
        parity = fold['parity']
        require(parity['passes'] is True and parity['window_starts_local'] == [0,(end-start-257)//2,end-start-257] and
                parity['prefix_lengths'] == [1,2,3,256] and parity['maximum_orders'] == [2,3,4,5] and
                parity['comparisons'] == 48 and parity['full_distribution_forward_target_occurrences'] == 3144 and
                parity['atol'] == 1e-7 and parity['rtol'] == 1e-5 and
                0 <= finite(parity['max_absolute_difference'],'OOF parity error') <= 1.011e-5,
                'Fixed train-only parity checks are incomplete')
    costs = metadata['costs']
    expected_cost = dict(target_probability_queries=4*(bc.TRAIN_TOKENS-10),parity_forward_target_occurrences=31440,
        new_count_scan_token_occurrences=0,reused_table_scan_cost_counted_in_phase_b_only=True,
        new_neural_training_targets=0,new_gate_training_targets=0)
    require(all(costs[key] == value for key,value in expected_cost.items()), 'D preprocessing costs double-count B scans or differ')
    require(finite(costs['seconds_total'],'OOF preparation seconds') > 0, 'Invalid preparation duration')
    require(metadata['valid_random_starts'] == bc.TRAIN_TOKENS-10*256 and
            metadata['valid_label_positions'] == bc.TRAIN_TOKENS-10 and
            metadata['invalid_label_positions'] == [row['excluded_token_interval'][1]-1 for row in expected_folds],
            'D random-start/label count differs')
    parity = metadata['parity']
    require(parity['passes'] is True and parity['fold_checks'] == 10 and parity['atol'] == 1e-7 and parity['rtol'] == 1e-5,
            'Aggregate preparation parity missing')
    bc.near(parity['maximum_absolute_difference'],max(row['parity']['max_absolute_difference'] for row in metadata['folds']),
            'Aggregate preparation parity difference')


def verify_array_header(path, record):
    require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Prepared training array changed')
    # Read only the NPY header, not any train-derived probability values.
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1,0),(2,0)), 'Unsupported NPY header')
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1,0)
                                 else np.lib.format.read_array_header_2_0(stream))
        payload_offset = stream.tell()
    require(list(shape) == record['shape'] and dtype == np.dtype('float32') and not fortran,
            'OOF cache header disagrees with metadata')
    require(payload_offset+math.prod(shape)*dtype.itemsize == record['bytes'], 'OOF array payload size differs')


def prepared_file_pins(root, paths, verified_source_sha256):
    """Mirror the frozen loader's file pins, including preparation-only sources.

    The caller has already compared the preparation source graph against the
    frozen graph. These sources need not be imports of the training module.
    """
    pins = {str(Path(path).resolve()):sha(path) for path in paths}
    pins.update({str(child(root,name)):digest for name,digest in verified_source_sha256.items()})
    return pins


def verify_prepared(root, parent, parent_path, parent_sha, selection_sha, row):
    directory = root/'runs/push150-oof-data'
    metadata = read(directory/'metadata.json'); identity = read(directory/'identity.json')
    build = read(root/'runs/push150-gate/build.json'); gate_identity = read(root/'runs/push150-gate/identity.json')
    row.update(metadata=metadata,metadata_sha256=sha(directory/'metadata.json'),identity_sha256=sha(directory/'identity.json'))
    require(not (directory/'prepare_failure.json').exists(), 'Failed preparation requires manual reconciliation')
    verify_prepared_schema(metadata,parent,parent_sha,selection_sha,build)
    require(metadata['identity_sha256'] == row['identity_sha256'] and
            all(metadata[key] == value for key,value in identity.items()), 'OOF identity differs from completed metadata')
    expected_graph = graph(root,['push150_oof_prepare.py','push150_phase_a_report.py',parent['implementation']+'.py',
                                  *gate_identity['source_sha256']])
    require(metadata['source_sha256'] == expected_graph, 'OOF preparation source graph differs')
    require(metadata['runtime']['device'] == 'cpu' and metadata['runtime']['threads'] == 2 and
            metadata['runtime']['numpy'] == a.RUNTIME['numpy'] and
            metadata['runtime']['torch'] in ('2.7.1+cpu','2.7.1+cu128'), 'OOF preparation runtime differs')
    for key,value in dict(parent_path=str(parent_path.resolve()),phase_a_selection_path=str((root/'runs/push150_phase_a_selection.json').resolve()),
        phase_a_plan_sha256=a.PLAN_SHA,phase_a_report_script_sha256=D_SOURCES['push150_phase_a_report.py'],
        gate_run_dir=str((root/'runs/push150-gate').resolve()),gate_identity_sha256=sha(root/'runs/push150-gate/identity.json'),
        gate_build_sha256=sha(root/'runs/push150-gate/build.json'),gate_plan_sha256=bc.PLAN_SHA['gate']).items():
        require(metadata[key] == value, 'OOF preparation binding differs: '+key)
    started = read(directory/'prepare_started.json')
    require(started['expected_parent_sha256'] == parent_sha and started['plan_sha256'] == D_PLAN_SHA and
            started['test_used'] is False, 'OOF preparation launch differs')
    for fold in metadata['folds']:
        require(read(directory/f"fold-{fold['fold']:02d}.json") == fold, 'Preparation fold sidecar differs')
    verify_array_header(directory/metadata['array']['path'],metadata['array'])
    pins = prepared_file_pins(root,(directory/'metadata.json',directory/'identity.json',
        directory/metadata['array']['path'],root/'runs/push150_phase_a_selection.json',root/'runs/push150_phase_d_oof_plan.json'),
        expected_graph)
    provenance = dict(metadata_sha256=row['metadata_sha256'],identity_sha256=row['identity_sha256'],
        array_sha256=metadata['array']['sha256'],array_shape=metadata['array']['shape'],array_dtype='float32',
        phase_a_selection_sha256=selection_sha,plan_sha256=D_PLAN_SHA,
        gate_identity_sha256=metadata['gate_identity_sha256'],gate_build_sha256=metadata['gate_build_sha256'],
        folds=copy.deepcopy(metadata['folds']),shared_preparation_costs=copy.deepcopy(metadata['costs']),
        only_count_component_is_out_of_fold=True,neural_parent_previously_saw_all_training_text=True,
        reused_B_count_scans_counted_again_in_D=0,training_only_array_excluded_from_inference_assets=True)
    row.update(accounting_verified=True,array_values_decoded=False,source_sha256=expected_graph)
    return provenance,pins


def expected_recipe(config, objective):
    require(objective in ('control','joint'), 'Unknown D objective')
    recipe = a.expected_recipe({'schedule':'cosine'},config)
    recipe.update(steps=12000,eval_every=2000,seed=211,warmup=200,decay_steps=2000,tail_snapshot_start_step=10000,
        objective=objective,training_count_weight=.15,count_block_dropout=.5,
        count_block_dropout_granularity='one coin per complete minibatch; control consumes same coin',
        count_coin_rng='CPU sampler generator shared with legal-window start draws; no global dropout RNG use',
        sampler='uniform over all valid fold-contained 257-token starts',
        objective_description='pure neural+pointer NLL' if objective == 'control' else
            'OOF convex-mixture NLL, with whole count block independently dropped per minibatch',
        declared_raw_ema_validation_choices=13,additional_calibration_choices=15)
    return recipe


def verify_training_history(metrics, parent_targets, objective):
    require(metrics['status'] == 'completed' and metrics['completed_steps'] == 12000, 'D run incomplete or failed')
    require(not metrics['resume_history'] and metrics['discarded_completed_steps'] == 0,
            'Resumed D run requires manual merged choices and discarded-work accounting')
    history = metrics['validation_history']
    expected = [(0,'raw')]+[(step,label) for step in range(2000,12001,2000) for label in ('raw','ema')]
    require([(row['step'],row['weights']) for row in history] == expected and
            all(row['session'] == 0 for row in history), 'D needs exactly 13 ordered raw/EMA choices')
    for row in history:
        a.verify_measurement(row)
    best = min(history,key=lambda row:row['bpb'])
    require(all(metrics['best'][key] == best[key] for key in ('step','weights','bpb')) and
            metrics['best']['selected_lineage_targets'] == parent_targets+best['step']*8192, 'D best snapshot/lineage differs')
    coins = metrics['count_block_coins']
    require(set(coins) == {'drawn','kept','dropped'} and all(natural(value,'coin count') >= 0 for value in coins.values()) and
            coins['drawn'] == 12000 and coins['kept']+coins['dropped'] == 12000, 'D count coin accounting differs')
    require(metrics['discarded_count_block_coins'] == dict(drawn=0,kept=0,dropped=0) and
            metrics['count_block_coins_including_discarded'] == coins and
            metrics['cached_target_probability_lookups'] == 12000*8192 and
            metrics['joint_count_kept_target_occurrences'] == (coins['kept']*8192 if objective == 'joint' else 0) and
            metrics['validation_choices_in_retained_history'] == 13, 'D lookup/coin/validation costs differ')
    require([row['step'] for row in metrics['history']] == list(range(400,12001,400)), 'D step logging incomplete')
    last = dict(drawn=0,kept=0,dropped=0)
    for row in metrics['history']:
        current = row['count_block_coins']
        require(row['session'] == 0 and isinstance(row['count_block_kept'],bool) and current['drawn'] == row['step'] and
                current['kept']+current['dropped'] == row['step'] and
                all(0 <= current[key]-last[key] <= 400 for key in current), 'D historical coin trajectory differs')
        last = current
    require(last == coins, 'D final coin history differs')
    return best


def verify_run(root, name, parent, parent_sha, preparation, pins, row):
    directory = root/'runs'/('push150-'+name); objective = name.removeprefix('oof-')
    metrics = read(directory/'metrics.json')
    require(not list(directory.glob('metrics-before-resume-*.json')) and not list(directory.glob('preflight-failure-*.json')),
            'D failures/resume evidence require explicit manual reconciliation')
    best = verify_training_history(metrics,parent['train_tokens'],objective)
    recipe = expected_recipe(parent['config'],objective)
    implementation = parent['implementation']
    sources = graph(root,['push150_oof_train.py',implementation+'.py','push148_muon.py','push150_oof_prepare.py'])
    identity = dict(parent_sha256=parent_sha,implementation=implementation,recipe=recipe,source_sha256=sources,
        data_sha256=a.DATA_SHA,oof_preparation=preparation,prepared_file_sha256=pins,runtime=a.RUNTIME)
    require(metrics['identity'] == identity and metrics['recipe'] == recipe, 'D parent/recipe/source/cache/runtime identity differs')
    for key,value in dict(parent_sha256=parent_sha,parent_train_targets=parent['train_tokens'],
        training_implementation=implementation,inference_implementation=implementation,inference_architecture_unchanged_from_parent=True,
        training_config=recipe['training_config'],validation_config=recipe['validation_config'],seed=211,steps=12000,eval_every=2000,
        batch_size=32,dropout_views_per_sample=1,processed_view_targets_per_step=8192,sampled_label_targets_per_step=8192,
        optimizer='muon',training_precision='bf16',ema_decay=.99,no_statistical_tables_in_training=True,
        source_splits=['train','validation'],test_used=False,prior_test_exposure_disclosed=True,
        source_sha256=sources,data_sha256=a.DATA_SHA,implementation_sha256=sources[implementation+'.py'],
        oof_preparation=preparation,objective=objective,
        optimizer_initialization='Fresh optimizers and EMA from scoring parent; old optimizer states unavailable').items():
        require(metrics[key] == value, 'D training field differs: '+key)
    cost = a.training_accounting(metrics); row.update(cost)
    canonical = directory/'checkpoint.pt'; digest = sha(canonical)
    require(digest == metrics['best']['scoring_checkpoint_sha256'], 'D scoring CP hash differs from canonical field')
    archive = child(directory,metrics['best']['checkpoint_file'])
    require(sha(archive) == metrics['best']['checkpoint_sha256'], 'D best archive hash differs')
    trained, archived = checkpoint(canonical),checkpoint(archive)
    require(set(trained) == SNAPSHOT_KEYS, 'Scoring CP contains undeclared training payload/cache')
    same_tensors(trained['model'],archived['model'])
    require(trained['config'] == archived['config'] == recipe['validation_config'] and
            set(trained['model']) == set(parent['model']), 'D architecture/config/state key set changed')
    require(trained['implementation'] == implementation and trained['parent_sha256'] == parent_sha and
            trained['selected_step'] == best['step'] and trained['weights'] == best['weights'] and
            trained['train_tokens'] == parent['train_tokens']+best['step']*8192 and
            trained['training_recipe'] == recipe and trained['training_config'] == recipe['training_config'] and
            trained['ancestry'] == parent.get('ancestry',{}), 'D selected checkpoint ancestry differs')
    same_tensors(trained['model'],parent['model'],'ngram.')
    state_path = directory/'train_state.pt'; state = bc.raw_checkpoint(state_path)
    require(state['format_version'] == 1 and state['identity'] == identity and state['step'] == 12000 and
            state['log'] == metrics and len(state['optimizers']) == 2 and all(opt['state'] for opt in state['optimizers']),
            'D complete optimizer/raw/EMA state differs')
    keys = {key for key in parent['model'] if not key.startswith('ngram.')}
    require(set(state['raw']) == set(state['ema']) == keys and
            set(state['rng']) == {'python','numpy','torch_cpu','sampler','torch_cuda'} and len(state['rng']['torch_cuda']) == 1,
            'D complete state is missing raw/EMA/RNG or includes training cache tensors')
    tails = metrics['tail_snapshots']
    require([(r['step'],r['weights']) for r in tails] == [(step,label) for step in (10000,12000) for label in ('raw','ema')],
            'D tail checkpoint set incomplete')
    for tail_record in tails:
        path = child(directory,tail_record['path']); require(sha(path) == tail_record['sha256'], 'D tail changed')
        tail = checkpoint(path)
        require(set(tail) == SNAPSHOT_KEYS and tail['parent_sha256'] == parent_sha and tail['training_recipe'] == recipe and
                tail['selected_step'] == tail_record['step'] and tail['weights'] == tail_record['weights'] and
                tail['train_tokens'] == parent['train_tokens']+tail_record['step']*8192, 'D retained tail metadata differs')
        same_tensors(tail['model'],parent['model'],'ngram.')
        if tail_record['step'] == 12000:
            same_tensors(state[tail_record['weights']],{k:v for k,v in tail['model'].items() if not k.startswith('ngram.')})
    row.update(training_metrics_sha256=sha(directory/'metrics.json'),training_checkpoint_sha256=digest,
        archive_checkpoint_sha256=metrics['best']['checkpoint_sha256'],train_state_sha256=sha(state_path),
        selected_step=best['step'],selected_weights=best['weights'],source_sha256=sources,
        count_block_coins=metrics['count_block_coins'],cached_target_probability_lookups=metrics['cached_target_probability_lookups'],
        joint_count_kept_target_occurrences=metrics['joint_count_kept_target_occurrences'],
        new_count_scan_token_occurrences=0,inference_architecture_unchanged=True,training_cache_in_deployment=False)
    bc.verify_calibration(root,directory.with_name(directory.name+'-calibrated'),trained,digest,False,row)
    row.update(accounting_verified=True,phase='D')
    # Only small RNG tensors survive; never retain raw/EMA models in JSON.
    paired = dict(parent_sha256=parent_sha,recipe=recipe,source_sha256=sources,
        count_block_coins=metrics['count_block_coins'],
        trajectory=[dict(step=r['step'],count_block_kept=r['count_block_kept'],count_block_coins=r['count_block_coins']) for r in metrics['history']],
        sampler_state=state['rng']['sampler'].clone(),cuda_rng=[v.clone() for v in state['rng']['torch_cuda']])
    del state,archived
    return paired


def verify_pair(control, joint):
    require(control['parent_sha256'] == joint['parent_sha256'] and control['source_sha256'] == joint['source_sha256'],
            'D control/joint do not share parent and implementation')
    left,right = copy.deepcopy(control['recipe']),copy.deepcopy(joint['recipe'])
    require(left.pop('objective') == 'control' and right.pop('objective') == 'joint', 'D objective labels differ')
    left.pop('objective_description'); right.pop('objective_description')
    require(left == right, 'D paired recipes differ by more than the objective')
    require(control['count_block_coins'] == joint['count_block_coins'] and control['trajectory'] == joint['trajectory'],
            'D paired count-dropout streams differ')
    require(torch.equal(control['sampler_state'],joint['sampler_state']) and
            len(control['cuda_rng']) == len(joint['cuda_rng']) and
            all(torch.equal(x,y) for x,y in zip(control['cuda_rng'],joint['cuda_rng'])),
            'D final sampler/global-dropout RNG states differ')
    return dict(passed=True,shared_parent_sha256=control['parent_sha256'],seed=211,
        paired_steps=12000,sampled_label_occurrences_each=12000*8192,count_block_coins=control['count_block_coins'],
        history_checkpoints=30,final_sampler_states_equal=True,final_cuda_dropout_rng_states_equal=True,
        basis='Same frozen sampler code, exact common recipe/seed/folds, 30 matching coin checkpoints and final sampler/CUDA states; no resampling or data access in selector')


def verify_launch_ledger(root, parent_sha, selection_sha):
    directory = root/'runs/push150_followon_launches'
    plan = read(directory/'plan.json'); summary = read(directory/'summary.json')
    names = ['gate-build','oof-prepare','gate-extract','gate-train','gate-score','suffix-init',
             'suffix-control','suffix-control-calibrate','suffix-joint','suffix-joint-calibrate',
             'oof-control','oof-control-calibrate','oof-joint','oof-joint-calibrate']
    require(not (directory/'failure.json').exists() and summary['status'] == 'completed' and
            summary['test_used'] is False and plan['test_used'] is False, 'Follow-on launch ledger failed or incomplete')
    require(plan['parent_sha256'] == summary['parent_sha256'] == parent_sha and
            plan['phase_a_selection_sha256'] == selection_sha and summary['plan_sha256'] == sha(directory/'plan.json'),
            'Launch ledger parent/plan binding differs')
    require([row['name'] for row in plan['commands']] == names and [row['name'] for row in summary['launches']] == names,
            'Expected all fourteen declared serial launches')
    for name,digest in plan['source_sha256'].items():
        require(sha(child(root,name)) == digest, 'Launch-pinned runtime source changed: '+name)
    plans = {relative(root,path):sha(path) for path in sorted((root/'runs').glob('push150_phase_*_plan.json'))}
    require(plan['plan_sha256'] == plans, 'Launch-pinned phase plans differ')
    unit_path = child(root,relative(root,plan['unit_evidence_path']))
    require(sha(unit_path) == plan['unit_evidence_sha256'], 'Pre-launch unit evidence changed')
    unit = read(unit_path)
    require(unit['successful'] is True and not unit['errors'] and not unit['failures'] and unit['tests_run'] > 0,
            'Pre-launch unit tests did not pass')
    for index,(name,command,record) in enumerate(zip(names,plan['commands'],summary['launches'])):
        started = read(directory/(name+'.started.json')); process = read(directory/(name+'.process.json'))
        require(started['index'] == index and started['name'] == name and started['command'] == command['command'] and
                started['test_used'] is False and started['plan_sha256'] == sha(directory/'plan.json'), 'Launch start record differs')
        require(process == record and process['index'] == index and process['returncode'] == 0 and
                process['test_used'] is False and process['failure_cost_note'] is None and
                finite(process['seconds_total'],'launch seconds') > 0, 'Launch exit/status record differs')
        require(sha(directory/(name+'.output.log')) == process['output_log_sha256'], 'Preserved launch output changed')
    bc.near(summary['seconds_total'],sum(row['seconds_total'] for row in summary['launches']), 'Inclusive launch duration')
    return dict(plan_sha256=sha(directory/'plan.json'),summary_sha256=sha(directory/'summary.json'),
        launches=summary['launches'],seconds_total_inclusive_not_added_to_child_costs=summary['seconds_total'],
        all_fourteen_launches_completed=True,pre_launch_unit_evidence_path=relative(root,unit_path))


def observed_d(root):
    result = dict(preparation={},runs=[])
    directory = root/'runs/push150-oof-data'
    for path in sorted(directory.glob('*.json')):
        try:
            result['preparation'][path.name] = dict(sha256=sha(path),record=read(path))
        except ERRORS as error:
            result['preparation'][path.name] = dict(observation_error=str(error))
    for name in RUN_NAMES:
        directory = root/'runs'/('push150-'+name)
        row = dict(name=name,phase='D',selection_eligible=False,accounting_verified=False,records={})
        result['runs'].append(row)
        for path in sorted(directory.glob('*.json')):
            try:
                row['records'][path.name] = dict(sha256=sha(path),record=read(path))
            except ERRORS as error:
                row['records'][path.name] = dict(observation_error=str(error))
        metrics = row['records'].get('metrics.json',{}).get('record')
        if metrics:
            try:
                row.update(a.training_accounting(metrics))
            except ERRORS as error:
                row['accounting_error'] = str(error)
        calibrated = directory.with_name(directory.name+'-calibrated')
        row['calibration_records'] = {}
        for path in sorted(calibrated.glob('*.json')):
            try:
                row['calibration_records'][path.name] = dict(sha256=sha(path),record=read(path))
            except ERRORS as error:
                row['calibration_records'][path.name] = dict(observation_error=str(error))
    return result


def evidence_manifest(root, unit_paths, candidates):
    """Explicit development scope: never enumerate or open final-test logs/text."""
    paths = set(root.glob('push150*.py')) | set((root/'tests').glob('test_push150*.py'))
    paths |= {root/name for name in FROZEN_SOURCES}
    paths |= set((root/'runs').glob('push150_phase_*_plan.json'))
    for name in EVIDENCE_DIRECTORIES:
        paths |= set((root/'runs'/name).rglob('*.json'))
    paths |= set((root/'runs/push150_followon_launches').glob('*.output.log'))
    for name in ('push150_phase_a_selection.json','push150_phase_bc_selection.json','push150_preserved_packages.json',
                 *HISTORICAL_METADATA_EVIDENCE):
        path = root/'runs'/name
        if path.exists():
            paths.add(path)
    for name in ('measure_validation.py','official_resource_run.py','baseline.py',
                 'runs/push149-muon-low-calibrated/metrics.json','runs/push149-muon-low-calibrated/checkpoint.pt'):
        path = root/name
        if path.exists():
            paths.add(path)
    for unit in unit_paths:
        paths.add(child(root,relative(root,unit)))
    for row in candidates:
        if row.get('selection_eligible'):
            paths.add(child(root/'runs',row['selected_run'])/'checkpoint.pt')
    array = root/'runs/push150-oof-data/train_oof_probabilities.npy'
    if array.exists():
        paths.add(array)
    require(all('push150-final-test' not in path.parts and not path.name.startswith('wikitext_') for path in paths),
            'Forbidden test/raw corpus artifact entered evidence manifest')
    result = {relative(root,path):sha(path) for path in sorted(paths)}
    for name,digest in FROZEN_SOURCES.items():
        require(result[name] == digest, 'Frozen helper changed during evidence collection: '+name)
    for candidate in candidates:
        if candidate.get('selection_eligible'):
            name = 'runs/'+candidate['selected_run']+'/checkpoint.pt'
            require(result[name] == candidate['checkpoint_sha256'], 'Candidate changed during evidence collection: '+name)
    return result


def select(candidates, original149, errors):
    if errors:
        return dict(status='refused',winner=None,all_declared_phases_accounted=False,passes_predeclared_improvement_gate=False)
    eligible = [row for row in candidates if row.get('selection_eligible')]
    require(bool(eligible), 'No eligible completed candidate')
    winner = min(eligible,key=lambda row:finite(row['validation_bpb_gpu_fp32'],'candidate validation BPB'))
    improvement = original149-winner['validation_bpb_gpu_fp32']
    return dict(status='validation_choice_complete',winner=copy.deepcopy(winner),all_declared_phases_accounted=True,
        improvement_vs_original149=improvement,passes_predeclared_improvement_gate=improvement >= .0005)


def collect_report(root=ROOT, unit_paths=None):
    root = Path(root)
    unit_paths = unit_paths or [root/'runs/push150_unit_tests_all_phases.json']
    result = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),status='refused',selection_split='validation',
        selection_precision='fp32',selection_device='cuda:0',test_used=False,test_used_for_selection=False,
        no_raw_text_loaded_or_hashed=True,prior_test_exposure_disclosed=True,prior_exposure_history_is_exhaustive=False,
        all_declared_phases_accounted=False,passes_predeclared_improvement_gate=False,minimum_validation_improvement=.0005,
        selection_integrity_errors=[],winner=None,candidates=[],evidence_sha256={},
        final_resource_audit_pending=True,freeze_authorized_by_this_report=False,test_authorized_by_this_report=False,
        report_script_sha256=sha(__file__),required_phase_d_sources=D_SOURCES,observed_phase_d=observed_d(root))

    def attempt(name,function):
        try:
            return function()
        except ERRORS as error:
            result['selection_integrity_errors'].append(dict(name=name,error=str(error)))
            return None

    def global_checks():
        require(not (root/'runs/push150-final-test').exists(), 'A final-test ledger exists; development selection must not rerun after testing')
        for name,digest in FROZEN_SOURCES.items():
            require(sha(root/name) == digest, 'Frozen source changed: '+name)
        require(sha(root/'runs/push150_phase_d_oof_plan.json') == D_PLAN_SHA, 'Declared D plan changed')
        result['phase_d_plan'] = dict(sha256=D_PLAN_SHA,record=read(root/'runs/push150_phase_d_oof_plan.json'))
        result['unit_tests'],result['unit_test_execution_history'] = verify_latest_units(root,unit_paths)

    attempt('sources_plan_complete_unit_run',global_checks)
    phase_bc = result.setdefault('phase_bc',{})
    accepted = attempt('phase_bc',lambda:verify_bc(root,phase_bc))
    d_rows = []
    for observed in result['observed_phase_d']['runs']:
        row = {key:copy.deepcopy(value) for key,value in observed.items() if key not in ('records','calibration_records')}
        d_rows.append(row)
    if accepted is not None:
        saved,parent,parent_path,selected_a = accepted
        result['candidates'] = [copy.deepcopy(row) for row in saved['candidates'] if row['selection_eligible']]
        result.update(original149_validation_bpb_gpu_fp32=saved['original149_validation_bpb_gpu_fp32'],
            phase_a_parent_checkpoint_sha256=selected_a['checkpoint_sha256'],phase_a_selection_sha256=saved['phase_a_selection_sha256'])
        result['followon_launches'] = attempt('followon_launches',lambda:verify_launch_ledger(root,
            selected_a['checkpoint_sha256'],saved['phase_a_selection_sha256']))
        prep_row = result.setdefault('phase_d_preparation',{})
        prepared = attempt('phase_d_preparation',lambda:verify_prepared(root,parent,parent_path,
            selected_a['checkpoint_sha256'],saved['phase_a_selection_sha256'],prep_row))
        if prepared is not None:
            provenance,pins = prepared
            paired = []
            for name,row in zip(RUN_NAMES,d_rows):
                paired.append(attempt(name,lambda name=name,row=row:verify_run(root,name,parent,
                    selected_a['checkpoint_sha256'],provenance,pins,row)))
            if all(item is not None for item in paired):
                result['phase_d_matched_comparison'] = attempt('phase_d_matched_comparison',lambda:verify_pair(*paired))
    else:
        result['selection_integrity_errors'].append(dict(name='phase_d_parent',error='No verified A parent until BC evidence completes'))
    result['candidates'] += d_rows
    all_unit_paths = list(unit_paths)
    if phase_bc.get('record'):
        all_unit_paths += [item['path'] for item in phase_bc['record']['units']['evidence']]
    if result.get('followon_launches'):
        all_unit_paths.append(result['followon_launches']['pre_launch_unit_evidence_path'])
    manifest = attempt('evidence_manifest',lambda:evidence_manifest(root,all_unit_paths,result['candidates']))
    if manifest is not None:
        result['evidence_sha256'] = manifest
        result['historical_metadata_repair_evidence'] = [
            dict(path='runs/'+name,sha256=manifest['runs/'+name],role=role,
                 validated_as_current_runtime_or_unit_evidence=False)
            for name,role in HISTORICAL_METADATA_EVIDENCE.items() if 'runs/'+name in manifest]
    result.update(select(result['candidates'],result.get('original149_validation_bpb_gpu_fp32',0.),result['selection_integrity_errors']))
    bc_cost = phase_bc.get('record',{}).get('cost_summary',{})
    d_cost = sum(row.get('new_search_sampled_label_occurrences',0) for row in d_rows)
    metadata = result['observed_phase_d']['preparation'].get('metadata.json',{}).get('record',{})
    failure = result['observed_phase_d']['preparation'].get('prepare_failure.json',{}).get('record',{})
    result['cost_summary'] = dict(accounting_verified=not result['selection_integrity_errors'],
        phase_abc_recorded_new_gradient_label_occurrences=bc_cost.get('all_recorded_new_gradient_label_occurrences',0),
        phase_d_new_neural_view_targets=d_cost,phase_d_new_label_occurrences=d_cost,
        all_recorded_new_gradient_label_occurrences=bc_cost.get('all_recorded_new_gradient_label_occurrences',0)+d_cost,
        phase_d_preparation_costs=metadata.get('costs'),phase_d_failed_preparation_progress=failure.get('progress'),
        shared_B_count_scan_token_occurrences_added_again_in_D=0,
        shared_original149_ancestry_targets=a.PARENT_TARGETS,selected_lineage_is_not_whole_search_work=True,
        occurrence_counts_are_not_unique_tokens=True,prior_round_training_costs_not_exhaustively_reconstructed=True,
        missing_or_partial_work_remains_unquantified=bool(result['selection_integrity_errors']))
    result['next_action'] = 'Run actual selected-candidate CPU/asset/RAM/causality audit. Any resource exclusion needs an explicit separate pretest record; this script performs no fallback, freeze, submission or test.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-evidence',action='append',type=Path,
                        help='May repeat; LAST must cover every current push150 source and test')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/push150_selection.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve previous selection; choose another exclusive output path')
    torch.set_num_threads(2)
    result = collect_report(ROOT,args.unit_evidence)
    a.write_report(args.output,result)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    if result['status'] == 'refused':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
