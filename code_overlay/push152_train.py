"""Fixed-plan push152 Phase-B fresh-optimizer, train/validation-only branches.

No test split is loaded or hashed. The deployed parent architecture, statistics,
and calibration remain unchanged within each branch. Full optimizer/RNG states
are saved, while every scored raw/EMA snapshot has an exclusive archive name.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from common import ROOT, PROTOCOL, autocast, make_model, setup, sha
from evaluate import score
from push147_train import training_loss
from push148_train import build_optimizers
from push150_train import (training_config, schedule_factor, cpu_clone, snapshot,
    pack_train_state, restore_train_state, atomic_torch_save, atomic_json_save)


PLAN_PATH = 'runs/push152_plan.json'
AUTHORIZATION_PATH = 'runs/push152-expanded-preflight/authorization.json'
PLAN_SHA = 'd15cd6276ace1894a37db7c005453c61acd2e5acb7e934cb6790f3860aa06b51'
PARENT_SHA = 'c72cffff6b4152492518473f46a274b53272f59dacb80072cd9e1497d2e0bdf5'
PARENT_PATH = 'runs/push151-combination/checkpoint.pt'
IMPLEMENTATION = 'push150_expand'
LINEAGE = 638976000
BATCH_SIZE, CONTEXT, TARGETS_PER_STEP = 32, 256, 8192
BRANCHES = ('control','auxiliary','expanded')
COMMON = dict(steps=20000,eval_every=4000,seed=193,optimizer='muon',schedule='cosine',
    warmup=200,decay_steps=4000,lr_floor=.1,backbone_lr=7.5e-6,head_lr=7.5e-5,
    scalar_lr=2.5e-4,muon_lr=5e-4,batch_size=32,context=256,dropout=.2,
    training_temperature=1.15,validation_temperature=1.3,validation_mkn_weight=.15,
    ema_decay=.99,gradient_clip=1.)
DATA_SHA = {'data/tokenizer.json':'020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e',
    'data/manifest.json':'f4790bb599d9fcd04e48263e685335e2f62d9bca0324b08061a10c38984f4fe4',
    'data/wikitext_train.txt':'c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8',
    'data/wikitext_validation.txt':'714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def child(root, name):
    name = str(name).replace('\\','/')
    require(name and ':' not in name and not name.startswith('/') and
            all(part not in ('','.','..') for part in name.split('/')), 'Unsafe relative path')
    result = (Path(root)/name).resolve()
    require(result.is_relative_to(Path(root).resolve()), 'Path escaped training root')
    return result


def metadata(path):
    require(Path(path).suffix == '.json', 'JSON metadata only')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, record):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
        stream.write('\n')


def save_snapshot_new(path, checkpoint):
    """An incomplete archive is failure evidence, never silently overwritten."""
    with Path(path).open('xb') as stream:
        torch.save(checkpoint,stream)


def source_digests(implementation=IMPLEMENTATION, root=ROOT):
    """Recursively pin executable local Python, including reused frozen helpers."""
    root = Path(root).resolve()
    pending = [root/'push152_train.py',root/(implementation+'.py'),root/'push148_muon.py']
    result = {}
    while pending:
        path = pending.pop().resolve()
        name = path.relative_to(root).as_posix()
        if name in result:
            continue
        result[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            imports = ([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                       [entry.name for entry in node.names] if isinstance(node,ast.Import) else [])
            for module in imports:
                local = root/(module.replace('.','/')+'.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(result.items()))


def branch_spec(plan, branch):
    """Strict Phase-B schema, finalized against the pre-execution plan."""
    require(branch in BRANCHES, 'Unknown Phase-B branch')
    phase = plan['phase_b']
    shared = plan['shared']
    actual = {key:phase[key] for key in ('steps','eval_every','seed','schedule','warmup','decay_steps','lr_floor','ema_decay')}
    actual.update({key:shared[key] for key in ('optimizer','batch_size','context','dropout',
        'training_temperature','validation_temperature','gradient_clip')})
    actual.update(backbone_lr=phase['learning_rates']['backbone'],head_lr=phase['learning_rates']['pointer'],
        scalar_lr=phase['learning_rates']['scalar'],muon_lr=phase['learning_rates']['muon'],validation_mkn_weight=shared['mkn_weight'])
    require(actual == COMMON and shared['targets_per_step'] == 8192 and shared['training_split'] == 'train' and
            shared['selection_split'] == 'validation' and shared['training_precision'] == 'bf16' and
            shared['validation_precision'] == 'fp32' and shared['validation_device'] == 'cuda' and
            shared['no_calibration_grid'] is True and phase['new_targets_per_completed_branch'] == 163840000,
            'Fixed Phase-B training recipe changed')
    require(plan['parent']['path'] == PARENT_PATH and plan['parent']['sha256'] == PARENT_SHA and
            plan['parent']['selected_lineage_targets'] == LINEAGE and
            plan['external_training_data_or_weights'] is False and plan['evaluation_network_access'] is False,
            'Plan parent/source constraint changed')
    require(set(phase['variants']) == set(BRANCHES), 'Changed branch menu')
    spec = dict(copy.deepcopy(phase['variants'][branch]),name=branch)
    expected_alpha = .1 if branch == 'auxiliary' else 0.
    require(spec['auxiliary_weight'] == expected_alpha and
            spec['run_dir'] == 'runs/push152-'+branch, 'Branch objective/run path differs')
    if branch != 'expanded':
        require(spec['parent'] == PARENT_PATH, 'Control/auxiliary parent differs')
        spec['parent_sha256'] = PARENT_SHA
    else:
        require(spec['parent'] == 'runs/push152-expanded-init/checkpoint.pt', 'Expanded initializer path differs')
        spec['resource_authorization'] = plan['expansion']['authorization']
        require(spec['resource_authorization'] == 'runs/push152-expanded-preflight/authorization.json', 'Authorization path changed')
    return copy.deepcopy(spec)


def check_expanded_timing(timing, checkpoint_sha):
    """Three actual fresh full-validation pairs; no extra worst-case gate."""
    from push150_final_test import BASELINE_SHA, SCORER_SHA
    expected = dict(test_used=False,split='validation',device='cpu',precision='fp32',threads=4,
                    affinity_mask=15,candidate_sha256=checkpoint_sha,baseline_sha256=BASELINE_SHA,evaluator_sha256=SCORER_SHA)
    require(all(timing.get(key) == value for key,value in expected.items()), 'Expanded timing identity differs')
    rows = timing['measurements']
    require([(row['repeat'],row['name']) for row in rows] ==
        [(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')],
        'Expanded training needs all three alternating fresh timing pairs')
    for row in rows:
        require(row['targets'] == 376599 and row['utf8_bytes'] == 1148007 and
                type(row['seconds']) in (float,int) and math.isfinite(row['seconds']) and row['seconds'] > 0 and
                type(row['peak_process_working_set_bytes']) is int and row['peak_process_working_set_bytes'] > 0,
                'Expanded timing measurement is incomplete/nonfinite')
    baseline = [row['seconds'] for row in rows if row['name'] == 'baseline']
    candidate = [row['seconds'] for row in rows if row['name'] == 'candidate']
    ratio = statistics.median(candidate)/statistics.median(baseline)
    require(abs(ratio-timing['cpu_time_ratio']) < 1e-12 and ratio <= 5,
            'Expanded training not authorized: median CPU ratio exceeds five')
    require(max(row['peak_process_working_set_bytes'] for row in rows if row['name'] == 'candidate') <= 4*2**30,
            'Expanded timing RAM exceeds limit')
    return ratio


def verify_authorization(root, spec, plan_sha):
    # The plan requires a completed expansion feasibility decision before ANY
    # GPU branch. A failed expansion does not forbid control/auxiliary work.
    path = child(root,AUTHORIZATION_PATH)
    record = metadata(path)
    require(record.get('status') == 'completed' and record.get('plan_sha256') == plan_sha and
            record.get('test_used') is False and type(record.get('training_authorized')) is bool and
            record.get('all_limits_pass') is record['training_authorized'] and
            record.get('no_layout_fallback') is True and
            record.get('script_sha256') == sha(child(root,'push152_preflight.py')),
            'Every GPU branch requires the completed fixed expansion preflight decision')
    if spec['name'] != 'expanded':
        return {AUTHORIZATION_PATH:sha(path),'push152_preflight.py':record['script_sha256']}, spec['parent_sha256']
    require(spec['resource_authorization'] == AUTHORIZATION_PATH, 'Expanded authorization path differs')
    require(record.get('status') == 'completed' and record.get('training_authorized') is True and
            record.get('all_limits_pass') is True and record.get('no_layout_fallback') is True and
            record.get('test_used') is False and record.get('plan_sha256') == plan_sha and
            record.get('script_sha256') == sha(child(root,'push152_preflight.py')),
            'Expanded branch lacks explicit root preflight authorization')
    checkpoint_sha = record['initializer_checkpoint_sha256']
    require(sha(child(root,spec['parent'])) == checkpoint_sha, 'Expanded authorized checkpoint changed')
    timing_path = child(root,record['timing_summary'])
    require(sha(timing_path) == record['timing_summary_sha256'], 'Expanded preflight timing changed')
    timing = metadata(timing_path)
    ratio = check_expanded_timing(timing,checkpoint_sha)
    audit_path = child(root,record['audit'])
    require(sha(audit_path) == record['audit_sha256'], 'Expanded initializer audit changed')
    audit = metadata(audit_path)
    require(audit.get('status') == 'completed' and audit.get('plan_sha256') == plan_sha and
            audit.get('checkpoint_sha256') == checkpoint_sha and audit.get('parent_sha256') == PARENT_SHA and
            audit.get('script_sha256') == record['script_sha256'] and audit.get('test_used') is False and
            audit.get('any_corpus_loaded') is False and audit.get('normalization_finite_pass') is True and
            audit.get('parent_prediction_equivalence_pass') is True and audit.get('untouched_statistical_buffers') == 21 and
            audit.get('limits_except_timing_pass') is True and audit.get('asset_limit_pass') is True and
            audit.get('ram_limit_pass') is True and 0 < audit['peak_process_working_set_bytes'] <= 4*2**30 and
            audit['diagnostic_summary'].get('predeclared_supplemental_diagnostic_pass') is True and
            0 < audit['inference_assets_bytes'] <= 64*2**20,
            'Expanded inference asset preflight failed')
    peak = max(row['peak_process_working_set_bytes'] for row in timing['measurements'])
    require(record['cpu_ratio'] == ratio and record['peak_process_working_set_bytes'] == peak <= 4*2**30 and
            record['inference_assets_bytes'] == audit['inference_assets_bytes'], 'Authorization resource quantities differ')
    pins = {str(path.relative_to(Path(root).resolve()).as_posix()):sha(path),
            str(timing_path.relative_to(Path(root).resolve()).as_posix()):sha(timing_path),
            str(audit_path.relative_to(Path(root).resolve()).as_posix()):sha(audit_path)}
    pins['push152_preflight.py'] = record['script_sha256']
    init_path = child(root,spec['parent']).parent/'initialization.json'
    require(sha(init_path) == audit['initialization_sha256'], 'Initializer provenance changed')
    initialization = metadata(init_path)
    require(initialization['checkpoint_sha256'] == checkpoint_sha and initialization['plan_sha256'] == plan_sha,
            'Initializer identity differs')
    pins[init_path.relative_to(Path(root).resolve()).as_posix()] = sha(init_path)
    for row in timing['measurements']:
        raw_path = timing_path.parent/(row['name']+'-'+str(row['repeat'])+'.json')
        raw = metadata(raw_path)
        expected_sha = checkpoint_sha if row['name'] == 'candidate' else timing['baseline_sha256']
        require(raw.get('checkpoint_sha256') == expected_sha and raw.get('threads') == 4 and
                raw.get('cpu_affinity_mask') == 15 and raw.get('split') == 'validation' and len(raw['repeats']) == 1 and
                raw.get('evaluator_sha256') == timing['evaluator_sha256'], 'Original timing identity differs')
        require(row == dict(name=row['name'],repeat=row['repeat'],**raw['repeats'][0],
                peak_process_working_set_bytes=raw['peak_process_working_set_bytes']), 'Original timing measurement differs')
        require(math.isfinite(row['bpb']) and row['bpb'] > 0 and
                abs(row['bpb']-row['nll_nats']/math.log(2)/1148007) < 1e-10, 'Timing BPB denominator differs')
        if row['name'] == 'candidate':
            plan = metadata(child(root,PLAN_PATH))
            require(abs(row['bpb']-plan['parent']['validation_bpb_cpu_fp32']) <= 1e-5, 'Expanded parent validation parity failed')
        pins[raw_path.relative_to(Path(root).resolve()).as_posix()] = sha(raw_path)
    total = 0
    for name,entry in audit['assets'].items():
        asset = child(root,spec['parent']) if name == 'checkpoint.pt' else child(root,name)
        require(sha(asset) == entry['sha256'] and asset.stat().st_size == entry['bytes'], 'Preflight inference asset changed')
        pins[asset.relative_to(Path(root).resolve()).as_posix()] = entry['sha256']
        total += entry['bytes']
    require(total == audit['inference_assets_bytes'], 'Preflight inference asset sum differs')
    for name,digest in record.get('evidence_sha256',{}).items():
        require(sha(child(root,name)) == digest, 'Expanded resource evidence changed: '+name)
        pins[name] = digest
    return pins, checkpoint_sha


def recipe_for(parent_config, branch, auxiliary_weight):
    require(branch in BRANCHES and auxiliary_weight == (.1 if branch == 'auxiliary' else 0.), 'Unexpected auxiliary objective')
    train = training_config(parent_config,COMMON['dropout'])
    evaluation = copy.deepcopy(parent_config)
    evaluation.update(temperature=1.3,mkn_weight=.15,dropout=.2)
    return dict(**copy.deepcopy(COMMON),branch=branch,auxiliary_weight=auxiliary_weight,
        dropout_views_per_sample=1,training_precision='bf16',training_config=train,validation_config=evaluation,
        adamw_betas=[.9,.95],backbone_weight_decay=.1,pointer_weight_decay=.01,scalar_weight_decay=0.,
        muon_momentum=.95,muon_ns_steps=5,validation_choices=11,all_validation_snapshots_saved=True,
        objective='(1-alpha)*NLL(neural+causal-pointer) + alpha*NLL(neural); no statistical tables in loss')


def validate_parent(parent, branch):
    require(parent.get('protocol') == PROTOCOL and parent.get('implementation') == IMPLEMENTATION and
            parent.get('train_tokens') == LINEAGE, 'Parent protocol/implementation/shared ancestry differs')
    config = parent['config']
    require(all(config.get(key) == value for key,value in dict(vocab=2048,width=320,heads=8,depth=6,context=256,
        ffn_hidden=864,temperature=1.3,mkn_order=5,mkn_weight=.15,discount_scale=.7).items()), 'Parent architecture/calibration changed')
    ranks = dict(qkv=160,proj=0,gate=176 if branch == 'expanded' else 144,
                 value=176 if branch == 'expanded' else 144,out=176 if branch == 'expanded' else 144)
    require(config.get('push150_ranks') == ranks, 'Unexpected branch projection capacity')
    require(sum(name.startswith('ngram.') for name in parent['model']) == 21, 'Need all current statistical buffers')
    require(torch.equal(parent['model']['head.weight'],parent['model']['token.weight']), 'Parent embedding/head not tied by value')


def validate_statistics(parent, reference):
    candidate = {name:value for name,value in parent['model'].items() if name.startswith('ngram.')}
    fixed = {name:value for name,value in reference['model'].items() if name.startswith('ngram.')}
    require(len(candidate) == 21 and set(candidate) == set(fixed) and all(
        value.dtype == fixed[name].dtype and value.shape == fixed[name].shape and torch.equal(value,fixed[name])
        for name,value in candidate.items()), 'All21 statistical buffers must match fixed push151 parent bitwise')


def sample_starts(token_count, sampler, batch_size=BATCH_SIZE):
    require(type(token_count) is int and token_count > CONTEXT, 'Training split too short')
    # randint high is exclusive; len-257 is the last valid 257-token start.
    return torch.randint(token_count-CONTEXT,(batch_size,),generator=sampler)


def branch_loss(model, ids, targets, auxiliary_weight):
    require(auxiliary_weight in (0.,.1), 'Only the two predeclared auxiliary weights are allowed')
    require(not hasattr(model,'ngram') and not any(name.startswith('ngram.') for name in model.state_dict()),
            'Statistics must not participate in training')
    return training_loss(model,ids,targets,auxiliary_weight)


def target_accounting(completed, discarded, auxiliary_weight):
    require(type(completed) is int and type(discarded) is int and completed >= 0 and discarded >= 0 and
            auxiliary_weight in (0.,.1), 'Invalid work accounting')
    targets = (completed+discarded)*TARGETS_PER_STEP
    return dict(incremental_search_targets=targets,incremental_sampled_label_targets=targets,
        lineage_new_targets=completed*TARGETS_PER_STEP,mixed_nll_target_exposures=targets,
        auxiliary_neural_nll_target_exposures=targets if auxiliary_weight else 0)


def make_snapshot(model, parent, eval_config, parent_sha, recipe, step, weights, plan_sha):
    require(step in range(0,recipe['steps']+1,recipe['eval_every']) and
            weights in (('raw',) if step == 0 else ('raw','ema')), 'Undeclared validation snapshot')
    result = snapshot(model,parent,eval_config,parent_sha,recipe,step,weights)
    result.update(push152_plan_sha256=plan_sha,push152_branch=recipe['branch'],
                  auxiliary_neural_weight=recipe['auxiliary_weight'])
    return result


def load_allowed_data(root):
    hashes = {}
    for name,expected in DATA_SHA.items():
        require(sha(child(root,name)) == expected, 'Fixed train/validation/tokenizer asset changed: '+name)
        hashes[name] = expected
    manifest = metadata(child(root,'data/manifest.json'))
    for filename in ('tokenizer.json','wikitext_train.txt','wikitext_validation.txt'):
        require(manifest['sha256'][filename] == DATA_SHA['data/'+filename], 'Manifest identity differs')
    tokenizer = Tokenizer.from_file(str(child(root,'data/tokenizer.json')))
    data = {}
    for split in ('train','validation'):
        raw = child(root,'data/wikitext_'+split+'.txt').read_bytes()
        data[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids),len(raw))
    require(len(data['train'][0]) == 3613343 and len(data['validation'][0]) == 376600 and
            data['validation'][1] == 1148007, 'Fixed tokenization/counts differ')
    return data, hashes


def run(args, root=ROOT):
    root = Path(root).resolve()
    plan_path = args.plan.resolve()
    require(plan_path == child(root,PLAN_PATH) and sha(plan_path) == PLAN_SHA, 'Frozen push152 plan changed')
    plan = metadata(plan_path)
    spec = branch_spec(plan,args.branch)
    authorization, parent_sha = verify_authorization(root,spec,PLAN_SHA)
    parent_path, directory = child(root,spec['parent']),child(root,spec['run_dir'])
    require(sha(parent_path) == parent_sha, 'Declared parent changed')
    state_path = directory/'train_state.pt'
    if args.resume:
        require(args.resume.resolve() == state_path and state_path.is_file(), 'Resume only the same complete training state')
    else:
        directory.mkdir(parents=True,exist_ok=False)
    session = 0
    completed = 0
    log = None
    metrics_mutation_allowed = not bool(args.resume)
    wall_start = time.perf_counter()
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    write_new(directory/('initialization-resume-'+attempt_id+'.json'
        if args.resume else 'started.json'),dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        branch=args.branch,plan_sha256=PLAN_SHA,parent_sha256=parent_sha,script_sha256=sha(root/'push152_train.py'),
        test_used=False,resume_requested=bool(args.resume)))
    try:
        device, precision = setup('cuda','bf16',4)
        random.seed(COMMON['seed']); np.random.seed(COMMON['seed']); torch.manual_seed(COMMON['seed'])
        parent = torch.load(parent_path,map_location='cpu',weights_only=True)
        validate_parent(parent,args.branch)
        if args.branch == 'expanded':
            require(sha(child(root,PARENT_PATH)) == PARENT_SHA, 'Fixed statistical reference parent changed')
            reference_parent = torch.load(child(root,PARENT_PATH),map_location='cpu',weights_only=True)
            validate_statistics(parent,reference_parent)
            del reference_parent
        recipe = recipe_for(parent['config'],args.branch,spec['auxiliary_weight'])
        model,implementation_sha = make_model(IMPLEMENTATION,recipe['training_config'],device)
        model.load_state_dict({name:value for name,value in parent['model'].items() if not name.startswith('ngram.')},strict=True)
        require(not hasattr(model,'ngram') and not any(name.startswith('ngram.') for name in model.state_dict()),
                'Training model unexpectedly contains statistics')
        averaged = copy.deepcopy(model).eval().requires_grad_(False)
        evaluator,_ = make_model(IMPLEMENTATION,recipe['validation_config'],device)
        evaluator.eval().requires_grad_(False)
        optimizers,muon_names = build_optimizers(model,'muon',recipe['backbone_lr'],recipe['head_lr'],recipe['scalar_lr'],recipe['muon_lr'])
        sampler = torch.Generator().manual_seed(recipe['seed'])
        data,data_hashes = load_allowed_data(root)
        tokens = data['train'][0].to(device)
        # Model constructors differ between capacity branches; do not let their
        # initialization draws alter training dropout's initial RNG state.
        random.seed(recipe['seed']); np.random.seed(recipe['seed']); torch.manual_seed(recipe['seed'])
        identity = dict(plan_sha256=PLAN_SHA,branch=args.branch,parent_sha256=parent_sha,
            statistical_reference_sha256=PARENT_SHA,
            implementation=IMPLEMENTATION,recipe=copy.deepcopy(recipe),source_sha256=source_digests(IMPLEMENTATION,root),
            data_sha256=data_hashes,resource_authorization_sha256=authorization,
            runtime=dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),
                cuda=torch.version.cuda,device=str(device),device_name=torch.cuda.get_device_name(device),
                cuda_device_count=torch.cuda.device_count(),cpu_threads=torch.get_num_threads()))
        log = dict(status='running',identity=identity,branch=args.branch,plan_sha256=PLAN_SHA,parent_sha256=parent_sha,
            parent_train_targets=parent['train_tokens'],training_implementation=IMPLEMENTATION,inference_implementation=IMPLEMENTATION,
            inference_architecture_unchanged_from_parent=True,recipe=copy.deepcopy(recipe),
            training_config=recipe['training_config'],validation_config=recipe['validation_config'],
            optimizer_initialization='Fresh Muon and AdamW plus EMA from declared scoring parent; no old optimizer state continued',
            parameters=sum(parameter.numel() for parameter in model.parameters()),muon_parameter_names=muon_names,
            source_sha256=identity['source_sha256'],data_sha256=data_hashes,implementation_sha256=implementation_sha,
            source_splits=['train','validation'],test_used=False,prior_test_exposure_disclosed=True,
            no_statistical_tables_in_training=True,processed_view_targets_per_step=8192,sampled_label_targets_per_step=8192,
            training_precision=precision,ancestry=copy.deepcopy(parent.get('ancestry',{})),history=[],validation_history=[],
            validation_snapshots=[],discarded_validation_history=[],resume_history=[],discarded_completed_steps=0,completed_steps=0,
            seconds=0.,cost_note='Sampled label/view occurrences; shared parent counted once. Partial failed optimizer work is separately disclosed.')
        offset_seconds = 0.
        if args.resume:
            saved = torch.load(state_path,map_location='cpu',weights_only=True)
            completed,log = restore_train_state(saved,model,averaged,optimizers,sampler,identity,cuda=True)
            disk = metadata(directory/'metrics.json')
            require(disk.get('identity') == identity, 'On-disk resume recipe/source/data identity differs')
            require(not (completed == recipe['steps'] and disk.get('status') == 'completed'), 'Completed run cannot be extended/rescored')
            session = max(len(log['resume_history']),len(disk.get('resume_history',[])))+1
            archive = directory/f'metrics-before-resume-{session:03d}.json'
            write_new(archive,disk)
            extra = max(0,disk['completed_steps']-completed)
            log['discarded_completed_steps'] = max(log.get('discarded_completed_steps',0),disk.get('discarded_completed_steps',0))+extra
            discarded = copy.deepcopy(disk.get('discarded_validation_history',[]))
            discarded += copy.deepcopy(disk['validation_history'][len(log['validation_history']):])
            log['discarded_validation_history'] = discarded
            log['resume_history'] = copy.deepcopy(disk.get('resume_history',[]))
            log['resume_history'].append(dict(session=session,resumed_from_step=completed,discarded_completed_steps=extra,
                discarded_completed_validation_calls=len(discarded),prior_metrics=archive.name,prior_metrics_sha256=sha(archive)))
            offset_seconds = max(disk.get('seconds',0.),log.get('seconds',0.))
            best = directory/log['best']['checkpoint_file']
            require(sha(best) == log['best']['checkpoint_sha256'], 'Restored best validation archive changed')
            atomic_torch_save(torch.load(best,map_location='cpu',weights_only=True),directory/'checkpoint.pt')
            log['best']['scoring_checkpoint_sha256'] = sha(directory/'checkpoint.pt')
            log['status'] = 'running'
            metrics_mutation_allowed = True
        started = time.perf_counter()

        def refresh():
            log.update(completed_steps=completed,**target_accounting(completed,log['discarded_completed_steps'],recipe['auxiliary_weight']),
                completed_validation_calls=len(log['validation_history'])+len(log['discarded_validation_history']),
                seconds=offset_seconds+time.perf_counter()-started,
                peak_cuda_allocated_bytes=max(log.get('peak_cuda_allocated_bytes',0),torch.cuda.max_memory_allocated()))

        def check_pins():
            require(sha(plan_path) == PLAN_SHA and sha(parent_path) == parent_sha and
                    sha(child(root,PARENT_PATH)) == PARENT_SHA and
                    source_digests(IMPLEMENTATION,root) == identity['source_sha256'], 'Plan/source/parent changed during run')
            for name,digest in authorization.items():
                require(sha(child(root,name)) == digest, 'Resource authorization changed')
            for name,digest in data_hashes.items():
                require(sha(child(root,name)) == digest, 'Pinned train/validation/tokenizer source changed')

        def save_state():
            check_pins(); refresh()
            atomic_torch_save(pack_train_state(model,averaged,optimizers,sampler,identity,completed,log,cuda=True),state_path)
            atomic_json_save(log,directory/'metrics.json')

        def validate(step):
            check_pins()
            for label,candidate in ([('raw',model)] if step == 0 else [('raw',model),('ema',averaged)]):
                checkpoint = make_snapshot(candidate,parent,recipe['validation_config'],parent_sha,recipe,step,label,PLAN_SHA)
                name = f'validation-session{session:03d}-step{step:08d}-{label}.pt'
                destination = directory/name
                require(not destination.exists(), 'Never overwrite a scored validation snapshot')
                save_snapshot_new(destination,checkpoint)
                snapshot_sha = sha(destination)
                evaluator.load_state_dict(checkpoint['model'],strict=True)
                measured = score(evaluator,*data['validation'],device,'fp32')
                measured.pop('window_nll_nats')
                require(math.isfinite(measured['bpb']) and measured['targets'] == 376599 and measured['utf8_bytes'] == 1148007,
                        'Nonfinite/incomplete full validation')
                row = dict(step=step,weights=label,session=session,checkpoint_file=name,checkpoint_sha256=snapshot_sha,**measured)
                log['validation_history'].append(row)
                log['validation_snapshots'].append(dict(step=step,weights=label,session=session,path=name,sha256=snapshot_sha))
                if measured['bpb'] < log.get('best',{}).get('bpb',float('inf')):
                    atomic_torch_save(checkpoint,directory/'checkpoint.pt')
                    log['best'] = dict(row,scoring_checkpoint_sha256=sha(directory/'checkpoint.pt'),
                                      selected_lineage_targets=checkpoint['train_tokens'])
                refresh()
                atomic_json_save(log,directory/'metrics.json')
                print(json.dumps(dict(validation=row)),flush=True)
            check_pins()

        if not args.resume:
            validate(0); save_state()
        for step in range(completed+1,recipe['steps']+1):
            model.train()
            starts = sample_starts(len(tokens),sampler).to(device)
            batch = tokens[starts[:,None]+torch.arange(257,device=device)]
            factor = schedule_factor(step,recipe['steps'],'cosine',recipe['warmup'],recipe['decay_steps'],recipe['lr_floor'])
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr']*factor
                optimizer.zero_grad(set_to_none=True)
            with autocast(device,precision):
                loss = branch_loss(model,batch[:,:-1],batch[:,1:],recipe['auxiliary_weight'])
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
            project = getattr(model,'project_cache_parameters',None)
            if project is not None:
                project()
            with torch.no_grad():
                for target,current in zip(averaged.parameters(),model.parameters()):
                    target.lerp_(current,.01)
            completed = step
            if step % 400 == 0:
                torch.cuda.synchronize(); refresh()
                row = dict(step=step,loss=loss.item(),gradient_norm=float(norm),schedule_factor=factor,seconds=log['seconds'],session=session)
                log['history'].append(row)
                atomic_json_save(log,directory/'metrics.json')
                print(json.dumps(row),flush=True)
            if step % recipe['eval_every'] == 0:
                validate(step); save_state()
        log['status'] = 'completed'
        save_state()
        return log
    except BaseException as error:
        if log is not None and metrics_mutation_allowed:
            if 'refresh' in locals():
                refresh()
            log.update(status='failed',completed_steps=completed,failure_type=type(error).__name__,failure_message=str(error),
                attempted_step=locals().get('step',0),failed_partial_optimizer_or_score_work_unquantified=True,
                failure_policy='Preserve completed work and last complete train_state; no automatic retry.')
            atomic_json_save(log,directory/'metrics.json')
        write_new(directory/('failure-'+attempt_id+'.json'),dict(status='failed',session=session,branch=args.branch,plan_sha256=PLAN_SHA,
            parent_sha256=parent_sha,completed_steps=completed,failure_type=type(error).__name__,failure_message=str(error),
            seconds_this_attempt=time.perf_counter()-wall_start,test_used=False,no_automatic_retry=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',choices=BRANCHES,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/PLAN_PATH)
    parser.add_argument('--resume',type=Path)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(dict(status=result['status'],best=result['best'],completed_steps=result['completed_steps'],
                         seconds=result['seconds']),allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
