"""Lock a metadata-verified parent, screen four fixed validation mixtures, confirm.

Only screen/confirm import Torch or read validation. Lock is metadata/hash-only.
No command reads test text, trains, or stores heldout probabilities/labels.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
PLAN='runs/push153_plan.json'
PLAN_SHA='3851b1c321ca34b32d4eefd4c52640ce35cfee38bd36aa18ad09329f1a6e3293'
LOCK='runs/push153/parent_lock.json'
SCREEN='runs/push153/screen.json'
SELECTION='runs/push153/selection.json'
IMPLEMENTATION='push153_mixture'
PARENT_IMPLEMENTATION='push150_expand'
MENU=(('control',None),('k8',8),('k32',32),('k128',128))
TARGETS,BYTES=376599,1148007
PROTOCOL='7506-mp1-wt2-v2'


def require(condition,message):
    if not condition:raise ValueError(message)


def child(root,name):
    name=str(name).replace('\\','/')
    require(name and ':' not in name and not name.startswith('/') and
        all(part not in ('','.','..') for part in name.split('/')),'Unsafe evidence path')
    path=(Path(root)/name).resolve()
    require(path.is_relative_to(Path(root).resolve()),'Evidence escaped root')
    return path


def sha(path):
    path=Path(path)
    require(not path.name.lower().startswith('wikitext_'),'No corpus hashing by metadata helper')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):
    require(Path(path).suffix=='.json','JSON metadata only')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes(root=ROOT):
    root=Path(root); pending=['push153_screen.py','push153_mixture.py','evaluate.py','push150_mkn_sweep.py'];seen={}
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=root/name;seen[name]=sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules=[entry.name for entry in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module and node.level==0 else [])
            for module in modules:
                candidate=module.split('.')[0]+'.py'
                if (root/candidate).is_file():pending.append(candidate)
    return dict(sorted(seen.items()))


def pin_map(root,values):
    for name,digest in values.items():require(sha(child(root,name))==digest,'Pinned evidence/source changed: '+name)


def verify_plan(root=ROOT):
    require(sha(Path(root)/PLAN)==PLAN_SHA,'Predeclared push153 plan changed')
    plan=read(Path(root)/PLAN)
    require([(row['name'],row['kappa']) for row in plan['menu']]==list(MENU) and plan['base_weight']==.15 and
        plan['selection']['minimum_improvement_over_locked_parent']==.001 and
        plan['selection']['maximum_resource_candidates']==3,'Fixed mixture/menu changed')
    require(not (Path(root)/'runs/push152-final-test').exists() and
        not (Path(root)/'runs/push153-final-test').exists() and
        not (Path(root)/'runs/push153_candidate_frozen.json').exists(),'Development must precede153 freeze and all new test scoring')
    return plan


def read_work_module(root,name):
    work=Path(root).parents[1]
    sys.path.insert(0,str(work))
    try:
        spec=importlib.util.spec_from_file_location('_'+name.replace('.','_'),work/name)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path.pop(0)
    return module


def predecessor_fallback(root,selection,final):
    """Every allowed push152 rank must truly fail before historical fallback."""
    if not selection['resource_audit_candidates']:
        require(selection['status']=='no_new_validation_qualified_candidate','Missing predecessor candidates')
        return {},[]
    _,work_sources=final.verify_units(root,root/final.UNIT,root/final.AUDIT_UNIT)
    auditor=final.load_auditor(root,work_sources)
    evidence={};outcomes=[];selection_sha=sha(root/final.SELECTION)
    for rank,row in enumerate(selection['resource_audit_candidates']):
        pins={};auditor.verify_prior_ranks(root,selection,rank,pins)
        evidence.update({Path(p).relative_to(root).as_posix():digest for p,digest in pins.items()})
        directory=root/f'runs/push152-resource-rank{rank}'
        failed=directory/'audit.failure.json'
        if failed.exists():
            value=read(failed);auditor.failure_identity(value,row,rank,selection_sha)
            require(value['candidate_audit_started'] is True and value['limits_pass'] is False and
                value['audit_script_sha256']==work_sources['push152_audit.py'] and
                not (directory/'audit.json').exists(),'Preflight/order failure is not candidate rejection')
            evidence[failed.relative_to(root).as_posix()]=sha(failed)
            outcomes.append(dict(rank=rank,stage='audit',record=failed.relative_to(root).as_posix()));continue
        audit,pins=final.verify_audit(root,rank,row,selection_sha,work_sources,False);evidence.update(pins)
        timing_failure=directory/'timing.failure.json'
        if timing_failure.exists():
            value=read(timing_failure);auditor.failure_identity(value,row,rank,selection_sha)
            require(not (directory/'timing/summary.json').exists(),'Conflicting timing completion/failure')
            started_path,process_path=child(root,value['started_record']),child(root,value['process_record'])
            require(sha(started_path)==value['started_record_sha256'] and sha(process_path)==value['process_record_sha256'],
                'Timing failure ledger changed')
            started,process=read(started_path),read(process_path);command=started['command']
            require(len(command)==7 and Path(command[2]).resolve()==root/'push145_fresh_benchmark.py' and
                command[1]=='-B' and command[3]=='--candidate' and command[5]=='--run-dir' and
                (root/command[4]).resolve()==child(root,row['checkpoint']) and
                (root/command[6]).resolve()==directory/'timing' and started['test_used'] is False and
                started['plan_sha256']==final.PLAN_SHA and started['script_sha256']==final.TIMING_SCRIPT_SHA and
                process['started_record_sha256']==value['started_record_sha256'] and process['test_used'] is False and
                process['child_process_exited'] is True and process['child_tree_review_required'] is False and
                process['failure'] is None and type(process['returncode']) is int and process['returncode']!=0,
                'Timing did not actually terminate unsuccessfully')
            for path in (timing_failure,started_path,process_path):evidence[path.relative_to(root).as_posix()]=sha(path)
            outcomes.append(dict(rank=rank,stage='timing_process',record=timing_failure.relative_to(root).as_posix()));continue
        timing,pins=final.verify_timing_files(root,rank,row,audit,False);evidence.update(pins)
        require(not (timing['cpu_limit_pass'] and timing['ram_limit_pass'] and audit['asset_limit_pass'] and
            audit['ram_limit_pass']),'Passing predecessor must be frozen, not skipped')
        outcomes.append(dict(rank=rank,stage='measured_resources',record=(directory/'timing/summary.json').relative_to(root).as_posix()))
    return evidence,outcomes


def lock_parent(root=ROOT):
    root=Path(root).resolve();verify_plan(root)
    path=root/LOCK;require(not path.exists(),'Parent lock is immutable')
    import push152_select as selector
    import push152_finalize as final
    from push152_launch import preserve_ten
    protected=preserve_ten()
    selection=read(root/final.SELECTION);fresh=selector.collect(root)
    require({k:v for k,v in selection.items() if k!='created_utc'}=={k:v for k,v in fresh.items() if k!='created_utc'},
        'Predecessor five-run accounting does not match frozen evidence')
    evidence=dict(selection['evidence_sha256']);evidence[final.SELECTION]=sha(root/final.SELECTION)
    if (root/final.FREEZE).exists():
        state=final.verify_freeze(root,root/final.SELECTION);frozen=state['freeze'];winner=state['winner']
        parent=dict(checkpoint=winner['checkpoint'],checkpoint_sha256=winner['checkpoint_sha256'],
            validation_bpb_gpu_fp32=winner['validation_bpb_gpu_fp32'],validation_bpb_cpu_fp32=frozen['validation_bpb_cpu_fp32'],
            selected_lineage_targets=winner['selected_lineage_targets'],parameters=winner['parameters'],
            implementation=PARENT_IMPLEMENTATION,implementation_sha256=final.IMPLEMENTATION_SHA,
            audit_record=frozen['audit_record'],audit_sha256=frozen['audit_sha256'])
        mode='frozen_push152';provenance=dict(freeze=final.FREEZE,freeze_sha256=sha(state['freeze_path']))
        evidence.update(frozen['evidence_sha256']);evidence[final.FREEZE]=sha(state['freeze_path'])
        config_file=winner['run_dir']+'/metrics.json'
    else:
        extra,outcomes=predecessor_fallback(root,selection,final);evidence.update(extra)
        old=selection['parent'];parent=dict(checkpoint=old['path'],checkpoint_sha256=old['sha256'],
            validation_bpb_gpu_fp32=old['validation_bpb_gpu_fp32'],validation_bpb_cpu_fp32=old['validation_bpb_cpu_fp32'],
            selected_lineage_targets=old['selected_lineage_targets'],parameters=5597331,
            implementation=old['implementation'],implementation_sha256=old['implementation_sha256'],
            audit_record=old['audit'],audit_sha256=old['audit_sha256'])
        require(sha(root/parent['audit_record'])==parent['audit_sha256'],'Historical parent audit changed')
        evidence[parent['audit_record']]=parent['audit_sha256']
        mode='retained_push151';provenance=dict(predecessor_rejections=outcomes)
        choices=[row for row in selection['run_records'] if row['status']=='completed' and row['name']!='expanded']
        require(choices,'No verified metadata configuration for historical fallback')
        config_file=choices[0]['run_dir']+'/metrics.json'
    require(sha(root/parent['checkpoint'])==parent['checkpoint_sha256'],'Parent checkpoint bytes changed')
    configuration=read(root/config_file)['recipe']['validation_config']
    require(all(configuration.get(k)==v for k,v in dict(temperature=1.3,mkn_weight=.15,mkn_order=5,
        discount_scale=.7,context=256).items()),'Parent fixed configuration differs')
    evidence[config_file]=sha(root/config_file);sources=source_hashes(root)
    record=dict(status='parent_locked',round='push153',plan_sha256=PLAN_SHA,mode=mode,parent=parent,
        configuration=configuration,configuration_sha256=canonical_sha(configuration),configuration_record=config_file,
        predecessor_selection_sha256=sha(root/final.SELECTION),predecessor_costs=selection['costs'],provenance=provenance,
        source_sha256=sources,evidence_sha256=evidence,script_sha256=sha(__file__),
        created_utc=datetime.now(timezone.utc).isoformat(),production_weights_loaded=False,test_used=False,
        test_deferral_enforced=True,new_training_targets=0,prior_test_exposure_disclosed=True,
        prior_test_history_is_exhaustive=False,preserved_archives=protected)
    pin_map(root,sources);pin_map(root,evidence)
    path.parent.mkdir(parents=True,exist_ok=True);write_new(path,record);return record


def load_lock(root=ROOT):
    root=Path(root);verify_plan(root);record=read(root/LOCK)
    require(record['status']=='parent_locked' and record['plan_sha256']==PLAN_SHA and record['test_used'] is False and
        record['production_weights_loaded'] is False and record['test_deferral_enforced'] is True and
        record['configuration_sha256']==canonical_sha(record['configuration']) and
        record['source_sha256']==source_hashes(root),'Parent lock/source identity changed')
    pin_map(root,record['evidence_sha256'])
    require(sha(root/record['parent']['checkpoint'])==record['parent']['checkpoint_sha256'],'Locked parent changed')
    return record


def validate_parent(saved,lock,torch):
    require(saved['protocol']==PROTOCOL and saved['implementation']==PARENT_IMPLEMENTATION and
        saved['config']==lock['configuration'] and saved['train_tokens']==lock['parent']['selected_lineage_targets'],
        'Loaded parent differs from metadata-only lock')
    require(sum(name.startswith('ngram.') for name in saved['model'])==21 and
        torch.equal(saved['model']['token.weight'],saved['model']['head.weight']),'Parent statistics/tied weights differ')


def target_losses(neural,counts,targets,kappa):
    """One batch only; returned losses/weights are never saved as target caches."""
    import torch
    from push153_mixture import weights_from_observed_targets
    require(neural.dtype==counts.dtype==torch.float32 and neural.shape==counts.shape and
        neural.ndim==3 and targets.shape==neural.shape[:2],'Screen expects FP32 full expert distributions')
    chosen=targets.clamp_min(0).unsqueeze(-1)
    n=neural.gather(-1,chosen).squeeze(-1);m=counts.gather(-1,chosen).squeeze(-1)
    weights=weights_from_observed_targets(n[:,:-1],m[:,:-1],kappa)
    fixed=n.clone().mul_(.85).add_(m,alpha=.15)
    mixed=fixed if kappa is None else n*(1-weights)+m*weights
    if kappa is not None:mixed[:,0].copy_(fixed[:,0])
    require(bool(torch.isfinite(mixed).all()) and bool((mixed>0).all()),'Invalid target mixture probability')
    losses=-mixed.clamp_min(1e-30).log();losses.masked_fill_(targets==-100,0)
    return losses,weights


def score_metadata(nll,count,seconds):
    require(count==TARGETS and math.isfinite(nll) and nll>0,'Incomplete/nonfinite screen')
    return dict(bpb=nll/math.log(2)/BYTES,nll_nats=nll,token_ppl=math.exp(nll/count),targets=count,
        utf8_bytes=BYTES,seconds=seconds,protocol=PROTOCOL,split='validation',precision='fp32',device='cuda:0',test_used=False)


def ranked_trials(trials,parent):
    require([(row['name'],row['kappa']) for row in trials]==list(MENU),'Fixed four-choice screen required')
    require(abs(trials[0]['validation_bpb_gpu_fp32']-parent['validation_bpb_gpu_fp32'])<=1e-5,
        'Shared-pass control differs from locked parent')
    order={name:index for index,(name,_) in enumerate(MENU)}
    return sorted([row for row in trials[1:] if row['passes_validation_improvement_gate']],
        key=lambda row:(row['validation_bpb_gpu_fp32'],order[row['name']]))


def screen(root=ROOT):
    root=Path(root);lock=load_lock(root);directory=root/'runs/push153'
    require(not any((directory/name).exists() for name in ('screen.started.json','screen.json','screen.failure.json','selection.json')),
        'Preserve prior screen attempt')
    started=time.perf_counter();batches=targets=0;sources=lock['source_sha256']
    write_new(directory/'screen.started.json',dict(plan_sha256=PLAN_SHA,parent_lock_sha256=sha(root/LOCK),
        source_sha256=sources,test_used=False,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        import numpy as np
        import torch
        from common import make_model,setup,windows
        from push150_mkn_sweep import load_one_split
        device,precision=setup('cuda','fp32',4);require(device.type=='cuda' and precision=='fp32','GPU FP32 screen only')
        saved=torch.load(root/lock['parent']['checkpoint'],map_location='cpu',weights_only=True);validate_parent(saved,lock,torch)
        model,implementation_sha=make_model(PARENT_IMPLEMENTATION,saved['config'],device)
        model.load_state_dict(saved['model'],strict=True);model.eval();del saved
        require(sum(p.numel() for p in model.parameters())==lock['parent']['parameters'],'Parent parameters differ')
        ids,data=load_one_split('validation',root)
        require(data['utf8_bytes']==BYTES and len(ids)-1==TARGETS,'Full fixed validation required')
        tokens=torch.from_numpy(ids);del ids
        losses={name:[] for name,_ in MENU}
        with torch.no_grad():
            for x,y in windows(tokens,32,256):
                x,y=x.to(device),y.to(device)
                neural=model.neural_and_cache(x);counts=model.ngram.probabilities(x,5)
                for name,kappa in MENU:
                    loss,weight=target_losses(neural,counts,y,kappa)
                    losses[name].extend(loss.double().sum(-1).cpu().tolist())
                    del loss,weight
                targets+=int((y!=-100).sum());batches+=1
                del neural,counts,x,y
        torch.cuda.synchronize(device);seconds=time.perf_counter()-started;trials=[];evidence={}
        for name,kappa in MENU:
            measured=score_metadata(sum(losses[name]),targets,seconds)
            path=directory/(name+'.validation.window-nll.npy')
            with path.open('xb') as stream:np.save(stream,np.asarray(losses[name],dtype=np.float64))
            evidence[path.relative_to(root).as_posix()]=sha(path)
            gain=lock['parent']['validation_bpb_gpu_fp32']-measured['bpb']
            trials.append(dict(name=name,kappa=kappa,validation_bpb_gpu_fp32=measured['bpb'],screen_measurement=measured,
                improvement_vs_parent=gain,passes_validation_improvement_gate=(name!='control' and gain>=.001),
                selected_lineage_targets=lock['parent']['selected_lineage_targets'],parameters=lock['parent']['parameters'],
                expected_ranks=lock['configuration']['push150_ranks'],parent_checkpoint_sha256=lock['parent']['checkpoint_sha256']))
        ranked=ranked_trials(trials,lock['parent'])
        report=dict(status='validation_screen_complete',round='push153',plan_sha256=PLAN_SHA,parent_lock_sha256=sha(root/LOCK),
            script_sha256=sha(__file__),source_sha256=sources,evidence_sha256=evidence,selection_split='validation',test_used=False,
            menu=[dict(name=name,kappa=kappa) for name,kappa in MENU],trials=trials,validation_qualified_in_rank_order=ranked,
            resource_audit_candidates=ranked[:3],minimum_improvement=.001,parent=lock['parent'],validation=data,
            actual_shared_expert_passes=1,actual_expert_batches=batches,full_validation_scoring_choices=4,
            new_training_targets=0,new_parameters=0,new_statistics_scans=0,seconds_total=seconds,
            only_aggregate_window_losses_saved=True,heldout_probability_or_label_arrays_saved=False,
            inherited_ancestry_not_added_as_new_cost=True,predecessor_costs=lock['predecessor_costs'])
        load_lock(root);write_new(root/SCREEN,report)
        selection=copy.deepcopy(report);selection['screen_record_sha256']=sha(root/SCREEN)
        selection['evidence_sha256'][SCREEN]=sha(root/SCREEN)
        write_new(root/SELECTION,selection);return selection
    except BaseException as error:
        write_new(directory/'screen.failure.json',dict(status='failed',plan_sha256=PLAN_SHA,parent_lock_sha256=sha(root/LOCK),
            script_sha256=sha(__file__),source_sha256=sources,failure_type=type(error).__name__,failure_message=str(error),
            completed_expert_batches=batches,completed_targets=targets,seconds_total=time.perf_counter()-started,
            new_training_targets=0,test_used=False,no_automatic_retry=True));raise


def check_selection(selection,lock):
    require(selection['status']=='validation_screen_complete' and selection['plan_sha256']==PLAN_SHA and
        selection['test_used'] is False and selection['selection_split']=='validation' and
        selection['source_sha256']==lock['source_sha256'] and selection['parent']==lock['parent'] and
        selection['full_validation_scoring_choices']==4 and selection['actual_shared_expert_passes']==1 and
        selection['heldout_probability_or_label_arrays_saved'] is False,'Screen identity differs')
    ranked=ranked_trials(selection['trials'],lock['parent'])
    require(selection['validation_qualified_in_rank_order']==ranked and selection['resource_audit_candidates']==ranked[:3],
        'Screen ranking changed')
    for row in selection['trials']:
        measured=row['screen_measurement'];gain=lock['parent']['validation_bpb_gpu_fp32']-measured['bpb']
        require(measured['targets']==TARGETS and measured['utf8_bytes']==BYTES and measured['precision']=='fp32' and
            measured['device']=='cuda:0' and measured['split']=='validation' and measured['test_used'] is False and
            measured['protocol']==PROTOCOL and all(type(measured.get(key)) in (int,float) and
                math.isfinite(measured[key]) and measured[key]>0 for key in ('bpb','nll_nats','token_ppl','seconds')) and
            abs(measured['bpb']-measured['nll_nats']/math.log(2)/BYTES)<1e-10 and
            row['validation_bpb_gpu_fp32']==measured['bpb'] and row['improvement_vs_parent']==gain and
            row['passes_validation_improvement_gate'] is (row['name']!='control' and gain>=.001) and
            row['selected_lineage_targets']==lock['parent']['selected_lineage_targets'] and
            row['parameters']==lock['parent']['parameters'],'Screen scoring/accounting differs')


def verify_confirmation_failure(record,parent_lock_sha,selection_sha,rank,row):
    require(record.get('status')=='failed' and record.get('plan_sha256')==PLAN_SHA and
        record.get('parent_lock_sha256')==parent_lock_sha and record.get('selection_sha256')==selection_sha and
        record.get('resource_rank')==rank and record.get('name')==row['name'] and record.get('kappa')==row['kappa'] and
        record.get('candidate_confirmation_started') is True and record.get('test_used') is False and
        record.get('no_automatic_retry') is True and isinstance(record.get('failure_type'),str) and record['failure_type'] and
        isinstance(record.get('failure_message'),str),'Failure is not a bound actual production confirmation')


def check_confirmation(record,lock,selection,rank):
    row=selection['resource_audit_candidates'][rank];measured=record['measurement']
    require(record['status']=='completed' and record['plan_sha256']==PLAN_SHA and record['resource_rank']==rank and
        record['name']==row['name'] and record['kappa']==row['kappa'] and record['implementation']==IMPLEMENTATION and
        record['selected_lineage_targets']==lock['parent']['selected_lineage_targets'] and
        record['all_parent_tensors_bitwise_identical'] is True and record['statistics_buffers']==21 and
        record['parameters']==lock['parent']['parameters'] and record['new_training_targets']==0 and record['test_used'] is False,
        'Production confirmation identity differs')
    require(record['source_sha256']==lock['source_sha256'] and record['script_sha256']==lock['source_sha256']['push153_screen.py'] and
        record['implementation_sha256']==lock['source_sha256']['push153_mixture.py'] and
        measured['checkpoint_sha256']==record['checkpoint_sha256'] and
        measured['implementation_sha256']==record['implementation_sha256'] and
        measured['evaluator_sha256']==lock['source_sha256']['evaluate.py'] and measured['protocol']==PROTOCOL,
        'Production confirmation source or checkpoint identity differs')
    difference=abs(measured['bpb']-row['validation_bpb_gpu_fp32'])
    require(measured['targets']==TARGETS and measured['utf8_bytes']==BYTES and
        all(type(measured.get(key)) in (int,float) and math.isfinite(measured[key]) and measured[key]>0
            for key in ('bpb','nll_nats','token_ppl','seconds')) and
        abs(measured['bpb']-measured['nll_nats']/math.log(2)/BYTES)<1e-10 and
        measured['precision']=='fp32' and measured['device']=='cuda:0' and measured['split']=='validation' and
        measured['test_used'] is False and difference<=1e-5 and
        record['shortcut_full_bpb_absolute_difference']==difference,'Production scorer differs from shared screen')


def candidate_checkpoint(parent,kappa,lock,torch):
    require(kappa in (8,32,128),'Only declared dynamic candidates can be rebuilt')
    candidate=copy.deepcopy(parent);candidate['implementation']=IMPLEMENTATION
    candidate['config']['push153_kappa']=kappa
    candidate['push153_parent_sha256']=lock['parent']['checkpoint_sha256']
    candidate['push153_plan_sha256']=PLAN_SHA
    candidate['push153_new_training_targets']=0
    require(set(candidate['model'])==set(parent['model']),'State keys changed')
    for name,value in parent['model'].items():
        other=candidate['model'][name]
        require(value.dtype==other.dtype and value.shape==other.shape and torch.equal(value,other),'Parent tensor changed: '+name)
    return candidate


def confirm(rank,root=ROOT):
    root=Path(root);lock=load_lock(root);selection=read(root/SELECTION);check_selection(selection,lock)
    require(selection['parent_lock_sha256']==sha(root/LOCK) and selection['screen_record_sha256']==sha(root/SCREEN),
        'Screen/parent provenance changed');pin_map(root,selection['evidence_sha256'])
    require(type(rank) is int and 0<=rank<len(selection['resource_audit_candidates'])<=3,'Undeclared confirmation rank')
    auditor=read_work_module(root,'push153_audit.py');pins={}
    auditor.verify_prior_dispositions(root,selection,rank,pins)
    row=selection['resource_audit_candidates'][rank];directory=root/f'runs/push153-resource-rank{rank}'
    directory.mkdir(exist_ok=True)
    require(not any((directory/name).exists() for name in ('confirmation.started.json','confirmation.json',
        'confirmation.failure.json','checkpoint.pt')),'Preserve every prior confirmation attempt')
    started=time.perf_counter();cp_sha=None;full_scored=False;active=False
    identity=dict(plan_sha256=PLAN_SHA,parent_lock_sha256=sha(root/LOCK),selection_sha256=sha(root/SELECTION),
        resource_rank=rank,name=row['name'],kappa=row['kappa'],script_sha256=sha(__file__),
        source_sha256=lock['source_sha256'],test_used=False)
    write_new(directory/'confirmation.started.json',dict(**identity,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        import numpy as np
        import torch
        from common import make_model,setup
        from evaluate import score
        from push150_mkn_sweep import load_one_split
        saved=torch.load(root/lock['parent']['checkpoint'],map_location='cpu',weights_only=True);validate_parent(saved,lock,torch)
        candidate=candidate_checkpoint(saved,row['kappa'],lock,torch);del saved
        checkpoint=directory/'checkpoint.pt'
        with checkpoint.open('xb') as stream:torch.save(candidate,stream)
        cp_sha=sha(checkpoint);device,precision=setup('cuda','fp32',4)
        require(device.type=='cuda' and precision=='fp32','GPU FP32 production confirmation only')
        active=True
        model,implementation_sha=make_model(IMPLEMENTATION,candidate['config'],device)
        model.load_state_dict(candidate['model'],strict=True);model.eval()
        require(sum(p.numel() for p in model.parameters())==lock['parent']['parameters'] and model.head.weight is model.token.weight,
            'Production parameter count or tying changed')
        for name,value in model.state_dict().items():
            require(torch.equal(value.cpu(),candidate['model'][name]),'Scored model differs from saved checkpoint: '+name)
        del candidate
        ids,data=load_one_split('validation',root)
        measured=score(model,torch.from_numpy(ids),data['utf8_bytes'],device,precision);full_scored=True
        losses=measured.pop('window_nll_nats');loss_path=directory/'confirmation.validation.window-nll.npy'
        with loss_path.open('xb') as stream:np.save(stream,np.asarray(losses,dtype=np.float64))
        measured.update(protocol=PROTOCOL,split='validation',precision='fp32',device='cuda:0',test_used=False,
            checkpoint_sha256=cp_sha,implementation_sha256=implementation_sha,evaluator_sha256=sha(root/'evaluate.py'),
            tokenizer_sha256=data['tokenizer_sha256'])
        report=dict(**identity,status='completed',checkpoint=checkpoint.relative_to(root).as_posix(),checkpoint_sha256=cp_sha,
            checkpoint_bytes=checkpoint.stat().st_size,implementation=IMPLEMENTATION,implementation_sha256=implementation_sha,
            selected_lineage_targets=lock['parent']['selected_lineage_targets'],parameters=lock['parent']['parameters'],
            measurement=measured,shortcut_full_bpb_absolute_difference=abs(measured['bpb']-row['validation_bpb_gpu_fp32']),
            all_parent_tensors_bitwise_identical=True,statistics_buffers=21,new_training_targets=0,validation=data,
            window_nll_sha256=sha(loss_path),full_validation_passes=1,seconds_total=time.perf_counter()-started)
        check_confirmation(report,lock,selection,rank);load_lock(root)
        require(sha(root/SELECTION)==identity['selection_sha256'] and sha(checkpoint)==cp_sha,'Confirmation inputs changed')
        for path,digest in pins.items():require(sha(path)==digest,'Earlier-rank disposition changed')
        write_new(directory/'confirmation.json',report);return report
    except BaseException as error:
        write_new(directory/'confirmation.failure.json',dict(**identity,status='failed',failure_type=type(error).__name__,
            failure_message=str(error),candidate_confirmation_started=active,checkpoint_sha256=cp_sha,
            full_validation_scored=full_scored,seconds_total=time.perf_counter()-started,no_automatic_retry=True));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('lock');sub.add_parser('screen');confirm_parser=sub.add_parser('confirm')
    confirm_parser.add_argument('--rank',type=int,choices=range(3),default=0)
    args=parser.parse_args()
    result=lock_parent() if args.command=='lock' else screen() if args.command=='screen' else confirm(args.rank)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
