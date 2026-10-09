"""Metadata-only full-log auditing and restart assembly for the four A1 seeds."""
from pathlib import Path
import hashlib,json,math,statistics

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    import os
    p=Path(p);tmp=p.with_name(p.name+'.new')
    with tmp.open('w') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    fd=os.open(str(p.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def rows(p,byte_count=None):
    with Path(p).open('rb') as f:b=f.read(byte_count)
    return [json.loads(line) for line in b.splitlines()]
def check_group(group,phase,epoch,n,aggregate):
    assert [r['offset'] for r in group]==list(range(0,n,32)),f'{phase} epoch{epoch} offsets'
    assert [r['samples'] for r in group]==[min(32,n-o) for o in range(0,n,32)],f'{phase} epoch{epoch} batches'
    assert sum(r['samples'] for r in group)==n and aggregate['samples']==n
    if phase!='test':
        assert sum(r['correct'] for r in group)==aggregate['correct']
        assert math.isclose(sum(r['loss']*r['samples'] for r in group)/n,aggregate['loss'],rel_tol=1e-9,abs_tol=1e-9)
    else:assert all('correct' not in r and 'loss' not in r for r in group),'Partial test score leaked'
def check_epochs(log,epochs):
    assert [e['epoch'] for e in epochs]==list(range(1,len(epochs)+1))
    for e in epochs:
        for phase,n in [('train',55000),('validation',5000)]:
            check_group([r for r in log if r['epoch']==e['epoch'] and r['phase']==phase],phase,e['epoch'],n,e[phase])
def cursor_counter(cursor):
    epochs=cursor['completed_epochs'];last=len(epochs)
    assert [e['epoch'] for e in epochs]==list(range(1,last+1))
    if cursor['phase']=='train':
        n=cursor['next_offset']
        assert cursor['epoch']==last+1 and 0<n<=55000 and (n==55000 or n%32==0)
        expected=1719*last+math.ceil(n/32)
    else:
        assert cursor['phase']=='epoch_complete' and cursor['epoch']==last and last>0
        expected=1719*last
    assert cursor['adam_step']==expected
    return expected
def prefix(root,contract):
    if not contract.get('resume'):return []
    resume=contract['resume'];p=root/resume['source_batches']
    assert sha(p)==resume['source_batches_sha256'],'Recovery prefix identity changed'
    return rows(p)
def assemble(root,prior,assembly):
    from validate_a1_arm64_long_slots_r1 import checkpoint_counter
    cursor_path=prior/'recovery-cursor.json'
    if not cursor_path.exists():return None
    cursor=read(cursor_path);contract=read(prior.parent/'contract.json')
    cursor_counter(cursor)
    log=prefix(root,contract)+rows(prior/'batches.jsonl',cursor['batch_log_bytes'])
    completed=cursor['completed_epochs'];last=len(completed)
    accepted=[r for r in log if r['phase']!='test' and (r['epoch']<=last or
        (cursor['phase']=='train' and r['phase']=='train' and r['epoch']==last+1 and r['offset']<cursor['next_offset']))]
    check_epochs(accepted,completed)
    if cursor['phase']=='train':
        assert cursor['epoch']==last+1
        group=[r for r in accepted if r['phase']=='train' and r['epoch']==last+1]
        n=cursor['next_offset'];assert 0<n<=55000 and (n==55000 or n%32==0)
        check_group(group,'train',last+1,n,dict(samples=n,correct=cursor['correct'],loss=cursor['total_loss']/n))
        expected=1719*last+math.ceil(n/32)
    else:
        assert cursor['phase']=='epoch_complete' and cursor['epoch']==last
        expected=1719*last
    assert cursor['adam_step']==expected
    checkpoint=root/cursor['checkpoint'];assert sha(checkpoint)==cursor['checkpoint_sha256']
    counter=checkpoint_counter(checkpoint,'atlas');assert counter['counters']==[expected]
    assert counter['runtime_sha256']==contract['runtime']['sha256']
    assembly.mkdir(exist_ok=False)
    p=assembly/'accepted-prefix.jsonl'
    with p.open('x') as f:
        for r in accepted:f.write(json.dumps(r,allow_nan=False)+'\n')
        f.flush()
        import os;os.fsync(f.fileno())
    checkpoints={f.name:dict(path=str(f.relative_to(root)),sha256=sha(f)) for f in prior.glob('best-epoch-*.json')}
    if cursor['selection']:
        assert checkpoints[cursor['selection']['checkpoint']]['sha256']==cursor['selection']['sha256']
    result=dict(cursor_path=str(cursor_path.relative_to(root)),cursor_sha256=sha(cursor_path),
        source_batches=str(p.relative_to(root)),source_batches_sha256=sha(p),retained_checkpoints=checkpoints,
        reason='Resume only durable committed training prefix or completed epoch; discarded segment preserved')
    save(assembly/'assembly.json',dict(resume=result,accepted_batch_rows=len(accepted),restored_adam_step=expected))
    return result
def audit(root,dest,contract):
    from validate_a1_arm64_long_slots_r1 import checkpoint_counter
    result=read(dest/'result.json');log=prefix(root,contract)+rows(dest/'batches.jsonl')
    assert result['status']=='completed' and result['test_status']=='completed_once'
    assert result['runtime_sha256']==contract['runtime']['sha256'] and result['A1_shape_qualification']=='passed'
    assert result['contract_sha256']==sha(dest.parent/'contract.json')
    assert result['source_hashes']['tools/run_a1_mnist_arm64_remaining_r1.py']==contract['implementation_sha256']
    manifest=read(root/'fixtures/a1-arm64-long-r1/manifest.json')
    assert result['shared_manifest_sha256']==sha(root/'fixtures/a1-arm64-long-r1/manifest.json')
    assert result['common_arrays_sha256']==manifest['files'][f"seed-{result['seed']}.npz"]['sha256']
    assert len(result['epochs'])==10
    check_epochs(log,result['epochs'])
    best=None;correct=-1
    for e in result['epochs']:
        if e['validation']['correct']>correct:
            best=e['epoch'];correct=e['validation']['correct']
            cp=checkpoint_counter(dest/f'best-epoch-{best}.json','atlas')
            assert cp['counters']==[1719*best] and cp['runtime_sha256']==contract['runtime']['sha256']
    selection=result['selection'];assert selection['epoch']==best and selection['validation_correct']==correct
    assert sha(dest/selection['checkpoint'])==selection['sha256']
    test=result['test'];assert test['samples']==10000 and test['accuracy']==test['correct']/10000
    check_group([r for r in log if r['phase']=='test'],'test',best,10000,test)
    marker=read(dest/'test-started.json')
    ledger=root/'evidence/a1-arm64-long-r1-test-once'/f"atlas-eager-seed-{result['seed']}-{contract['original_contract_sha256']}.json"
    assert read(ledger)==marker and marker['attempt']==1 and marker['selected']==selection
    assert marker['run_directory']==str(dest) and marker['contract_sha256']==result['contract_sha256']
    assert all(v=='1' for v in result['thread_environment'].values())
    if contract.get('resume'):
        assert result['resume']['restored_step']==read(root/contract['resume']['cursor_path'])['adam_step']
    return dict(status='evidence_consistent_full_seed',seed=result['seed'],completed_epochs=10,
        selected_epoch=best,best_validation_accuracy=correct/5000,test=test,errors=[],
        test_score_scope='Complete source-bound worker aggregate; held-out labels/predictions not reread',
        performance_ranking=False,restored=bool(contract.get('resume')))
def summary(root,records):
    seed11=read(root/'evidence/a1-arm64-resume-r3/atlas-seed-11/result.json')
    baseline=read(root/'evidence/a1-arm64-resume-r3/validation-readback-r1.json')
    assert baseline['audit']['errors']==[] and seed11['test']['samples']==10000
    results={11:seed11}
    for item in records:
        r=read(root/item['result']);assert item['audit']['errors']==[] and r['status']=='completed';results[r['seed']]=r
    assert sorted(results)==[11,23,37,51,71]
    competitor=read(root/'evidence/a1-queue-arm64-r1/validation-full-r1.json');assert competitor['errors']==[]
    atlas=[results[s]['test']['accuracy'] for s in sorted(results)]
    views={}
    for view in ['sj-layerwise','sj-compile','snn-layerwise','snn-compile']:
        source=[r for r in competitor['rows'] if r['view']==view]
        source.sort(key=lambda r:r['seed']);assert [r['seed'] for r in source]==sorted(results)
        scores=[r['test']['recorded_full_aggregate']['accuracy'] for r in source]
        views[view]=dict(mean=statistics.mean(scores),sample_sd=statistics.stdev(scores),
            paired_atlas_minus_contender=[a-b for a,b in zip(atlas,scores)])
    return dict(status='five_seed_quality_complete',seeds=sorted(results),per_seed=[dict(seed=s,test=results[s]['test'],selection=results[s]['selection']) for s in sorted(results)],
        atlas_mean=statistics.mean(atlas),atlas_sample_sd=statistics.stdev(atlas),contenders=views,
        performance_ranking=False,scope='Same numerical long-profile seed11 plus four remaining seeds; separate from prior1800s censored profile')
def self_test():
    good=[dict(offset=o,samples=min(32,70-o),correct=1,loss=.25) for o in range(0,70,32)]
    aggregate=dict(samples=70,correct=3,loss=.25)
    check_group(good,'train',1,70,aggregate)
    rejected=[]
    for name,bad in [('gap',good[1:]),('duplicate',good+[good[-1]]),('wrong_tail',[*good[:-1],dict(good[-1],samples=32)]),('wrong_correct',[dict(good[0],correct=2),*good[1:]])]:
        try:check_group(bad,'train',1,70,aggregate)
        except AssertionError:rejected.append(name)
    assert len(rejected)==4
    try:check_group(good,'test',1,70,aggregate)
    except AssertionError:rejected.append('partial_test_scores')
    assert len(rejected)==5
    epochs=[dict(epoch=i) for i in range(1,8)]
    for state in [dict(phase='train',epoch=8,next_offset=51200,adam_step=13633,completed_epochs=epochs),
                  dict(phase='train',epoch=8,next_offset=55000,adam_step=13752,completed_epochs=epochs),
                  dict(phase='epoch_complete',epoch=7,next_offset=55000,adam_step=12033,completed_epochs=epochs)]:
        assert cursor_counter(state)==state['adam_step']
    for key,value in [('next_offset',51201),('adam_step',13632),('epoch',9),('phase','validation')]:
        bad=dict(phase='train',epoch=8,next_offset=51200,adam_step=13633,completed_epochs=epochs);bad[key]=value
        try:cursor_counter(bad)
        except AssertionError:rejected.append('cursor_'+key)
    assert len(rejected)==9
    return dict(status='passed',checks=13,rejected=rejected,models_imported=False)
if __name__=='__main__':print(json.dumps(self_test()))
