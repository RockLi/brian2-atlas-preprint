"""Static-only preparation of the remaining four unchanged ARM64 A1 seeds."""
from pathlib import Path
import ast, hashlib, json

ROOT=Path(__file__).resolve().parents[1]

def change(source, before, after):
    assert source.count(before)==1, before[:100]
    return source.replace(before,after)

def main():
    original=(ROOT/'tools/run_a1_mnist_arm64_long_r1.py').read_text()
    s=original
    s=change(s,"schema='a1-mnist-run-arm64-long-r1'","schema='a1-mnist-run-arm64-remaining-r1'")
    s=change(s,"{sha(args.contract)}.json'","{read(args.contract)['original_contract_sha256']}.json'")
    s=change(s,"manifest['contract_sha256']!=sha(args.contract)","manifest['contract_sha256']!=spec['original_contract_sha256']")
    s=change(s,"contract.get('profile') != 'arm64-long-r1'","contract.get('profile') != 'arm64-remaining-r1'")
    s=change(s,"        log=(dest/'batches.jsonl').open('x')",'''        resume=spec.get('resume')
        cursor=None
        if resume:
            if sha(root/resume['cursor_path'])!=resume['cursor_sha256']:
                raise Unqualified('Frozen recovery cursor changed')
            cursor=read(root/resume['cursor_path'])
            checkpoint=root/cursor['checkpoint']
            if sha(checkpoint)!=cursor['checkpoint_sha256']:
                raise Unqualified('Frozen recovery checkpoint changed')
            completed=cursor['completed_epochs']
            if [e['epoch'] for e in completed]!=list(range(1,len(completed)+1)):
                raise Unqualified('Recovery complete epoch sequence differs')
            if any(e['train']['samples']!=55000 or e['validation']['samples']!=5000 for e in completed):
                raise Unqualified('Recovery epoch denominators differ')
            if cursor['phase']=='train':
                if cursor['epoch']!=len(completed)+1 or not 0<cursor['next_offset']<=55000:
                    raise Unqualified('Recovery train cursor differs')
                if cursor['next_offset']!=55000 and cursor['next_offset']%32:
                    raise Unqualified('Recovery batch boundary differs')
                expected=1719*len(completed)+math.ceil(cursor['next_offset']/32)
            elif cursor['phase']=='epoch_complete':
                expected=1719*len(completed)
            else:raise Unqualified('Unknown recovery phase')
            if cursor['adam_step']!=expected:raise Unqualified('Recovery Adam counter differs')
            model.restore(checkpoint)
            if model.counter()!=expected:raise Unqualified('Restored Adam counter differs')
            import shutil
            for name,identity in resume['retained_checkpoints'].items():
                source=root/identity['path']
                if sha(source)!=identity['sha256']:raise Unqualified('Prior best checkpoint changed')
                shutil.copyfile(source,dest/name)
            report['epochs']=copy.deepcopy(completed)
            report['selection']=copy.deepcopy(cursor['selection'])
            report['resume']=dict(resume,restored_step=expected,phase=cursor['phase'],next_offset=cursor['next_offset'])
            dump(dest/'resume-receipt.json',report['resume'])
        log=(dest/'batches.jsonl').open('x')
        def recovery(epoch,samples,correct,total_loss,phase='train'):
            log.flush();os.fsync(log.fileno())
            path=dest/f'recovery-epoch{epoch}-samples{samples}.json'
            if not path.exists():model.save(path)
            with path.open('rb') as durable:os.fsync(durable.fileno())
            state=dict(schema='a1-remaining-recovery-r1',seed=args.seed,phase=phase,epoch=epoch,
                next_offset=samples,training_samples=samples,correct=correct,total_loss=total_loss,
                adam_step=model.counter(),checkpoint=str(path.relative_to(root)),checkpoint_sha256=sha(path),
                completed_epochs=copy.deepcopy(report['epochs']),selection=copy.deepcopy(report['selection']),
                contract_sha256=sha(args.contract),source_directory=str(dest.relative_to(root)),
                batch_log_bytes=log.tell(),scope='Recovery state; complete validation alone permits selection')
            dump(dest/'recovery-cursor.json',state,replace=True)''')
    s=change(s,"            samples=correct=0;total_loss=0.;start=time.monotonic()",'''            prefix=cursor if cursor and cursor['phase']=='train' and training and epoch==cursor['epoch'] else None
            samples=prefix['training_samples'] if prefix else 0
            correct=prefix['correct'] if prefix else 0
            total_loss=prefix['total_loss'] if prefix else 0.
            start=time.monotonic()''')
    s=change(s,"            for offset in range(0,len(ids),32):","            for offset in range(prefix['next_offset'] if prefix else 0,len(ids),32):")
    s=change(s,"                log.write(json.dumps(row,allow_nan=False)+'\\n');log.flush()",'''                log.write(json.dumps(row,allow_nan=False)+'\\n');log.flush()
                if training and ((offset//32+1)%100==0 or samples==len(ids)):
                    recovery(epoch,samples,correct,total_loss)''')
    s=change(s,"        for epoch in range(1,11):","        for epoch in range(len(report['epochs'])+1,11):")
    s=change(s,"            if epoch==10 or remaining<=row['continue_threshold_s']:break",'''            recovery(epoch,trained['samples'],trained['correct'],trained['loss']*trained['samples'],'epoch_complete')
            if epoch==10:break''')
    s=change(s,"        if report['selection'] is None:raise CapReached('no complete epoch selected')","        if len(report['epochs'])!=10 or report['selection'] is None:raise CapReached('All10epochs required before final test')")
    s=change(s,"        primary_timing='coordinator total wall includes cold/input construction/full train/validation/checkpoint/finaltest',","        primary_timing='Current session wall; any recovery segments recorded separately; no spliced end-to-end timing',")
    s=change(s,"                checkpoint=dest/(f'best-epoch-{epoch}'+model.checkpoint_suffix);model.save(checkpoint);check_cap(deadline)","                checkpoint=dest/(f'best-epoch-{epoch}'+model.checkpoint_suffix);model.save(checkpoint);check_cap(deadline)\n                with checkpoint.open('rb') as durable:os.fsync(durable.fileno())")
    s=change(s,"        partial=path.with_name(path.name+'.new');partial.write_text(content);os.replace(partial,path)",'''        partial=path.with_name(path.name+'.new')
        with partial.open('w') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.replace(partial,path)
        descriptor=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)''')
    s=change(s,"        with path.open('x') as stream:stream.write(content)","        with path.open('x') as stream:\n            stream.write(content);stream.flush();os.fsync(stream.fileno())")
    ast.parse(s)
    old=ast.parse(original);new=ast.parse(s)
    names=['encode','Atlas','Torch','compare','qualify']
    def nodes(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if hasattr(n,'name')}
    for name in names:assert nodes(old)[name]==nodes(new)[name],name
    worker=ROOT/'tools/run_a1_mnist_arm64_remaining_r1.py'
    with worker.open('x') as f:f.write(s)
    receipt={'schema':'a1-remaining-static-equivalence-r1','unchanged_AST':names,'training_runtime_unchanged':True,'model_imported':False,'worker_sha256':hashlib.sha256(s.encode()).hexdigest(),'parent_sha256':hashlib.sha256(original.encode()).hexdigest()}
    preparation=ROOT/'evidence/a1-arm64-remaining-preparation-r1';preparation.mkdir(exist_ok=False)
    (preparation/'static-equivalence.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
