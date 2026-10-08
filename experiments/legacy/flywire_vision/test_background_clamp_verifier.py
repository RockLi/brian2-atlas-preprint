"""Mutation checks against a complete real P12b archive; never launches a simulator.

Run: python -m flywire_vision.test_background_clamp_verifier ROOT
Temporary views hardlink immutable arrays, copy JSON, and atomically replace any
mutated array. No frozen original is modified, including through hardlinks.
"""
import argparse
import contextlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save
from .verify_background_clamp import verify,delivery_checks


class DeliveryAuditChecks(unittest.TestCase):
    def setUp(self):
        self.offsets=np.array([0,0,1,2]);self.targets=np.array([0,0]);self.weights=np.array([[2.,-3.]])
        self.events=[(20,1),(20,2)];rows=[]
        for source,edge,w in ((1,0,2.),(2,1,-3.)):
            ge=(.275/52)*(w*int(w>0));gi=-((.275/52)*4)*(w*int(w<0));v=np.array([ge,gi,.1,.1+ge,.2,.2+gi],dtype=np.float64)
            rows.append(np.concatenate([np.array([38,source,edge,0],dtype=np.uint64),v.view(np.uint64)]))
        self.audit=np.stack(rows)
    def check_data(self,data):return delivery_checks(data,self.events,self.offsets,self.targets,self.weights)
    def test_excitation_inhibition_and_delay(self):
        self.assertTrue(all(self.check_data(self.audit).values()))
        wrong=self.audit.copy();wrong[0,0]+=1
        self.assertFalse(self.check_data(wrong)['actual_deliveries_full_CSR'])
        wrong=self.audit.copy();wrong[0,3]=1
        self.assertFalse(self.check_data(wrong)['actual_deliveries_full_CSR'])
    def test_reject_wrong_sign_or_actual_increment(self):
        wrong=self.audit.copy();wrong[1,5]=np.array([-1.],dtype=np.float64).view(np.uint64)[0]
        self.assertFalse(self.check_data(wrong)['actual_weights_and_state_writes'])
        wrong=self.audit.copy();wrong[0,7]=np.array([.9],dtype=np.float64).view(np.uint64)[0]
        self.assertFalse(self.check_data(wrong)['actual_weights_and_state_writes'])
    def test_duplicate_and_missing_delivery(self):
        self.assertFalse(self.check_data(np.concatenate([self.audit,self.audit[:1]]))['no_duplicate_injection'])
        self.assertFalse(self.check_data(self.audit[:1])['actual_deliveries_full_CSR'])


def run(source):
    initial=read(source/'verification.json')
    if not initial['all_passed'] or initial['report_sha256']!=sha(source/'report.json'):raise ValueError('need verified real archive')
    original={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    outcomes=[]
    mutations=[('metric_sign','all_metrics_independently_recomputed'),('background_changed_per_order','literal_stimuli_fixed_background'),('duplicate_actual_write','no_duplicate_injection'),('missing_restore','three_prespecified_postchecks')]
    for mutation,expected_check in mutations:
        with tempfile.TemporaryDirectory(prefix='p12b-verifier-') as td:
            dest=Path(td)/'view'
            def copy(src,dst):
                if Path(src).suffix in ('.npz','.bin') or Path(src).name in ('b2-native','unit-tests'):os.link(src,dst)
                else:shutil.copy2(src,dst)
                return dst
            shutil.copytree(source,dest,copy_function=copy)
            if mutation=='metric_sign':
                report=read(dest/'report.json');report['per_site'][0]['voltage_interaction_mean_mv']+=1.;save(dest/'report.json',report)
            elif mutation=='missing_restore':
                post=read(dest/'postchecks.json');save(dest/'postchecks.json',post[:-1]);report=read(dest/'report.json');report['postchecks_sha256']=sha(dest/'postchecks.json');save(dest/'report.json',report)
            else:
                rows=read(dest/'rows.json');row=next(r for r in rows if r['background']==783 and r['condition']=='with_background' and r['site']==0 and r['kind']=='AB' and r['delay']==200);path=dest/'trials'/(row['key']+'.npz');encoded=load_npz(path)
                if mutation=='background_changed_per_order':encoded['background_ticks']=encoded['background_ticks']+1
                else:encoded['audit']=np.concatenate([encoded['audit'],encoded['audit'][:1]],axis=0)
                # Write to a new inode, so the original hardlinked archive stays intact.
                temp=path.with_name('mutation.npz');np.savez_compressed(temp,**encoded);os.replace(temp,path)
                decoded={k:v.copy() for k,v in encoded.items()};blank=load_npz(dest/'trials'/(row['blank_key']+'.npz'))
                for k in ('v','ge','gi'):decoded[k]=np.bitwise_xor(decoded[k],blank[k].view(np.uint64)).view(np.float64)
                row['archive_sha256']=sha(path);row['array_sha256']={k:digest(v) for k,v in decoded.items()};row['array_shapes']={k:list(v.shape) for k,v in decoded.items()};save(dest/'rows.json',rows)
                report=read(dest/'report.json');report['rows_sha256']=sha(dest/'rows.json');save(dest/'report.json',report)
            rejected=False
            with contextlib.redirect_stdout(io.StringIO()):
                try:verify(dest)
                except ValueError as e:
                    if str(e)!='P12b independent verification failed':raise
                    rejected=True
            verdict=read(dest/'verification.json')
            if not rejected or verdict['all_passed'] or verdict['checks'][expected_check]:raise AssertionError('mutation escaped semantic check: '+mutation)
            outcomes.append({'mutation':mutation,'rejected':True,'expected_check':expected_check,'failed_checks':[k for k,v in verdict['checks'].items() if not v]})
            print(outcomes[-1],flush=True)
    unchanged=original=={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    if not unchanged:raise AssertionError('mutation tests altered real evidence')
    result={'all_passed':True,'mutations':outcomes,'original_files_unchanged':True,'verifier_sha256':sha(Path(__file__).with_name('verify_background_clamp.py')),'test_source_sha256':sha(Path(__file__))}
    save(source/'verifier-mutation-tests.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();run(args.root)
