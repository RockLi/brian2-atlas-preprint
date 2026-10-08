"""Frozen-model input replacement with byte-for-byte snapshot verification.

Only a sorted SpikeGenerator schedule may change. The ordinary compatibility
validator runs once when opening the artifact; every sample verifies the frozen
snapshot and executable, then sends a small, natively validated event sidecar.
No Definition/Run/graph/initial-state changes are admitted by this interface.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
from ..protocol import file_sha


class FrozenCPU:
    numeric_profile = 'rustc-aot-f64'

    def __init__(self, template, native, directory, *, population='pixels', threads=1):
        from brian2_rust import write_compatible_instance
        from brian2_rust.native import write_streamed_instance
        self._model=copy.deepcopy(template)
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=False)
        self.native=Path(native).resolve();self.threads=threads
        self.population=next(i for i,p in enumerate(template['definition']['populations']) if p['name']==population)
        pop=template['definition']['populations'][self.population]
        self.count=pop['count'];self.steps=pop['steps']
        if 0. != struct.unpack('>d',bytes.fromhex(template['run']['start']))[0]:
            raise ValueError('frozen MNIST input requires a tick-zero initial snapshot')
        if template['instance']['populations'][self.population]['spike_generator'] is None:
            raise ValueError('selected population is not a SpikeGenerator')
        self.base=self.directory/'base.bin'
        verified=write_compatible_instance(template,self.native,self.base)
        layout={}
        actual=write_streamed_instance(template,self.base,schedule_layout=layout)
        if actual!=verified['instance_sha256']:
            raise ValueError('streamed snapshot differs from validated artifact')
        self.base_hash=actual;self.start,self.stop=layout[self.population]
        self.binary=self.native/'b2-native';self.binary_hash=file_sha(self.binary)
        (self.directory/'identity.json').write_text(json.dumps({**verified,'base_sha256':actual,
            'binary_sha256':self.binary_hash,'mutable_population':population,
            'schedule_byte_range':[self.start,self.stop]},indent=2)+'\n')

    def validate_schedule(self, indices, ticks):
        indices,ticks=np.asarray(indices),np.asarray(ticks)
        if (indices.ndim!=1 or ticks.ndim!=1 or indices.shape!=ticks.shape
                or indices.dtype.kind not in 'iu' or ticks.dtype.kind not in 'iu'
                or np.any(indices<0) or np.any(indices>=self.count)
                or np.any(ticks<0) or np.any(ticks>=self.steps)):
            raise ValueError('invalid frozen spike schedule')
        if len(ticks)>1 and np.any((ticks[1:]<ticks[:-1]) |
                                  ((ticks[1:]==ticks[:-1]) & (indices[1:]<=indices[:-1]))):
            raise ValueError('spikes must be sorted by tick/index and unique per bin')
        return indices,ticks

    def write(self, indices, ticks, path):
        """Diagnostic full-instance writer, used to prove sidecar equivalence."""
        indices,ticks=self.validate_schedule(indices,ticks)
        checksum=hashlib.sha256()
        with self.base.open('rb') as source, Path(path).open('xb') as out:
            remaining=self.start
            while remaining:
                block=source.read(min(1024**2,remaining))
                if not block:raise ValueError('truncated frozen snapshot')
                checksum.update(block);out.write(block);remaining-=len(block)
            old=source.read(self.stop-self.start);checksum.update(old)
            out.write(struct.pack('<Q',len(ticks)))
            out.write(ticks.astype('<u8').tobytes());out.write(indices.astype('<u8').tobytes())
            while block:=source.read(1024**2):checksum.update(block);out.write(block)
        if checksum.hexdigest()!=self.base_hash:
            Path(path).unlink()
            raise ValueError('frozen snapshot changed')

    def run(self, indices, ticks, key):
        from brian2_rust.results import load_results
        if file_sha(self.binary)!=self.binary_hash:raise ValueError('AOT executable changed')
        started=time.perf_counter();path=self.directory/f'{key}.spikes';out=self.directory/key
        if file_sha(self.base)!=self.base_hash:raise ValueError('frozen snapshot changed')
        indices,ticks=self.validate_schedule(indices,ticks)
        with path.open('xb') as stream:
            stream.write(b'B2SPIK01'+struct.pack('<Q',len(ticks)))
            stream.write(ticks.astype('<u8').tobytes());stream.write(indices.astype('<u8').tobytes())
        env=os.environ.copy();env['B2_NUM_THREADS']=str(self.threads);env['B2_THREAD_AFFINITY']='auto'
        p=subprocess.run([str(self.binary),str(self.base),str(out),'--spike-input',str(self.population),str(path)],
                         env=env,capture_output=True,text=True,timeout=300)
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
        model=copy.copy(self._model)
        model['instance']=copy.copy(self._model['instance'])
        model['instance']['populations']=list(self._model['instance']['populations'])
        model['instance']['populations'][self.population]=dict(
            self._model['instance']['populations'][self.population],
            spike_generator={'spike_indices':indices.tolist(),'spike_ticks':ticks.tolist()})
        result=load_results(model,out);result['wall_seconds']=time.perf_counter()-started
        return result

    def close(self):pass
