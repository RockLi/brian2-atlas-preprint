"""P12-only instrumentation of the frozen P11 native source; no legacy edits."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import numpy as np
from brian2_rust.results import load_results
from .motion_refinement import read, sha, load_npz
from .multispeed_data import digest
from .run_experiment import save

RUST = r'''
fn p12_replace(native:&[usize], injected:&[usize], selected:&[usize], mode:u64)->Vec<usize>{
    if mode!=1{return native.to_vec()}
    let mut events:Vec<usize>=native.iter().copied().filter(|s|selected.binary_search(s).is_err()).collect();
    events.extend_from_slice(injected);events.sort_unstable();events
}
fn p12_config()->Result<(u64,Vec<usize>,SpikeInput)>{
    let path=std::env::var("B2_P12_CONFIG")?;let bytes=fs::read(path)?;
    check(bytes.len()>=24 && &bytes[..8]==b"P12CFG01","bad P12 config")?;
    let mode=u64::from_le_bytes(bytes[8..16].try_into()?);let n=u64::from_le_bytes(bytes[16..24].try_into()?) as usize;
    check(mode<=2 && n>0 && bytes.len()==24+8*n,"P12 config size/mode")?;
    let selected:Vec<usize>=bytes[24..].chunks_exact(8).map(|v|u64::from_le_bytes(v.try_into().unwrap()) as usize).collect();
    check(selected.iter().all(|&s|s<139255) && selected.windows(2).all(|w|w[0]<w[1]),"P12 selected cells")?;
    let supplied=SpikeInput::load(1,Path::new(&std::env::var("B2_P12_EVENTS")?))?;supplied.validate(139255,0,6000)?;
    check(supplied.indices.iter().all(|s|selected.binary_search(s).is_ok()),"P12 unselected event")?;
    check(mode==1 || supplied.indices.is_empty(),"events require replacement mode")?;
    Ok((mode,selected,supplied))
}
#[cfg(test)] mod p12_tests {
 use super::*;
 #[test] fn replaces_without_double_delivery(){assert_eq!(p12_replace(&[1,2,4],&[1,4],&[1,4],1),vec![1,2,4]);assert_eq!(p12_replace(&[1,2,4],&[],&[1,4],1),vec![2]);}
 #[test] fn passive_and_cut_keep_native_queue(){assert_eq!(p12_replace(&[1,2,4],&[],&[1,4],0),vec![1,2,4]);assert_eq!(p12_replace(&[1,2,4],&[],&[1,4],2),vec![1,2,4]);}
 #[test] fn small_network_replay_delay_and_edges(){
  let mut q=vec![Vec::<usize>::new();19];let mut ledger=Vec::new();let edges=vec![vec![(2usize,3.0)],vec![(2usize,-2.0)]];let mut ge=0.;let mut gi=0.;
  for tick in 0..50 {let native=if tick==2{vec![0,1]}else{vec![]};let events=p12_replace(&native,&native,&[0,1],1);q[(tick+18)%19].extend(events);let due=std::mem::take(&mut q[tick%19]);
   for source in due{for &(target,w) in &edges[source]{ge+=0.5*w*f64::from(u8::from(w>0.));gi-=2.0*w*f64::from(u8::from(w<0.));ledger.push((tick,source,target));}}
  }
  assert_eq!(ledger,vec![(20,0,2),(20,1,2)]);assert_eq!(ge,1.5);assert_eq!(gi,4.0);
 }
}
'''


def patch_source(source):
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('native anchor not unique: '+old[:70])
        source=source.replace(old,new)
    source+='\n'+RUST
    replace('    let p1_dt = data.f64()?;', '''    if let Ok(path)=std::env::var("B2_P12_BACKGROUND") {
        let bg=SpikeInput::load(0,Path::new(&path))?;bg.validate(512,0,6000)?;
        p0_generated_ticks=bg.ticks;p0_generated_indices=bg.indices;
    }
    let (p12_mode,p12_selected,p12_input)=p12_config()?;
    let mut p12_cursor=0usize;let mut p12_audit:Vec<[u64;10]>=Vec::new();
    let p1_dt = data.f64()?;''')
    replace('    let mut p1_state_3 = data.f64_vec(139255)?;', '''    let mut p1_state_3 = data.f64_vec(139255)?;
    if p12_mode==2 {for &source in &p12_selected {p1_state_2[source]=0.;}}''')
    replace('                r1_queue[r1_delivery % r1_queue_size].extend_from_slice(&p1_fired);', '''                let first=p12_cursor;
                while p12_cursor<p12_input.ticks.len() && p12_input.ticks[p12_cursor]==p1_tick {p12_cursor+=1;}
                let p12_outgoing=p12_replace(&p1_fired,&p12_input.indices[first..p12_cursor],&p12_selected,p12_mode);
                r1_queue[r1_delivery % r1_queue_size].extend_from_slice(&p12_outgoing);''')
    replace('            if parallel.events_parallel(r1_event_count) {','            if false && parallel.events_parallel(r1_event_count) {')
    anchor='                        unsafe { *p1_state_1.get_unchecked_mut(target_state) = r11; }'
    replace(anchor,anchor+'''
                        if p12_selected.binary_search(&source).is_ok(){
                            p12_audit.push([p1_tick as u64,source as u64,edge as u64,target as u64,
                                r5.to_bits(),(-r10).to_bits(),input_1_1_15.to_bits(),p1_state_0[target_state].to_bits(),
                                input_1_1_16.to_bits(),p1_state_1[target_state].to_bits()]);
                        }''')
    replace('    fs::create_dir_all(output)?;', '''    fs::create_dir_all(output)?;
    let mut audit=BufWriter::new(File::create(output.join("p12-deliveries.bin"))?);
    for row in &p12_audit {for value in row {audit.write_all(&value.to_le_bytes())?;}}audit.flush()?;''')
    return source


def compile_engine(root,p11):
    native=root/'native';native.mkdir(exist_ok=False)
    old=p11/'native'/'main.rs';source=native/'main.rs';source.write_text(patch_source(old.read_text()))
    flags=['--edition=2021','-C','opt-level=3','-C','codegen-units=1','-C','panic=abort']
    subprocess.run(['rustc',*flags,str(source),'-o',str(native/'b2-native')],capture_output=True,text=True,check=True)
    subprocess.run(['rustc','--edition=2021','--test',str(source),'-o',str(native/'unit-tests')],capture_output=True,text=True,check=True)
    result=subprocess.run([str(native/'unit-tests'),'--nocapture'],capture_output=True,text=True,check=True)
    (native/'unit-tests.log').write_text(result.stdout+result.stderr)
    save(native/'identity.json',{'parent_source_sha256':sha(old),'source_sha256':sha(source),'binary_sha256':sha(native/'b2-native'),
        'engine_source_sha256':sha(Path(__file__)),'flags':flags,'rustc':subprocess.run(['rustc','--version'],capture_output=True,text=True,check=True).stdout.strip()})
    return native


def write_spikes(path,indices,ticks):
    indices=np.asarray(indices,dtype='<i8');ticks=np.asarray(ticks,dtype='<i8')
    if indices.shape!=ticks.shape or indices.ndim!=1:raise ValueError('bad schedule shape')
    pairs=list(zip(ticks.tolist(),indices.tolist()))
    if pairs!=sorted(set(pairs)):raise ValueError('schedule must be sorted unique')
    path.write_bytes(b'B2SPIK01'+struct.pack('<Q',len(ticks))+ticks.astype('<u8').tobytes()+indices.astype('<u8').tobytes())


def atomic_json(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


class Budget:
    def __init__(self,path):
        self.path=path
        if not path.exists():atomic_json(path,{'limit':300,'attempts':[]})
    def reserve(self,key,metadata):
        data=read(self.path)
        if len(data['attempts'])>=data['limit']:raise RuntimeError('300-run budget exhausted')
        if any(a['key']==key for a in data['attempts']):raise RuntimeError('attempt key already reserved; inspect existing attempt')
        number=len(data['attempts'])+1;data['attempts'].append({'number':number,'key':key,'status':'reserved','time_unix':time.time(),**metadata});atomic_json(self.path,data);return number
    def update(self,number,**fields):
        data=read(self.path);data['attempts'][number-1].update(fields);atomic_json(self.path,data)


class Runner:
    def __init__(self,root,p11,artifact,parent,budget):
        self.root=root;self.p11=p11;self.budget=budget;self.native=root/'native';self.binary=self.native/'b2-native';self.base=parent/'intact'/'base.bin'
        self.identity=read(self.native/'identity.json');self.base_hash=read(parent/'intact'/'identity.json')['base_sha256']
        self.model=read(artifact/'model.json');self.model['definition']=read(p11/'monitor-definition.json')
        self.cells=read(p11/'protocol.json')['monitor_cells'];self.ni=1
        (root/'work').mkdir(exist_ok=True)
    def run(self,key,selected,visual,events,mode=0,background=None):
        # Reserve before any subprocess. Failed/time-limited attempts remain counted.
        if sha(self.base)!=self.base_hash or sha(self.binary)!=self.identity['binary_sha256']:raise ValueError('immutable native identity changed')
        work=self.root/'work'/key;work.mkdir(exist_ok=False);vi,vt=visual;ei,et=events
        write_spikes(work/'visual.bin',vi,vt);write_spikes(work/'outgoing.bin',ei,et)
        selected=sorted(selected);(work/'config.bin').write_bytes(b'P12CFG01'+struct.pack('<QQ',mode,len(selected))+np.array(selected,dtype='<u8').tobytes())
        env=os.environ.copy();env.update({'B2_NUM_THREADS':'4','B2_THREAD_AFFINITY':'auto','B2_P12_CONFIG':str((work/'config.bin').resolve()),'B2_P12_EVENTS':str((work/'outgoing.bin').resolve())});env.pop('B2_P12_BACKGROUND',None)
        if background is not None:env['B2_P12_BACKGROUND']=str(background.resolve())
        number=self.budget.reserve(key,{'binary_sha256':self.identity['binary_sha256'],'base_sha256':self.base_hash,'mode':mode,'selected':selected,'visual_sha256':sha(work/'visual.bin'),'outgoing_sha256':sha(work/'outgoing.bin'),'background_sha256':sha(background) if background else None})
        start=time.perf_counter()
        try:
            proc=subprocess.Popen([str(self.binary),str(self.base),str(work/'result'),'--spike-input','2',str(work/'visual.bin')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.budget.update(number,status='running',pid=proc.pid)
            stdout,stderr=proc.communicate(timeout=300)
            (work/'stdout.log').write_text(stdout);(work/'stderr.log').write_text(stderr)
            if proc.returncode:raise RuntimeError(stderr or stdout)
            model=copy.deepcopy(self.model);model['instance']['populations'][2]['spike_generator']={'spike_indices':list(map(int,vi)),'spike_ticks':list(map(int,vt))}
            if background is not None:
                values=np.frombuffer(background.read_bytes()[16:],dtype='<u8');n=len(values)//2;model['instance']['populations'][0]['spike_generator']={'spike_ticks':values[:n].tolist(),'spike_indices':values[n:].tolist()}
            if mode==2:
                state=model['instance']['populations'][1]['initial_state']['transmission']
                for c in selected:state[c]='0000000000000000'
            result=load_results(model,work/'result');pop=result['populations'][1]
            audit=np.fromfile(work/'result'/'p12-deliveries.bin',dtype='<u8').reshape(-1,10)
            summary={'attempt':number,'seconds':time.perf_counter()-start,'events_sha256':digest(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8')),
                'states_sha256':{k:digest(v) for k,v in pop['states'].items()},'trace_sha256':{k:digest(v) for k,v in pop['trace'].items()},'audit_sha256':digest(audit)}
            self.budget.update(number,status='completed',seconds=summary['seconds'],result=summary)
            return result,audit,summary,work
        except BaseException as error:
            if 'proc' in locals() and proc.poll() is None:proc.kill();proc.communicate()
            self.budget.update(number,status='failed',error=str(error),seconds=time.perf_counter()-start);raise
