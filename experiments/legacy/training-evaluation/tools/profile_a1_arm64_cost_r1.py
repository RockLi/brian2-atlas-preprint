"""Remote-only diagnostic; independent from formal A1 evidence and scores."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, sys, time, subprocess, shutil, cProfile, pstats
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import socket,platform
 if socket.gethostname()!='rock-mac-studio-1.local' or platform.machine()!='arm64':raise RuntimeError('Authorized ARM64 remote only')
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 source=out/'rust-source';source.mkdir()
 frozen=ROOT/'snapshot/brian2-rust'
 for n in ['Cargo.toml','Cargo.lock']:shutil.copy2(frozen/n,source/n)
 shutil.copytree(frozen/'src',source/'src')
 original=(source/'src/train_main.rs').read_text()
 instrumented=original.replace('        let bytes = std::fs::read(&args[1])?;', '''        let phase_start=std::time::Instant::now();
        let bytes = std::fs::read(&args[1])?;
        eprintln!("ATLAS_PROFILE read_s={}",phase_start.elapsed().as_secs_f64());''')
 instrumented=instrumented.replace('        let request = serde_json::from_slice(&bytes)?;', '''        let phase_start=std::time::Instant::now();
        let request = serde_json::from_slice(&bytes)?;
        eprintln!("ATLAS_PROFILE decode_s={}",phase_start.elapsed().as_secs_f64());
        let phase_start=std::time::Instant::now();''')
 instrumented=instrumented.replace('        if mpi.as_ref().is_none_or', '''        eprintln!("ATLAS_PROFILE execute_s={}",phase_start.elapsed().as_secs_f64());
        if mpi.as_ref().is_none_or''')
 instrumented=instrumented.replace('            std::fs::write(&args[2], serde_json::to_vec(&result)?)?;', '''            let phase_start=std::time::Instant::now();
            let encoded=serde_json::to_vec(&result)?;
            eprintln!("ATLAS_PROFILE encode_s={}",phase_start.elapsed().as_secs_f64());
            let phase_start=std::time::Instant::now();
            std::fs::write(&args[2], encoded)?;
            eprintln!("ATLAS_PROFILE write_s={}",phase_start.elapsed().as_secs_f64());''')
 assert instrumented!=original
 (source/'src/train_main.rs').write_text(instrumented)
 rust='/atlas-home/0004/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/'
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('CARGO_') or key.startswith('RUST'):env.pop(key)
 env.update(RUSTC=rust+'rustc',RUSTDOC=rust+'rustdoc',RUSTUP_TOOLCHAIN='1.98.1-aarch64-apple-darwin',CARGO_HOME=str(ROOT/'cache/cargo-home'),CARGO_NET_OFFLINE='true')
 cmd=[rust+'cargo','build','--manifest-path',str(source/'Cargo.toml'),'--target','aarch64-apple-darwin','--locked','--offline','--release','--bin','b2-train','-j4','--target-dir',str(out/'build')]
 with (out/'compile.log').open('x') as log:
  q=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 if q.returncode:raise RuntimeError('Diagnostic instrumented build failed')
 sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'adapters'));sys.path.insert(0,str(ROOT/'snapshot'))
 from arm64_artifact_gate_r1 import gate_artifact
 from run_a1_mnist_arm64_r1 import Atlas,encode
 from prepare_data import read_idx
 from atlas_adapter import admission
 import numpy as np,copy
 runner=out/'build/aarch64-apple-darwin/release/b2-train';identity=gate_artifact(runner,'runner')
 baseline=gate_artifact(ROOT/'runtime/arm64-r2/b2-train','runner','1dc6fac642e1829a942105ec6288fe24e5632f66ca54cf12a671258f6014e5a4')
 arrays=np.load(ROOT/'fixtures/a1-arm64-r1/seed-11.npz')
 ids=arrays['epoch_indices'][0,:32]
 images=read_idx(ROOT/'data/mnist/raw/train-images-idx3-ubyte',(60000,28,28));labels=read_idx(ROOT/'data/mnist/raw/train-labels-idx1-ubyte',(60000,))
 t=time.perf_counter();x=encode(images,ids);y=np.asarray(labels[ids],dtype=np.int64);encoding=time.perf_counter()-t
 m=Atlas(ROOT,[arrays['bank_0'],arrays['bank_1']],11);state=copy.deepcopy(m.trainer.state);initial=np.zeros((32,138),dtype=np.float64)
 t=time.perf_counter();check=admission(m.plan,state,x,y,operation='train',initial=initial);admission_s=time.perf_counter()-t
 assert check['status']=='admitted'
 request=dict(plan=m.plan,state=state,operation='train',inputs=x.tolist(),labels=y.tolist(),initial=initial.tolist())
 profiler=cProfile.Profile();t=time.perf_counter()
 actual=profiler.runcall(m.trainer.execute,x,y,operation='train',initial=initial)
 public_s=time.perf_counter()-t;profiler.dump_stats(str(out/'python-public-api.prof'))
 from brian2_rust.protocol import canonical_bytes
 t=time.perf_counter();payload=canonical_bytes(request);serialize_s=time.perf_counter()-t
 p=out/'request.json';p.write_bytes(payload);result=out/'instrumented-result.json'
 t=time.perf_counter();ran=subprocess.run([str(runner),str(p),str(result)],capture_output=True,text=True,timeout=120);native_wall=time.perf_counter()-t
 (out/'native-stderr.txt').write_text(ran.stderr);assert ran.returncode==0,ran.stderr
 t=time.perf_counter();other=json.loads(result.read_bytes());parse_s=time.perf_counter()-t
 assert actual==other,'Instrumented runner numerics differ from frozen runner'
 phases={line.split('=',1)[0].split()[-1]:float(line.split('=',1)[1]) for line in ran.stderr.splitlines() if line.startswith('ATLAS_PROFILE')}
 stats=pstats.Stats(profiler);top=sorted([dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_s=v[2],cumulative_s=v[3]) for k,v in stats.stats.items()],key=lambda v:v['cumulative_s'],reverse=True)[:25]
 batches=[]
 for seed in [11,23,37,51,71]:
  p=ROOT/f'evidence/a1-queue-arm64-r1/atlas-seed-{seed}/batches.jsonl'
  rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  train=[z for z in rows if z['phase']=='train']
  import statistics
  batches.append(dict(seed=seed,batches=len(train),samples=sum(z['samples'] for z in train),full_epoch_denominator=55000,median_batch_s=statistics.median(z['elapsed_s'] for z in train),median_public_api_s=statistics.median(z['public_api_s'] for z in train)))
 report=dict(status='completed',diagnostic_only=True,formal_scores_modified=False,held_out_opened=False,shape=[32,100,784],baseline_runner=baseline,instrumented_runner=identity,numeric_results_exactly_equal=True,encoding_s=encoding,admission_s=admission_s,profiled_public_api_s=public_s,standalone_python_serialization_s=serialize_s,native_cli_wall_s=native_wall,native_phases=phases,python_result_parse_s=parse_s,request_bytes=len(payload),result_bytes=result.stat().st_size,python_profile_top=top,formal_censored_batch_summaries=batches,instrumentation='Only diagnostic main timing markers; all library source bytes match frozen snapshot',timing_limits='One diagnostic full-shape batch; Python cProfile adds overhead; no cross-engine speed ratio',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['python_profile_top','baseline_runner','instrumented_runner']},indent=2))
if __name__=='__main__':main()

