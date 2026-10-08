"""Bounded local1750 terminal collection:900 s, sampled1.5 GiB RSS and kernel CPU/file caps."""
import argparse,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--seconds',type=int,default=900);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.command and a.seconds==900
assert Path(a.command[0]).resolve()==Path(__file__).with_name('mam_collect_confirmation_terminal.py').resolve()
assert a.command.count('--wall-seconds')==1 and a.command[a.command.index('--wall-seconds')+1]=='900'
assert not a.report.exists() and not a.log.exists()
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
start=time.monotonic();peak=0;reason=None
code="import resource,runpy,sys,os; resource.setrlimit(resource.RLIMIT_CPU,(120,120)); resource.setrlimit(resource.RLIMIT_FSIZE,(64*2**20,64*2**20)); sys.argv=sys.argv[1:]; sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0]))); runpy.run_path(sys.argv[0],run_name='__main__')"
with a.log.open('x') as stream:
 child=subprocess.Popen([sys.executable,'-c',code,*a.command],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 while child.poll() is None:
  try:
   stat=subprocess.run(['/bin/ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True,timeout=2)
   if stat.returncode==0 and stat.stdout.strip():peak=max(peak,int(stat.stdout.strip())*1024)
  except subprocess.TimeoutExpired:reason='own RSS observation timeout'
  if time.monotonic()-start>a.seconds:reason='wall limit'
  if peak>1536*2**20:reason='sampled RSS limit'
  if reason:
   os.killpg(child.pid,signal.SIGKILL);break
  time.sleep(.1)
 rc=child.wait()
r=dict(returncode=rc,stop_reason=reason,wall_seconds=time.monotonic()-start,sampled_peak_rss_bytes=peak,sampled_rss_limit_bytes=1536*2**20,rss_poll_seconds=.1,kernel_cpu_limit_seconds=120,kernel_file_limit_bytes=64*2**20,wall_limit_seconds=a.seconds,command=a.command,host='local',memory_limit_kind='sampled own-process RSS; not a kernel cgroup limit',openblas_threads=1,omp_threads=1)
a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));print(a.log.read_text()[-5000:])
if rc or reason:raise SystemExit(2)
