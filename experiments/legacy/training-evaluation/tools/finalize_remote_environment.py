"""Use the complete frozen source tree with extensions built on this host.

The first packaging attempt omitted runtime templates from its wheel. We keep
that failure and use the already hashed source templates, not a patched model.
"""
from pathlib import Path
import sys,sysconfig,hashlib,json,shutil,platform,datetime
ROOT=Path(__file__).resolve().parents[1]
site=Path(sysconfig.get_paths()['purelib']);extensions=[]
for directory in ['memory','synapses']:
    for source in (site/'brian2'/directory).glob('*.so'):
        dest=ROOT/'snapshot/brian2'/directory/source.name
        if dest.exists():assert dest.read_bytes()==source.read_bytes()
        else:shutil.copy2(source,dest)
        extensions.append(dict(source=str(source),destination=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
assert len(extensions)==2,extensions
sys.path.insert(0,str(ROOT/'snapshot'))
import brian2,torch,numpy,psutil
assert Path(brian2.__file__).resolve().is_relative_to(ROOT/'snapshot')
manifest=json.loads((ROOT/'sources/snapshot-manifest.json').read_text())
bad=[p for p,h in manifest['source_hashes'].items() if hashlib.sha256((ROOT/'snapshot'/p).read_bytes()).hexdigest()!=h]
assert not bad,bad
x=dict(platform=platform.platform(),python=sys.version,torch=torch.__version__,numpy=numpy.__version__,brian2=brian2.__version__,brian2_path=brian2.__file__,cuda=torch.cuda.is_available(),mps=torch.backends.mps.is_available(),ram_bytes=psutil.virtual_memory().total,logical_cpu=psutil.cpu_count(),disk=psutil.disk_usage(ROOT)._asdict(),runtime_sha256=hashlib.sha256((ROOT/'runtime/b2-train').read_bytes()).hexdigest(),source_tree_sha256=manifest['source_tree_sha256'],extensions=extensions,source_integrity=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_template_policy='complete immutable snapshot with separately hashed host-compiled extension modules')
out=ROOT/'environment/hardware.json'
with out.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
with (ROOT/'evidence/bootstrap-complete.json').open('x') as f:json.dump(dict(completed=True,hardware=x),f,indent=2);f.write('\n')
print(json.dumps(x,indent=2))
