"""Read-only complete source and fixture identity verification."""
from pathlib import Path
import hashlib,json
import control
HERE=Path(__file__).resolve().parent
identity=json.loads((HERE/'source-identity.json').read_text())
expected={n:v['sha256'] for n,v in identity['files'].items()}
expected.update(json.loads((HERE/'test-fixture-stage.json').read_text())['files'])
code=f"""from pathlib import Path
import hashlib,json
b=Path({control.SOURCE!r});files={{{{name:hashlib.sha256((b/name).read_bytes()).hexdigest() for name in {list(expected)!r}}}}}
print(json.dumps({{{{'files':files,'reference_binary_sha256':hashlib.sha256((b/'target/release/b2-runner').read_bytes()).hexdigest()}}}}))"""
# Expand escaped braces in this saved literal to normal Python mappings.
code=code.replace('{{','{').replace('}}','}')
r=control.c.remote(control.c.NODES[0],code);assert r['files']==expected
control.c.record('engine-source-readback.json',dict(passed=True,files_checked=len(expected),source_archive_sha256=identity['archive_sha256'],**r))
