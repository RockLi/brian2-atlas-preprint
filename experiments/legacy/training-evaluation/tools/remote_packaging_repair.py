from pathlib import Path
import difflib,re
p=Path('build/brian-source/pyproject.toml')
old=p.read_text()
new=re.sub(r'fallback_version\s*=\s*[\'"]unknown[\'"]','fallback_version = "2.10.1.post729"',old)
assert old!=new,'No unknown fallback remains; inspect before retry'
Path('evidence/packaging-only.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozen/pyproject.toml',tofile='build-copy/pyproject.toml')))
p.write_text(new)
print('Only build-copy fallback version metadata changed; frozen source unchanged.')
