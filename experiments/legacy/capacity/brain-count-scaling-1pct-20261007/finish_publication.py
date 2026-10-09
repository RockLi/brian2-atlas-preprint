"""Update paper artifacts only after the complete, audited endpoint exists.

This observer starts no remote work. PDF layout still requires visual review.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAPER = REPO / 'docs/preprint/scripts'
BUNDLED_PYTHON = Path('/atlas-home/0004/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3')


def main():
    deadline = time.monotonic() + 4 * 3600
    previous = None
    while True:
        state = json.loads((HERE / 'sweep-state.json').read_text())
        marker = (state['status'], state.get('action'), state.get('case'))
        if marker != previous:
            print(json.dumps({'observed_state': marker, 'accepted_hosts': state['accepted_hosts']}), flush=True)
            previous = marker
        if state['status'] == 'endpoint_complete':
            assert state['accepted_hosts'] == [3, 6, 12, 24, 30]
            break
        assert state['status'] != 'stopped_on_failure', state.get('error')
        assert time.monotonic() < deadline, 'Publication observer deadline; no manuscript update performed.'
        time.sleep(35)
    steps = [
        (sys.executable, HERE / 'verify_study.py'),
        (sys.executable, HERE / 'report_scaling.py'),
        (sys.executable, PAPER / 'collect_scaling_evidence.py'),
        (sys.executable, PAPER / 'update_scaling_manuscript.py'),
        (sys.executable, PAPER / 'build_manuscript.py'),
        (str(BUNDLED_PYTHON), PAPER / 'export_pdf.py'),
    ]
    for python, script in steps:
        print(json.dumps({'step': script.name}), flush=True)
        subprocess.run([python, str(script)], cwd=REPO, check=True)
    print(json.dumps({'artifacts_generated': True, 'visual_pdf_review': 'pending'}), flush=True)


if __name__ == '__main__':
    main()
