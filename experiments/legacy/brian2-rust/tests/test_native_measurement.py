"""Native RSS must not inherit a large caller's resident-memory high-water mark."""
import json
from pathlib import Path
import subprocess
import sys


MEASURE = Path(__file__).resolve().parents[1] / 'examples' / 'measure_native.py'


def test_fresh_supervisor_excludes_large_parent_and_propagates_failure(tmp_path):
    parent = '''
import subprocess, sys
resident = bytearray(128 * 1024 * 1024)
subprocess.run(sys.argv[1:], check=True)
'''
    usage, log = tmp_path/'usage.json', tmp_path/'native.log'
    command = [sys.executable, str(MEASURE), '--usage', str(usage), '--log', str(log)]
    subprocess.run([sys.executable, '-c', parent, *command, '--',
                    sys.executable, '-c', 'print("native-output")'], check=True)
    result = json.loads(usage.read_text())
    assert 0 < result['peak_rss_bytes'] < 96 * 1024 * 1024
    assert result['wall_seconds'] > 0
    assert log.read_text().strip() == 'native-output'
    failed = subprocess.run([*command, '--', sys.executable, '-c', 'raise SystemExit(7)'])
    assert failed.returncode == 7
    assert json.loads(usage.read_text())['returncode'] == 7
