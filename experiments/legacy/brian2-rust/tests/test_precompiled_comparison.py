"""Checks on the precompiled benchmark's compilation boundary."""
from pathlib import Path
import subprocess
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from gpu_precompiled import forbid_compilation,artifact_hashes


def test_compilation_guard_rejects_nested_compiler_and_restores_process_api():
    original=subprocess.Popen
    with pytest.raises(RuntimeError,match='Compilation attempted'):
        with forbid_compilation():subprocess.run(['sh','-c','make'],check=True)
    assert subprocess.Popen is original
    with forbid_compilation():
        result=subprocess.run([sys.executable,'-c','print(17)'],text=True,capture_output=True,check=True)
    assert result.stdout.strip()=='17' and subprocess.Popen is original


def test_artifact_identity_detects_changed_binary_but_not_new_output(tmp_path):
    binary=tmp_path/'main';binary.write_bytes(b'compiled')
    before=artifact_hashes(tmp_path);assert 'main' in before
    (tmp_path/'results.bin').write_bytes(b'output')
    assert artifact_hashes(tmp_path)==before
    binary.write_bytes(b'changed');assert artifact_hashes(tmp_path)!=before
