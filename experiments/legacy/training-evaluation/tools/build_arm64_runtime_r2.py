#!/usr/bin/env python3
"""Prepared ARM64 runtime build. No model execution; explicit coordinated --allow-build required."""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'evidence/runtime-arm64-preparation-r2/build-contract.json'
OUTPUT=ROOT/'evidence/runtime-arm64-build-r2'
BUILD=ROOT/'build/runtime_arm64_r2'
RUNTIME=ROOT/'runtime/arm64-r2'
LOCK=ROOT/'evidence/.arm64-runtime-build.lock'
CARGO_CACHE=ROOT/'cache/cargo-home'
TOOLCHAIN=Path('/atlas-home/0004/.rustup/toolchains/1.98.1-aarch64-apple-darwin')
TARGET='aarch64-apple-darwin'
FLOOR=50*1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value,update=False):
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if update:
        temp=path.with_suffix('.partial');temp.write_text(text);temp.replace(path)
    else:
        with path.open('x') as stream:stream.write(text)


def source_paths():
    base=ROOT/'snapshot/brian2-rust'
    return sorted([base/'Cargo.toml',base/'Cargo.lock']+[p for p in (base/'src').rglob('*') if p.is_file()])


def source_receipt():
    return {str(p.relative_to(ROOT)):sha(p) for p in source_paths()}


def verify_sources(contract):
    receipt=source_receipt()
    if receipt!=contract['source_identities']:
        raise RuntimeError('Frozen Rust source/Cargo lock identity changed')
    if sha(ROOT/'runtime/b2-train')!=contract['original_runtime_sha256']:
        raise RuntimeError('Original runtime changed')
    return receipt


def check_disk():
    free=shutil.disk_usage(ROOT).free
    if free<FLOOR:raise RuntimeError('50GiB free disk floor not met')
    return free


def arm64_only(file_output,lipo_output):
    return 'Mach-O 64-bit executable arm64' in file_output and lipo_output.split()==['arm64']


def build_command():
    return [str(TOOLCHAIN/'bin/cargo'),'build','--manifest-path',str(ROOT/'snapshot/brian2-rust/Cargo.toml'),
        '--target',TARGET,'--locked','--offline','--release','--bin','b2-train','-j4',
        '--target-dir',str(BUILD)]


def build_environment():
    env=os.environ.copy()
    for key in list(env):
        if key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER',
                   'RUSTC','RUSTDOC','CARGO_BUILD_TARGET','CARGO_TARGET_DIR','RUSTUP_TOOLCHAIN',
                   'RUSTUP_HOME','CARGO_HOME','CC','CXX','ARCHFLAGS','SDKROOT','DEVELOPER_DIR','MACOSX_DEPLOYMENT_TARGET') or key.startswith('CARGO_'):
            env.pop(key,None)
    env.update(RUSTUP_TOOLCHAIN='1.98.1-aarch64-apple-darwin',
        RUSTC=str(TOOLCHAIN/'bin/rustc'),RUSTDOC=str(TOOLCHAIN/'bin/rustdoc'),
        CARGO_HOME=str(CARGO_CACHE),CARGO_NET_OFFLINE='true',CARGO_BUILD_JOBS='4',
        CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER='/usr/bin/clang',
        TMPDIR=str(OUTPUT/'tmp'))
    return env


@contextmanager
def exclusive():
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another ARM64 build holds family lock') from None
        stream.seek(0);stream.truncate();stream.write(str(os.getpid()));stream.flush()
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def metadata(command,env):
    p=subprocess.run(command,capture_output=True,text=True,timeout=30,env=env,cwd=ROOT)
    row=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    if p.returncode:
        raise RuntimeError('Metadata command failed: '+json.dumps(row))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-build',action='store_true')
    args=parser.parse_args()
    if not args.allow_build:parser.error('Prepared only; coordinated --allow-build is required')
    if socket.gethostname()!='rock-mac-studio-1.local':
        raise RuntimeError('Remote host 100.90.28.27 only')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    report=dict(schema='runtime-arm64-build-r2',status='not_completed',performance_run=False,
        model_execution=False,original_runtime_preserved=True,source=str(CONTRACT.relative_to(ROOT)),
        planned_command=build_command(),maximum_compile_s=900,outer_recommended_cap_s=1200,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    contract=None
    try:
        with exclusive():
            sys.path.insert(0,str(ROOT/'tools'))
            from measure_e1_small_cold_r2 import require_idle
            from run_recurrent_queue import launch
            require_idle()
            contract=json.loads(CONTRACT.read_text())
            if sha(__file__)!=contract['driver_sha256']:
                raise RuntimeError('Build driver changed after preparation')
            for name,expected in contract['helper_identities'].items():
                if sha(ROOT/name)!=expected:raise RuntimeError('Build helper changed: '+name)
            report['sources_before']=verify_sources(contract)
            report['free_bytes_before']=check_disk()
            if BUILD.exists() or RUNTIME.exists():
                raise FileExistsError('Fresh build/runtime output required; existing artifacts are never replaced')
            BUILD.parent.mkdir(parents=True,exist_ok=True)
            (OUTPUT/'tmp').mkdir()
            if not CARGO_CACHE.is_dir():
                raise FileNotFoundError('Evaluation-private offline Cargo cache is absent')
            recorded_keys=['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','RUSTC','CARGO_HOME','CARGO_TARGET_DIR','MACOSX_DEPLOYMENT_TARGET']
            report['inherited_build_environment']={key:os.environ.get(key) for key in recorded_keys}
            report['cleared_cargo_environment_names']=[key for key in os.environ if key.startswith('CARGO_')]
            env=build_environment()
            report['environment_overrides']={k:env[k] for k in [
                'RUSTUP_TOOLCHAIN','RUSTC','RUSTDOC','CARGO_HOME','CARGO_NET_OFFLINE','CARGO_BUILD_JOBS',
                'CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER','TMPDIR']}
            report['metadata']={}
            for name,command in [
                ('rustc',[str(TOOLCHAIN/'bin/rustc'),'-vV']),
                ('cargo',[str(TOOLCHAIN/'bin/cargo'),'-Vv']),
                ('rustc_file',['/usr/bin/file','-L',str(TOOLCHAIN/'bin/rustc')]),
                ('cargo_file',['/usr/bin/file','-L',str(TOOLCHAIN/'bin/cargo')]),
                ('installed_targets',['/atlas-home/0004/.cargo/bin/rustup','target','list','--installed','--toolchain','1.98.1-aarch64-apple-darwin']),
                ('clang_arm64',['/usr/bin/arch','-arm64','/usr/bin/clang','--version']),
                ('sdk',['/usr/bin/xcrun','--sdk','macosx','--show-sdk-path'])]:
                report['metadata'][name]=metadata(command,env)
            for name in ('rustc','cargo'):
                text=report['metadata'][name]['stdout']
                if 'host: aarch64-apple-darwin' not in text or '1.98.1' not in text:
                    raise RuntimeError('Expected ARM64 Rust/Cargo1.98.1 toolchain')
                if 'Mach-O 64-bit executable arm64' not in report['metadata'][name+'_file']['stdout']:
                    raise RuntimeError('Compiler tool is not ARM64')
            if TARGET not in report['metadata']['installed_targets']['stdout'].split():
                raise RuntimeError('ARM64 target missing; no install permitted')
            report['tool_binary_sha256']={name:sha(TOOLCHAIN/'bin'/name) for name in ['rustc','cargo']}
            report['offline_cache_policy']='Use existing evaluation-private ROOT/cache/cargo-home without network; missing dependencies fail and remain recorded. Never install a toolchain or target.'
            report['cargo_config_hashes']={}
            for p in [CARGO_CACHE/'config',CARGO_CACHE/'config.toml',
                      ROOT/'.cargo/config',ROOT/'.cargo/config.toml']:
                if p.exists():report['cargo_config_hashes'][str(p)]=sha(p)
            verify_sources(contract);require_idle();report['free_bytes_before_compile']=check_disk()
            report['status']='compiling';save(OUTPUT/'progress.json',report,update=True)
            result=launch('cargo-build',build_command(),900,64*1024**3,OUTPUT,BUILD,env,'arm64_runtime_build')
            report['supervisor']=result
            report['sources_after']=verify_sources(contract)
            if result['exit_code']!=0 or result['termination_reason']!='exited':
                raise RuntimeError('Offline ARM64 build failed; inspect cargo-build.log and supervisor')
            artifact=BUILD/TARGET/'release/b2-train'
            if not artifact.is_file():raise FileNotFoundError('Cargo did not produce target b2-train')
            report['artifact_metadata']={name:metadata(command,env) for name,command in [
                ('file',['/usr/bin/file','-L',str(artifact)]),
                ('lipo',['/usr/bin/lipo','-archs',str(artifact)]),
                ('otool_header',['/usr/bin/otool','-hv',str(artifact)]),
                ('otool_libraries',['/usr/bin/otool','-L',str(artifact)]),
                ('otool_load_commands',['/usr/bin/otool','-l',str(artifact)])]}
            if not arm64_only(report['artifact_metadata']['file']['stdout'],report['artifact_metadata']['lipo']['stdout']):
                raise RuntimeError('Build artifact must be single architecture ARM64')
            report['free_bytes_before_publish']=check_disk()
            verify_sources(contract)
            RUNTIME.mkdir(parents=True,exist_ok=False)
            output=RUNTIME/'b2-train'
            with artifact.open('rb') as src,output.open('xb') as dst:shutil.copyfileobj(src,dst)
            output.chmod(0o755)
            if sha(output)!=sha(artifact):raise RuntimeError('Runtime publication copy hash mismatch')
            report.update(status='built_unqualified',runtime_path=str(output.relative_to(ROOT)),
                runtime_sha256=sha(output),runtime_architecture='arm64-only',
                original_runtime_sha256=sha(ROOT/'runtime/b2-train'),
                qualification='New profile only: no Q0, semantic, CPU performance or Metal result inferred')
    except BaseException as error:
        report.update(status='build_or_preflight_failed',error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc())
    finally:
        if contract is not None:
            try:
                report['sources_final']=verify_sources(contract)
                report['source_integrity_after_exit']=True
            except BaseException as error:
                report.update(status='source_integrity_failed',source_integrity_after_exit=False,
                    source_integrity_error=repr(error))
        report['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(OUTPUT/'terminal.json',report)
    print(json.dumps({k:report.get(k) for k in ['status','runtime_path','runtime_sha256','source_integrity_after_exit']}))
    return 0 if report['status']=='built_unqualified' else 1


if __name__=='__main__':
    raise SystemExit(main())
