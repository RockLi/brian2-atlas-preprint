"""Independent ARM64 E1-small: seven views, five fresh seeds, original workers.

Prepared entry only. --allow-run belongs in the single serial outer coordinator.
No waiter, compiler, installer, fixture generation or retry is provided here.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
from arm64_artifact_gate_r1 import gate_artifact

ROOT=Path(__file__).resolve().parents[1]
HOST='rock-mac-studio-1.local'
RUN=Path('evidence/arm64-r2/e1-small')
RUNTIME=Path('runtime/arm64-r2/b2-train')
PRODUCER=Path('evidence/runtime-arm64-build-r2/terminal.json')
BUILD_CONTRACT=Path('evidence/runtime-arm64-preparation-r2/build-contract.json')
ATLAS_Q0=Path('evidence/arm64-r2/q0/atlas/report.json')
SEEDS=[11,23,37,51,71]
VIEWS=[('atlas','cpu','atlas',[]),('sj-layerwise','cpu','spikingjelly_frontier',['--layerwise']),
       ('snn-layerwise','cpu','snntorch_fp64',['--layerwise']),('spyx','jax','spyx',[]),
       ('brainstate','jax','brainx_state',[]),('sj-compile','cpu','spikingjelly_frontier',['--layerwise','--compile']),
       ('snn-compile','cpu','snntorch_fp64',['--layerwise','--compile'])]
Q0_CASES=['base_negative_count_input','batch_duplicate','initial_threshold_boundary','single_sample_no_carry']
POLICY=dict(warmup_steps=10,measured_steps=50,per_seed_s=360,per_view_s=1800,
    finite_slots=35,outer_recommended_cap_s=13000,validator_recommended_cap_s=600,
    rss_guard_bytes=64*1024**3,free_disk_floor_bytes=50*1024**3,
    budget_scope='Per-slot wall includes gate rechecks, imports/oracle/qualification/JIT/training/I/O and cleanup; shared initial preflight recorded separately inside outer cap',
    resource_note='Atlas/Torch requested one compute thread; no core affinity. JAX pool is not single-thread qualified. RSS includes setup/diagnostics/caches, not pure training capacity.',
    numerical_scope='Unchanged original E1 workers: Atlas full-shape one Adam plus independent tiny dense Q0 three Adam; competitors actual timing path three Adam. No new full-shape three-Adam Atlas claim.',
    no_old_seed_splicing=True)
FIXED={'adapters/atlas_adapter.py': '89cdb5ac9db93e0d9110cac1782dc4720c8698b460ddd2c5cd7aa9290ee6c7ed', 'adapters/jax_adapter.py': '286cfa6df151d8ae3d05a937c0e9ca4545736db2434308da7c0ed02378624831', 'adapters/oracle.py': '80dd94c5be77bdf1857c2f22a2c7853eec871f47a1bee7caab2f946311d72bdb', 'adapters/torch_adapter.py': '8c49b9718578885aa14b3a539a665a3bf00d3555bb30b9d6771a3d38494b6f48', 'environment/cpu-lock.txt': 'd0b31ade6227e70132c886c420dd6a355378b5fc81588bdf34828de0ede1fce4', 'environment/hardware.json': 'f95a16795042edbb9b9f29f857875d6452bb0a42492b0834152ba1ba34e58018', 'environment/jax-lock.txt': '52eb289987704af80b6e59f96b5a91884e48ec71a68d8cfc9c8b68632470e9c3', 'evidence/remote-qualify-v1/freeze.json': 'a50a9abb7f9697b050c2e10ce30e77162e374a89b892f4db69f1aff162acec53', 'evidence/remote-qualify-v1/q0-brainstate-terminal.json': '4781ddcba290ca6305719cd554e27947b6a09a7c0b12b11c3e857726f1ad5d9d', 'evidence/remote-qualify-v1/q0-brainstate/base_negative_count_input.json': '98c89d0eae8949ba5f3c6d7760509ae9bdac6bf7a405495c0a5ebe1f7d071baa', 'evidence/remote-qualify-v1/q0-brainstate/batch_duplicate.json': '483340057621ff00585a94fad6287919e72cb35a18462450bca1aad131d42605', 'evidence/remote-qualify-v1/q0-brainstate/initial_threshold_boundary.json': '94715b02ff853a0a583b15c0e290580a852b685ff2b038702adacb95878d1672', 'evidence/remote-qualify-v1/q0-brainstate/report.json': '01267a7353a26a7087e49c16dab154c5195a7ebfc95c3300411d768af6ad5664', 'evidence/remote-qualify-v1/q0-brainstate/single_sample_no_carry.json': '723b7dcd9d0b5f6b9ff7921ea5c44f86c2ae29367b9893b04c5b631208853ca5', 'evidence/remote-qualify-v1/q0-sj-compile-terminal.json': '4310441a6d29d4a4c831d5e7d99a6acb25a9c778b5237431c74c0a0d39cfdb55', 'evidence/remote-qualify-v1/q0-sj-compile/base_negative_count_input.json': '09a13f727b5c76463b0fa10ad2cc797911ff45d4c2a3cc4493c0994196e8b7bb', 'evidence/remote-qualify-v1/q0-sj-compile/batch_duplicate.json': '401d398218fa65e60793c4bfc4b38b68b27dc511405f8d1cd223a30f10fd103d', 'evidence/remote-qualify-v1/q0-sj-compile/initial_threshold_boundary.json': '0dcbc4d9ccc8e31682bfedbd00254daef9668afcfa963e44851fe1b0bd25f1b4', 'evidence/remote-qualify-v1/q0-sj-compile/report.json': '51c70866b6055e9793b95625e9fb2afbf13ed4c30aea026ec081c5fb87e7d57b', 'evidence/remote-qualify-v1/q0-sj-compile/single_sample_no_carry.json': '9724f58dc0cc04d4944e9803a84001990572f4b3dba5f32ed92806b77efeb544', 'evidence/remote-qualify-v1/q0-sj-layerwise-terminal.json': 'aea06d9bd1dae3cc6e456f7e4ad49dee63e27ff457329030f212450671ef92b8', 'evidence/remote-qualify-v1/q0-sj-layerwise/base_negative_count_input.json': '12581d29bc9b911e8e7f4b7e80440043bda24fca3daad5b2b1657431283bb5dd', 'evidence/remote-qualify-v1/q0-sj-layerwise/batch_duplicate.json': '21fd01bc7b8779989a3bcd1520b2d8afa93c301377a2b2a5a36e3fff173e12b5', 'evidence/remote-qualify-v1/q0-sj-layerwise/initial_threshold_boundary.json': '038cfa0b87e24666b0d4610944167060a32db69eb7b6dd5849aeaebd19f46a83', 'evidence/remote-qualify-v1/q0-sj-layerwise/report.json': '27d572b5ecc148d39403d50da7b9e8c9abe4b0bb18ce8b0c790a9546ebd7397b', 'evidence/remote-qualify-v1/q0-sj-layerwise/single_sample_no_carry.json': '525a96e0093e9654a260465dfe8143dfd76e3605c396de7e28ef5735a9914d27', 'evidence/remote-qualify-v1/q0-snn-compile-terminal.json': '8c70ec22bff1710c76770769c3eadbaef86c9a565a14187157577b26cc6dfeab', 'evidence/remote-qualify-v1/q0-snn-compile/base_negative_count_input.json': 'a19f27b753abc5203a32b4b5c31ccf548f2bd3ddae95731928dc6aa99a369e1e', 'evidence/remote-qualify-v1/q0-snn-compile/batch_duplicate.json': '33d5772935e1abcb7445f0acc015543fce42f7befc5b78d728c91f1d8756beb2', 'evidence/remote-qualify-v1/q0-snn-compile/initial_threshold_boundary.json': '69cb669e71c5c095bb6b21fe578ab83736057bda825e82fa4164f926c096e967', 'evidence/remote-qualify-v1/q0-snn-compile/report.json': '133c2e8a6b8d911e3dcfc2e7c61e2029cf6e0ffec527a83ccf0ec08eb7cabf05', 'evidence/remote-qualify-v1/q0-snn-compile/single_sample_no_carry.json': 'f699d9e140218cb132f95995de4d8d2f141a02afaf39a83cc7fb6c69ec18905f', 'evidence/remote-qualify-v1/q0-snn-layerwise-terminal.json': '2018f57b6ee2bf8970f1a93287cd4b979dfe82209211c070358e71c08fbf377c', 'evidence/remote-qualify-v1/q0-snn-layerwise/base_negative_count_input.json': '7ac36bc7a29765bbb9f13b54cc1ea2d973f4577ba5dd5e7c2168a11a84f767b0', 'evidence/remote-qualify-v1/q0-snn-layerwise/batch_duplicate.json': '43502501a7ce9480dbc78ba3d2bebc0e895ac694562b6420544d37cb73245db3', 'evidence/remote-qualify-v1/q0-snn-layerwise/initial_threshold_boundary.json': '5d78ae8846e1f7b62a5c2149f87f8b9a36446d07a6ae464cc0082d58089b69c1', 'evidence/remote-qualify-v1/q0-snn-layerwise/report.json': 'cb0d2b066452bc333ffb80389fdc82cfcbc2585202dad6377ab1a727927c6961', 'evidence/remote-qualify-v1/q0-snn-layerwise/single_sample_no_carry.json': '825008d3f7505a63ca0f8d8dd95ccf6d3106ecebe508f0c571f6b1661981f66d', 'evidence/remote-qualify-v1/q0-spyx-terminal.json': '433237435ad63e4932bed0062f87b71feee73a7b59881b4c3a92abc2b1d4ec5b', 'evidence/remote-qualify-v1/q0-spyx/base_negative_count_input.json': 'dd00b7f27bf3870fa1b24d9f0184b117925c5d2146c819f2ddd07804fb343f60', 'evidence/remote-qualify-v1/q0-spyx/batch_duplicate.json': '6972d9ff246f96798315a42a7fe620a19b6d97bf486104fcb17e7b2c0d14e2fe', 'evidence/remote-qualify-v1/q0-spyx/initial_threshold_boundary.json': 'f88bcc02a8bd929c6b63e2c3caf43cff2a9b8dbf2c09ea17cca53c90a02b3346', 'evidence/remote-qualify-v1/q0-spyx/report.json': '7e0ab5076656f8ab7be0d87970ba2a7a29cca275b674ebb60aa527495d535e39', 'evidence/remote-qualify-v1/q0-spyx/single_sample_no_carry.json': '6e57bf3442bc32de4bd65e1bf17f81b77084b16210f6c03f837276083b4ca896', 'evidence/runtime-arm64-preparation-r2/build-contract.json': 'b1c5e0d44c85a4fcfada1ac32589547dee3c3bce3e4ebbc786b44c4dc5e6ee1f', 'fixtures/e1-small/manifest.json': '146a4691acefeea9128eac1360a61c0e6d9c130c28e6a5588de4ce8acd25e3b3', 'fixtures/e1-small/seed-11.json': '10e39bbdc76f3291264520415cc1f70b39b8a177bd7f9d916c5e78af371348dc', 'fixtures/e1-small/seed-23.json': '6a8c237f745d442cda4f1e817f37f3be0efd02839cd09d1a8c3b8bf844cc5c3f', 'fixtures/e1-small/seed-37.json': '74217ec216579b15e55916237b6e606188c88ab87ffa424e78684a9893ef961f', 'fixtures/e1-small/seed-51.json': '3a9cd2e91fc85a105af40e3bf2e01ac3d3f984db399cdde84f4d0c26f1d81da6', 'fixtures/e1-small/seed-71.json': '7f8854a105610ea32087d7130384d51c697e642d06c88375ee1990178de1e695', 'fixtures/q0.json': '625df75c9161212a19f72067f89f28dc4a4e279c29eafdb1f829c2f96a96eee9', 'protocol/execution-plan-r2.json': 'cc3dc9f0219db5d1ab3fdb16760cecce7114903244e62294110aa7b64b16d118', 'sources/snapshot-manifest.json': '9362676928272655074e0c83138cbdbd47cea8d8fc740123ca11802a6dabf7e1', 'tools/arm64_artifact_gate_r1.py': '37589818b19c5d17e89bbe547148034b4b908c181b7328804676c3b2ffddf20e', 'tools/benchmark_competitor.py': '2321b1ef829c8a9fe4820d558ddc7acdf88cc5080911e7ac95a3f10909789505', 'tools/benchmark_e1.py': '4810063fdf19a706031e64d66a0a2a9d36b9904216ad2662509d271e69200a85', 'tools/build_arm64_runtime_r2.py': 'ffa9ce415b011b8ebcff11892881d14e0b3a4cbb600ca2d08b849cca6e09e948', 'tools/generate_dense_large.py': '910033835413339d5544d8341f6deeb5a96f571e19fb75bcd748596b2f7c316c', 'tools/generate_recurrent_cases.py': '5da896fa7bce9c4ad294535439ccdbc18281d1b61c30640c4859e7488e137b75', 'tools/qualify_dense.py': '35016f98c480de65d244cc7de81a2a21f7539a0d406a77fd62fcd9aa9e43af19', 'tools/qualify_dense_arm64_r2.py': 'c9c3bad6c590e9a3974a7c614e427dd177fe716f635fdb15e3b3444a48e2c572', 'tools/qualify_jax.py': '954a072a26d630f7c56b56527b76bf4751a68ede8a8323797070bbf27268194a', 'tools/run_recurrent_queue.py': '2f29be1bdc9548866587a46eb7255ddf597c29a04330d585ea89ae8409106f3f', 'tools/validate_e1_run.py': '5f2ef025910d2a90d47b68bc4f9a2b4cff2437ca34f94b4550451143d1a04b77'}
FIXTURE_RULE={'B': 16, 'T': 128, 'beta': 0.95, 'input_probabilities': [0.9, 0.095, 0.005], 'input_values': [0, 1, 2], 'labels': 'arange(B)%C', 'numpy_rng': 'default_rng PCG64; draw inputs then banks in network order', 'precision': 'arrays cast to IEEE f64 in compute; integer count JSON is lossless', 'seeds': [11, 23, 37, 51, 71], 'sizes': [128, 128, 10], 'theta': 1.0, 'weight_distribution': 'independent Normal(0,sqrt(2/fan_in)) per bank'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value,replace=False):
    path=Path(path)
    if replace:
        temporary=path.with_suffix(path.suffix+'.partial')
        with temporary.open('w') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        temporary.replace(path)
    else:
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def views(root=ROOT):
    return [[name,str(root/f'environment/{environment}/bin/python'),engine,list(flags)]
            for name,environment,engine,flags in VIEWS]


def order(root=ROOT):
    configured=views(root)
    return [[seed,view] for offset,seed in enumerate(SEEDS) for view in configured[offset:]+configured[:offset]]


def slots(root=ROOT):
    return [dict(name=view[0],seed=seed,slot_name=f'{view[0]}-seed-{seed}',
                 execution_status='not_launched_pending',performance_run=False) for seed,view in order(root)]


def remaining_cap(spent,prelaunch_s=0.):
    return max(0.,min(360.,1800.-spent)-prelaunch_s)


def strict_timeout(record,slot_elapsed,slot_cap):
    return (record.get('termination_reason')=='timeout' or
            record.get('elapsed_s',0)>record.get('timeout_s',360) or slot_elapsed>slot_cap)


def active_evaluation_processes():
    import psutil
    own=psutil.Process()
    excluded={own.pid,*[p.pid for p in own.parents()]}
    names={p.name for p in (ROOT/'tools').glob('*.py')
        if p.name.startswith(('run_','qualify_','benchmark_','generate_','measure_','retry_'))}
    names.update(['capability_migration.py','frontier_capability.py','seal_a1_v4_fixture_receipt.py'])
    found=[]
    for p in psutil.process_iter():
        if p.pid in excluded:continue
        try:
            ids=p.uids()
            if ids.real!=os.getuid() or ids.effective!=os.geteuid():continue
            argv=p.cmdline()
            if not argv:continue
            known=any(Path(a).name in names for a in argv)
            native=any(Path(a).name=='b2-train' for a in argv) and any(str(ROOT) in a for a in argv)
            in_private_tree=False
            if not known and not native:
                cwd=p.cwd()
                in_private_tree=cwd==str(ROOT) or cwd.startswith(str(ROOT)+os.sep)
            if known or native or in_private_tree:
                found.append(dict(pid=p.pid,create_time=p.create_time(),argv=argv))
        except (psutil.NoSuchProcess,psutil.ZombieProcess):continue
        except psutil.AccessDenied as error:
            raise RuntimeError('Process inspection denied; cannot prove serial execution: '+str(p.pid)) from error
    return found


def require_idle():
    conflicts=active_evaluation_processes()
    if conflicts:raise RuntimeError('Another evaluation process is active: '+json.dumps(conflicts))


def assert_identities(identities,root=ROOT):
    for relative,expected in identities.items():
        path=root/relative
        if not path.is_file() or sha(path)!=expected:
            raise RuntimeError('Frozen identity missing/changed: '+relative)


def passing_checks(value):
    return bool(value) and all(isinstance(v,dict) and v.get('passed') is True for v in value.values())


def producer_gate(root=ROOT):
    path=root/PRODUCER
    report=read(path)
    if (report.get('schema')!='runtime-arm64-build-r2' or report.get('status')!='built_unqualified' or
        report.get('source_integrity_after_exit') is not True or report.get('runtime_path')!=str(RUNTIME) or
        report.get('source')!=str(BUILD_CONTRACT) or report.get('runtime_architecture')!='arm64-only'):
        raise RuntimeError('ARM64 producer must attest built_unqualified, source integrity and the separate runner path')
    contract=read(root/BUILD_CONTRACT)
    assert_identities(contract['source_identities'],root)
    if any(report.get(key)!=contract['source_identities'] for key in ('sources_before','sources_after','sources_final')):
        raise RuntimeError('Producer source receipts do not match the frozen 32-source contract')
    if (sha(root/'runtime/b2-train')!=contract['original_runtime_sha256'] or
        report.get('original_runtime_sha256')!=contract['original_runtime_sha256']):
        raise RuntimeError('Original runtime was changed during ARM64 publication')
    if contract['driver_sha256']!=sha(root/'tools/build_arm64_runtime_r2.py'):
        raise RuntimeError('ARM64 producer source differs from its frozen build contract')
    runtime=gate_artifact(root/RUNTIME,'runner',report.get('runtime_sha256'))
    if not report.get('runtime_sha256') or runtime['sha256']!=report['runtime_sha256']:
        raise RuntimeError('ARM64 producer/runtime SHA mismatch')
    return dict(path=str(PRODUCER),sha256=sha(path),status=report['status'],
                source_integrity_after_exit=True,build_contract_path=str(BUILD_CONTRACT),
                build_contract_sha256=sha(root/BUILD_CONTRACT),runner=runtime)


def q0_gate(name,engine,flags,runtime,root=ROOT):
    path=root/(ATLAS_Q0 if name=='atlas' else Path(f'evidence/remote-qualify-v1/q0-{name}/report.json'))
    report=read(path)
    if report.get('engine')!=engine or report.get('dense_qualification_status')!='passed' or set(report.get('cases',{}))!=set(Q0_CASES):
        raise RuntimeError('Matching complete dense Q0 did not pass: '+name)
    if any(v.get('passed') is not True or not passing_checks(v.get('checks')) for v in report['cases'].values()):
        raise RuntimeError('Dense Q0 does not contain four passing check banks: '+name)
    assert_identities(report.get('identities',{}),root)
    identities={str(path.relative_to(root)):sha(path)}
    api_sha=sha(root/'snapshot/brian2-rust/python/brian2_rust/training.py')
    for case in Q0_CASES:
        raw_path=path.parent/(case+'.json');raw_report=read(raw_path)
        if raw_report.get('checks')!=report['cases'][case]['checks']:
            raise RuntimeError('Q0 raw/report check bank differs: '+name+'/'+case)
        identities[str(raw_path.relative_to(root))]=sha(raw_path)
        if name=='atlas':
            raw=raw_report.get('raw',{})
            for entry in (raw,raw.get('sgd',{})):
                if entry.get('runner_sha256')!=runtime['sha256'] or entry.get('training_api_sha256')!=api_sha:
                    raise RuntimeError('New Atlas Q0 raw runtime/API mismatch: '+case)
    if name=='atlas':
        if (report.get('schema')!='dense-Q0-arm64-r2' or report.get('native_profile')!='arm64-r2' or
            report.get('execution_status')!='completed' or report.get('runner')!=runtime):
            raise RuntimeError('Atlas Q0 is not this completed ARM64 profile')
        required=['tools/qualify_dense_arm64_r2.py','tools/arm64_artifact_gate_r1.py','adapters/atlas_adapter.py','adapters/oracle.py','fixtures/q0.json']
        if any(report['identities'].get(n)!=sha(root/n) for n in required):
            raise RuntimeError('Atlas Q0 has incomplete frozen source identities')
    else:
        receipt_path=root/f'evidence/remote-qualify-v1/q0-{name}-terminal.json';receipt=read(receipt_path)
        if receipt.get('exit_code')!=0 or receipt.get('termination_reason')!='exited':
            raise RuntimeError('Inherited competitor Q0 supervisor did not close successfully: '+name)
        command=receipt.get('command',[])
        if any((flag in command)!=(flag in flags) for flag in ('--layerwise','--compile')):
            raise RuntimeError('Inherited competitor Q0 flags do not match view: '+name)
        if '--engine' not in command or command[command.index('--engine')+1]!=engine:
            raise RuntimeError('Inherited competitor Q0 engine differs: '+name)
        identities[str(receipt_path.relative_to(root))]=sha(receipt_path)
    return dict(path=str(path.relative_to(root)),sha256=sha(path),status='passed',engine=engine,
                flags=list(flags),inherited=name!='atlas',identities=identities)


def preflight(root=ROOT):
    assert_identities(FIXED,root)
    producer=producer_gate(root)
    source_manifest=read(root/'sources/snapshot-manifest.json')
    snapshot={str(Path('snapshot')/p):h for p,h in source_manifest['source_hashes'].items()}
    assert_identities(snapshot,root)
    fixture=read(root/'fixtures/e1-small/manifest.json')
    if fixture.get('rule')!=FIXTURE_RULE or set(fixture['files'])!={f'seed-{s}.json' for s in SEEDS}:
        raise RuntimeError('Original E1 common array rule or full seed denominator changed')
    arrays={f'fixtures/e1-small/{name}':entry['sha256'] for name,entry in fixture['files'].items()}
    assert_identities(arrays,root)
    for name,entry in fixture['files'].items():
        if (root/'fixtures/e1-small'/name).stat().st_size!=entry['bytes']:raise RuntimeError('Common array byte count differs')
    q0={name:q0_gate(name,engine,flags,producer['runner'],root) for name,_,engine,flags in VIEWS}
    interpreters={env:gate_artifact(root/f'environment/{env}/bin/python','runner') for env in ('cpu','jax')}
    identities={**FIXED,**snapshot,**arrays,str(PRODUCER):producer['sha256'],str(RUNTIME):producer['runner']['sha256']}
    for gate in q0.values():identities.update(gate['identities'])
    for env,info in interpreters.items():identities[f'environment/{env}/bin/python']=info['sha256']
    for script in ['tools/run_e1_small_arm64_r2.py','tools/validate_e1_small_arm64_r2.py']:
        identities[script]=sha(root/script)
    return dict(schema='e1-small-arm64-freeze-r1',native_profile='arm64-r2',
        host=HOST,phase='benchmark',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy=POLICY,views=views(root),runtime_path=str(RUNTIME),runtime_sha256=producer['runner']['sha256'],
        runtime=producer['runner'],producer=producer,atlas_q0_path=str(ATLAS_Q0),q0=q0,
        python_interpreters=interpreters,source_manifest=sha(root/'sources/snapshot-manifest.json'),
        scripts={n:h for n,h in identities.items() if n.endswith('.py') and n.startswith(('tools/','adapters/'))},
        identities=identities,locks={p.name:sha(p) for p in (root/'environment').glob('*lock.txt')},
        hardware=sha(root/'environment/hardware.json'),
        fixtures=dict(manifest_path='fixtures/e1-small/manifest.json',manifest_sha256=sha(root/'fixtures/e1-small/manifest.json'),
                      seed_files=[dict(seed=s,path=f'fixtures/e1-small/seed-{s}.json',**fixture['files'][f'seed-{s}.json']) for s in SEEDS]),
        actual_numerical_protocol=POLICY['numerical_scope'],jax_resource_qualified=False,
        prior_timing_used=False)


def command_for(seed,view,folder,root=ROOT):
    name,python,engine,flags=view
    output=folder/f'{name}-seed-{seed}'
    if name=='atlas':
        return [python,str(root/'tools/benchmark_e1.py'),'--worker','--base',str(root),
                '--source-root',str(root/'snapshot/brian2-rust'),'--runner',str(root/RUNTIME),
                '--output',str(output),'--seed',str(seed)]
    return [python,str(root/'tools/benchmark_competitor.py'),'--engine',engine,'--seed',str(seed),'--output',str(output),*flags]


def blocked_records(records,reason,status='not_launched_prerequisite'):
    for row in records:
        if row.get('execution_status')=='not_launched_pending':
            row.update(execution_status=status,reason=reason,performance_run=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-run',action='store_true')
    args=parser.parse_args()
    if not args.allow_run or socket.gethostname()!=HOST:
        parser.error('Prepared only; explicit --allow-run on the authorized remote host is required')
    folder=ROOT/RUN;folder.mkdir(parents=True,exist_ok=False)
    records=slots();plan=order();spent={v[0]:0. for v in VIEWS};freeze=None;active=None
    started=time.monotonic();lock=None
    terminal=dict(schema='e1-small-arm64-terminal-r1',status='preflight_rejected',supervisor_completed=False,
                  performance_run=False,native_profile='arm64-r2',policy=POLICY)
    save(folder/'order.json',plan);save(folder/'progress.json',records)
    pre=dict(schema='e1-small-arm64-preflight-r1',status='pending',stage='exclusive_idle_gate',
        dispatcher_sha256=sha(__file__),producer_path=str(PRODUCER),atlas_q0_path=str(ATLAS_Q0),
        expected_fixed_identities=FIXED,observed_prerequisites={})
    save(folder/'preflight.json',pre)
    try:
        lock=(ROOT/'evidence/arm64-e1-small-r1.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_idle()
        pre['stage']='producer_source_runtime_q0_fixtures'
        for rel in (PRODUCER,ATLAS_Q0):
            path=ROOT/rel
            pre['observed_prerequisites'][str(rel)]=dict(exists=path.is_file(),sha256=sha(path) if path.is_file() else None)
        save(folder/'preflight.json',pre,replace=True)
        freeze=preflight()
        if shutil.disk_usage(ROOT).free<POLICY['free_disk_floor_bytes']:raise RuntimeError('50 GiB free disk floor reached')
        require_idle()
        save(folder/'freeze.json',freeze)
        pre.update(status='passed',stage='completed',elapsed_s=time.monotonic()-started,freeze_sha256=sha(folder/'freeze.json'))
        save(folder/'preflight.json',pre,replace=True)
        from run_recurrent_queue import launch
        terminal.update(status='queue_running',performance_run=True,freeze_sha256=sha(folder/'freeze.json'))
        for index,(seed,view) in enumerate(plan):
            name=view[0];row=records[index];active=index;slot_started=time.monotonic()
            slot_cap=min(360.,1800.-spent[name]);row.update(slot_cap_s=max(0.,slot_cap),view_spent_before_s=spent[name])
            if slot_cap<=0:
                row.update(execution_status='timeout',reason='Per-view 1800 s aggregate wall budget exhausted before launch',phase='prelaunch_budget',performance_run=False)
                active=None;save(folder/'progress.json',records,replace=True);continue
            assert_identities(freeze['identities']);require_idle()
            if shutil.disk_usage(ROOT).free<POLICY['free_disk_floor_bytes']:raise RuntimeError('50 GiB free disk floor reached')
            cap=remaining_cap(spent[name],time.monotonic()-slot_started)
            if cap<=0:
                row.update(execution_status='timeout',reason='Per-slot budget exhausted during identity/idle gate',phase='prelaunch_gate',slot_elapsed_s=time.monotonic()-slot_started,performance_run=False)
                spent[name]+=row['slot_elapsed_s'];active=None;save(folder/'progress.json',records,replace=True);continue
            stem=row['slot_name'];job=folder/'control'/stem;job.mkdir(parents=True,exist_ok=False)
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg','MPLCONFIGDIR':str(job/'mpl'),
                'TMPDIR':str(job/'tmp'),'TORCHINDUCTOR_CACHE_DIR':str(job/'torchinductor'),
                'TRITON_CACHE_DIR':str(job/'triton'),'JAX_COMPILATION_CACHE_DIR':str(job/'jax'),
                'JAX_PLATFORM_NAME':'cpu','JAX_ENABLE_X64':'true','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                'MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
            Path(env['TMPDIR']).mkdir()
            command=command_for(seed,view,folder)
            row.update(execution_status='launching',performance_run=True,command=command,
                resource_environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','JAX_PLATFORM_NAME','JAX_ENABLE_X64')},
                cache_directories={k:env[k] for k in ('MPLCONFIGDIR','TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','JAX_COMPILATION_CACHE_DIR')})
            save(folder/'progress.json',records,replace=True)
            print('START',stem,flush=True)
            launched=launch(stem,command,cap,POLICY['rss_guard_bytes'],folder,folder/stem,env,'E1_import_oracle_qualification_training_or_IO_no_finer_worker_phase')
            elapsed=time.monotonic()-slot_started
            row.update(launched,name=name,seed=seed,slot_name=stem,slot_elapsed_s=elapsed,
                       supervisor_receipt_sha256=sha(folder/f'{stem}-terminal.json'),
                       resource_samples_sha256=sha(folder/f'{stem}-resources.jsonl'))
            row.pop('execution_status',None)
            if strict_timeout(launched,elapsed,slot_cap):
                row.update(execution_status='timeout',reason='Strict seed/remaining-view wall exceeded; polling/cleanup overrun never permits a timing ratio')
            elif launched['termination_reason']=='resource_limit':
                row.update(execution_status='resource_limit',reason='64 GiB process-tree guard; not physical OOM or engine capacity evidence')
            spent[name]+=elapsed;row['view_spent_after_s']=spent[name]
            active=None;save(folder/'progress.json',records,replace=True)
            print('END',stem,launched['exit_code'],row.get('execution_status',launched['termination_reason']),flush=True)
            require_idle()
        terminal.update(status='queue_closed',supervisor_completed=True)
        return 0
    except BaseException as error:
        pre.update(status='rejected' if freeze is None or not (folder/'freeze.json').exists() else pre['status'],
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started)
        save(folder/'preflight.json',pre,replace=True)
        if active is not None:
            row=records[active]
            receipt=folder/(row['slot_name']+'-terminal.json')
            if receipt.is_file():
                observed=read(receipt);row.update(observed,name=row['name'],seed=row['seed'],slot_name=row['slot_name'],
                    supervisor_receipt_sha256=sha(receipt),execution_status='supervision_error')
            else:row.update(execution_status='not_launched_gate_error' if row.get('execution_status')=='not_launched_pending' else 'launch_or_supervision_error')
            row.update(error_type=type(error).__name__,error=str(error))
        terminal.update(status='preflight_rejected' if not (folder/'freeze.json').exists() else 'queue_stopped',
                        error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        blocked_records(records,str(error))
        return 2
    finally:
        blocked_records(records,terminal['status'])
        terminal.update(records=records,closed_slots=len(records),view_spent_s=spent,
            elapsed_s=time.monotonic()-started,preflight_sha256=sha(folder/'preflight.json'),
            order_sha256=sha(folder/'order.json'),dispatcher_sha256=sha(__file__))
        save(folder/'progress.json',records,replace=True)
        save(folder/'terminal.json',terminal)
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':raise SystemExit(main())
