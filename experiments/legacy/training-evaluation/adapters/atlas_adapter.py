"""Atlas public-API adapter for the frozen synchronous LIF evaluation.

Run from the isolated evaluation interpreter. This module never compiles Atlas,
changes its sources or performs a Python forward/backward/optimizer calculation.
Every timed operation wraps the full coordinator public API call. Prefix replay
is only a qualification diagnostic because the API exposes final membrane, not
a membrane tape. Returned data are execution evidence, not a qualification claim.

case: sizes; inputs [B,T,I]; labels [B]; source-major flattened weight banks;
optional initial [B,sum(sizes[1:])], projections, masks, trainable, beta, threshold.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

class DependencyInitializationError(RuntimeError):
    pass


INPUT_LIMIT = 64 * 1024**2
TAPE_LIMIT = 1024**3


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _canonical_identity(value):
    # Same bytes as protocol.canonical_bytes for this adapter's plain objects;
    # count/hash incrementally so rejected inputs do not need a giant byte copy.
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False)
    count = 0
    digest = hashlib.sha256()
    for text in encoder.iterencode(value):
        chunk = text.encode('utf-8')
        count += len(chunk)
        digest.update(chunk)
    return {'bytes': count, 'sha256': digest.hexdigest()}


def admission(plan, state, inputs, labels, *, operation='train', initial=None):
    """Exact native v1/v2 LIF guards, not a host/device peak-memory estimate.

    All JSON bytes (including state and masks) are counted each call. Dynamic,
    scalar equation and multi-state plans deliberately require a separate audit.
    """
    if plan['schema'] not in ('b2-lif-training-plan-v1', 'b2-lif-training-plan-v2'):
        raise NotImplementedError('exact admission implemented only for v1/v2 LIF')
    request = dict(plan=plan, state=state, operation=operation,
                   inputs=_plain(inputs), labels=_plain(labels), initial=_plain(initial))
    wire = _canonical_identity(request)
    sizes = plan['sizes']
    batch, ticks = len(request['inputs']), len(request['inputs'][0])
    neurons, parameters = sum(sizes[1:]), sum(map(len, plan['masks']))
    projections = plan.get('projections')
    topology = sum(len(p['sources']) * 24 + 128 for p in projections) if projections is not None else 0
    ranks = plan.get('mpi_ranks')
    mpi = max(neurons, max(map(len, plan['masks']), default=0)) * 8 * (ranks + 1) if ranks else 0
    parts = dict(tapes_and_returned_spikes=batch*ticks*neurons*(32 if projections is not None else 24),
                 live_state=batch*neurons*32, logits=batch*sizes[-1]*16,
                 parameter_gradient_optimizer=parameters*48, topology=topology,
                 mpi_workspace=mpi, inputs=batch*ticks*(sizes[0]*8+48)+batch*8)
    native_bytes = sum(parts.values())
    initial_bytes = batch*neurons*64 + parameters*48
    budget = plan['max_tape_bytes']
    reasons = []
    if wire['bytes'] > INPUT_LIMIT:
        reasons.append('request_json_exceeds_64_mib')
    if not 0 < budget <= TAPE_LIMIT:
        reasons.append('plan_tape_budget_outside_1_gib_limit')
    if parameters*48 + topology > budget:
        reasons.append('parameter_budget_exceeded')
    if initial_bytes > budget:
        reasons.append('initial_state_budget_exceeded')
    if native_bytes > budget:
        reasons.append('native_tape_budget_exceeded')
    return dict(status='budget_rejected' if reasons else 'admitted', reasons=reasons,
                request=wire, input_limit_bytes=INPUT_LIMIT, plan_max_tape_bytes=budget,
                hard_max_tape_bytes=TAPE_LIMIT, exact_native_tape_bytes=native_bytes,
                exact_initial_budget_bytes=initial_bytes, components=parts,
                scope='exact software admission; excludes JSON parser/output/allocator peak RSS')


def load_api(source_root):
    root = Path(source_root).resolve()
    python_root = root / 'python' if (root / 'python').is_dir() else root
    package_root = python_root / 'brian2_rust'
    if not (package_root / 'training.py').is_file():
        raise FileNotFoundError(f'Atlas training.py absent in {package_root}')
    # Normal package import exercises the real public API and dependencies.
    if (root.parent / 'brian2').is_dir():
        sys.path.insert(0, str(root.parent))
    sys.path.insert(0, str(python_root))
    try:
        module = importlib.import_module('brian2_rust.training')
    except Exception as error:
        raise DependencyInitializationError(f'{type(error).__name__}: {error}') from error
    if Path(module.__file__).resolve() != package_root / 'training.py':
        raise RuntimeError('another Atlas source tree is already imported in this process')
    return module


def failure_status(error):
    message = str(error).lower()
    if isinstance(error, (ImportError, ModuleNotFoundError, DependencyInitializationError)):
        return 'dependency_failed'
    if isinstance(error, subprocess.TimeoutExpired):
        return 'timeout'
    if isinstance(error, MemoryError):
        return 'oom'
    if 'nonfinite' in message:
        return 'numerical_divergence'
    if 'budget exceeded' in message or '64 mib input limit' in message:
        return 'budget_rejected'
    if isinstance(error, NotImplementedError):
        return 'unsupported'
    return 'software_rejected'


class AtlasAdapter:
    def __init__(self, case, source_root, runner, *, optimizer='adam',
                 backend='cpu', max_tape_bytes=TAPE_LIMIT, mpi_ranks=None):
        self.case = _plain(case)
        self.api = load_api(source_root)
        self.runner = Path(runner).resolve()
        self.source_root = str(Path(source_root).resolve())
        self.backend = backend
        options = dict(backend=backend, beta=self.case.get('beta', .95),
                       threshold=self.case.get('threshold', self.case.get('theta', 1.)), reset='subtract',
                       detach_reset=True, surrogate_slope=5., surrogate_scale=1.,
                       optimizer=optimizer, learning_rate=.001, seed=1,
                       logit_scale=5., max_tape_bytes=max_tape_bytes)
        for name in ('projections', 'masks', 'trainable'):
            if name in self.case:
                options[name] = self.case[name]
        if mpi_ranks is not None:
            options['mpi_ranks'] = mpi_ranks
        self.plan = self.api.lif_training_plan(self.case['sizes'], **options)
        self.trainer = self.api.NativeLIFTrainer(self.plan, runner=self.runner,
                                               weights=self.case['weights'])
        self.records = []

    def call(self, operation='train', *, inputs=None, initial=None):
        inputs = self.case['inputs'] if inputs is None else _plain(inputs)
        # Every call starts from the supplied initial state; no implicit carry.
        initial = self.case.get('initial') if initial is None else _plain(initial)
        labels = self.case['labels']
        check = admission(self.trainer.plan, self.trainer.state, inputs, labels,
                          operation=operation, initial=initial)
        record = dict(operation=operation, admission=check, status=check['status'])
        self.records.append(record)
        if check['status'] != 'admitted':
            return record
        start = time.perf_counter_ns()
        try:
            result = self.trainer.execute(inputs, labels, operation=operation, initial=initial)
            stop = time.perf_counter_ns()
            record.update(status='executed', public_api_ns=stop-start, result=result)
            record['tape_count_matches_runtime'] = result['tape_bytes'] == check['exact_native_tape_bytes']
        except Exception as error:
            stop = time.perf_counter_ns()
            record.update(status=failure_status(error), public_api_ns=stop-start,
                          error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
        return record

    def state_trace(self):
        """Replay prefixes with fixed parameter/optimizer state, outside step timing."""
        live = copy.deepcopy(self.trainer.state)
        traces = []
        diagnostics = []
        for stop in range(1, len(self.case['inputs'][0]) + 1):
            self.trainer.state = copy.deepcopy(live)
            record = self.call('evaluate', inputs=[row[:stop] for row in self.case['inputs']])
            diagnostics.append({k: v for k, v in record.items() if k != 'result'})
            if record['status'] != 'executed':
                self.trainer.state = live
                return dict(status=record['status'], complete_ticks=len(traces), diagnostics=diagnostics)
            traces.append(record['result']['final_membrane'])
        self.trainer.state = live
        # API prefix result is [B,N]; public trace contract is [B,T,N].
        return dict(status='executed', states=[[[v for v in traces[t][b]]
                         for t in range(len(traces))] for b in range(len(self.case['inputs']))],
                    method='public evaluate on every prefix with fixed weights; qualification only',
                    diagnostics=diagnostics)


def run_case(case, source_root, runner, *, optimizer='adam', steps=3,
             collect_states=True, backend='cpu', max_tape_bytes=TAPE_LIMIT,
             mpi_ranks=None):
    report = dict(schema='atlas-evaluation-adapter-v1', case_id=case.get('id'),
                  engine='Atlas', status='unqualified', source_root=str(Path(source_root).resolve()),
                  runner=str(Path(runner).resolve()), adapter_sha256=sha256_file(__file__),
                  timing_scope='coordinator full public execute call: JSON, tempfile, subprocess, MPI launch/communication, parse; no kernel-only substitution',
                  optimizer=optimizer, qualification='independent caller must compare arrays; executed is not qualified')
    try:
        start = time.perf_counter_ns()
        adapter = AtlasAdapter(case, source_root, runner, optimizer=optimizer,
                               backend=backend, max_tape_bytes=max_tape_bytes, mpi_ranks=mpi_ranks)
        report['constructor_ns'] = time.perf_counter_ns() - start
        report['runner_sha256'] = sha256_file(runner)
        report['training_api_sha256'] = sha256_file(adapter.api.__file__)
        report['plan'] = adapter.plan
        report['numeric_profile'] = dict(state='f64 on CPU; f32 accelerator profile must be separately qualified',
            parameter_storage='f64', optimizer='native host f64', batch_reduction='batch-mean cross entropy',
            beta=adapter.plan['beta'], threshold={'comparison':'strict >','values':adapter.plan['threshold']}, surrogate='1/(1+5*abs(u-theta))**2',
            reset='detached subtract; native subtract before source-major scalar edge addition',
            optimizer_detail='mhat/(sqrt(vhat)+1e-8); betas .9/.999; step increments once per train call')
        report['gradients'] = adapter.call('gradients')
        if report['gradients']['status'] != 'executed':
            report['status'] = report['gradients']['status']
            return report
        if collect_states:
            report['state_trace'] = adapter.state_trace()
            if report['state_trace']['status'] != 'executed':
                report['status'] = report['state_trace']['status']
                return report
        report['updates'] = []
        for _ in range(steps):
            record = adapter.call('train')
            report['updates'].append(record)
            if record['status'] != 'executed':
                report['status'] = record['status']
                return report
        report['status'] = 'executed'
        report['unsupported_observability'] = ['arbitrary external-cotangent spike VJP API; CE initial_gradients and parameter gradients are available']
        return report
    except Exception as error:
        report.update(status=failure_status(error), error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--runner', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--backend', choices=['cpu', 'metal', 'cuda'], default='cpu')
    parser.add_argument('--no-state-trace', action='store_true')
    args = parser.parse_args()
    case = json.loads(Path(args.case).read_text())
    report = run_case(case, args.source_root, args.runner, optimizer=args.optimizer,
                      steps=args.steps, collect_states=not args.no_state_trace, backend=args.backend)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('preserve prior run evidence: choose a new output path')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(output))))
    return 0 if report['status'] == 'executed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
