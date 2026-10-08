"""Shared CPU learning law and conductance-LIF equations.

Eligibility is reset by a delivered KC spike and decays between spikes.
An external compartment teaching signal causes depression; activity without
teaching restores gain towards one. Recovery is an explicit modelling choice.
"""
from pathlib import Path
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT.parents[0]))

NEURON_EQUATIONS = """
dv/dt=(-(v+52*mV)-ge*v-gi*(v+70*mV))/(20*ms) : volt (unless refractory)
dge/dt=-ge/(5*ms) : 1
dgi/dt=-gi/(5*ms) : 1
"""
PLASTIC_EQUATIONS = """
deligibility/dt=-eligibility/tau_eligibility : 1 (clock-driven)
dgain/dt=learning(t)*eligibility*(recovery*(1-teaching(t))*(1-gain)-depression*teaching(t)*gain) : 1 (clock-driven)
multiplicity : 1 (constant)
"""
PLASTIC_PRE = "ge_post += contact_weight*multiplicity*gain; eligibility = 1"


def setup(backend, directory, threads=1):
    import brian2 as b
    import brian2_rust  # noqa: F401
    compiler = Path('/atlas-home/0004/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin')
    if compiler.exists():
        os.environ['PATH'] = str(compiler) + os.pathsep + os.environ['PATH']
    b.device.reinit()
    b.start_scope()
    if backend in ('aot', 'reference'):
        b.set_device('rust_standalone', engine=backend, threads=threads,
                     runner=ROOT/'target/release/b2-runner', directory=Path(directory))
    elif backend == 'numpy':
        b.set_device('runtime')
        b.prefs.codegen.target = 'numpy'
    elif backend == 'cpp':
        b.set_device('cpp_standalone', build_on_run=False)
        b.prefs.codegen.cpp.extra_compile_args = ['-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off']
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ['-j2']
    else:
        raise ValueError(backend)


def plastic(source, target, pre, post, contacts, teaching, learning,
            delay_ms=1.8, tau_ms=100., depression_hz=4., recovery_hz=4.):
    import brian2 as b
    # Euler is a convex update for gain and eligibility under these bounds.
    dt = float(source.clock.dt / b.second)
    if not (0 < dt <= tau_ms / 1000 and
            0 <= dt * max(depression_hz, recovery_hz) <= 1):
        raise ValueError('unstable plasticity timestep/rates')
    s = b.Synapses(source, target, PLASTIC_EQUATIONS, on_pre=PLASTIC_PRE,
                   clock=source.clock, method='euler', delay=delay_ms*b.ms,
                   namespace={'teaching': teaching, 'learning': learning,
                              'tau_eligibility': tau_ms*b.ms,
                              'recovery': recovery_hz*b.Hz, 'depression': depression_hz*b.Hz,
                              'contact_weight': .275/52}, name='local_plastic')
    s.connect(i=np.asarray(pre,dtype=np.int32), j=np.asarray(post,dtype=np.int32))
    s.multiplicity = contacts
    s.gain = 1
    s.eligibility = 0
    return s
