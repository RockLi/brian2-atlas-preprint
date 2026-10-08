"""Extend only full-observation duration fields; preserve frozen model inputs."""
import copy
import struct


def bits(value):
    return struct.pack('>d', value).hex()


def number(value):
    return struct.unpack('>d', bytes.fromhex(value))[0]


def _target_steps(seconds):
    targets = {10.5: 105000, 50.5: 505000, 100.5: 1005000}
    if type(seconds) not in (int, float) or seconds not in targets:
        raise ValueError('target must be a declared 10.5, 50.5 or 100.5 second observation')
    return targets[seconds]


def audit_duration_delta(old, new, seconds=10.5):
    target_steps = _target_steps(seconds)
    if number(old['run']['duration']) != 2.5:
        raise ValueError('requires the frozen 2.5 second baseline')
    if old['schema'] != new['schema'] or old['instance'] != new['instance']:
        raise ValueError('schema or frozen instance changed')
    if len(old['definition']['clocks']) != 1 or len(old['run']['clocks']) != 1:
        raise ValueError('requires the single-clock full MAM condition')
    dt = number(old['definition']['clocks'][0]['dt'])
    if dt != .0001 or old['run']['clocks'][0]['start_tick'] != 0:
        raise ValueError('requires the frozen 0.1 ms grid and zero start')
    if old['run']['clocks'][0]['steps'] != 25000:
        raise ValueError('baseline clock does not cover the complete 2.5 seconds')
    run = copy.deepcopy(new['run'])
    if run['duration'] != bits(seconds) or run['clocks'][0]['steps'] != target_steps:
        raise ValueError('incorrect target duration')
    run['duration'] = old['run']['duration']
    run['clocks'][0]['steps'] = old['run']['clocks'][0]['steps']
    if run != old['run']:
        raise ValueError('undeclared run field change')
    definition = copy.deepcopy(new['definition'])
    for population, prior in zip(definition['populations'], old['definition']['populations'], strict=True):
        if (prior['steps'] != 25000 or prior['monitor']['window_steps'] != 25000
                or population['steps'] != target_steps or population['monitor']['window_steps'] != target_steps):
            raise ValueError('full monitor prefix must be retained')
        population['steps'] = prior['steps']
        population['monitor']['window_steps'] = prior['monitor']['window_steps']
    if definition != old['definition']:
        raise ValueError('undeclared definition field change')
    return dict(instance_exact=True, only_duration_and_full_monitor_lengths_changed=True,
                original_steps=25000, target_steps=target_steps, physical_prefix_ms=2500,
                prefix_execution_verified=False)


def extend(old, seconds=10.5):
    target_steps = _target_steps(seconds)
    new = dict(old, definition=copy.deepcopy(old['definition']), run=copy.deepcopy(old['run']))
    new['run']['duration'] = bits(seconds)
    new['run']['clocks'][0]['steps'] = target_steps
    for population in new['definition']['populations']:
        population['steps'] = population['monitor']['window_steps'] = target_steps
    audit_duration_delta(old, new, seconds)
    # The caller must rebuild and verify the derived protocol before writing.
    return new


def output_bytes(model, spikes, last_spikes, *, compact_spikes=False):
    """Exact full-MAM byte formula; v3 by default, opt-in v4 spike pairs.

    Reject monitors/topologies outside this frozen full-MAM dump contract.
    Future spike counts are workload scenarios, not a guarantee or admission.
    """
    if type(spikes) is not int or type(last_spikes) is not int or min(spikes, last_spikes) < 0:
        raise ValueError('nonnegative integer counts required')
    if type(compact_spikes) is not bool:
        raise TypeError('compact_spikes must be a boolean')
    if compact_spikes:
        for clock in model['run']['clocks']:
            if not 0 <= clock['start_tick'] < 2**32 or clock['start_tick'] + clock['steps'] > 2**32:
                raise ValueError('compact output tick exceeds u32 domain')
    sizes = {'bool': 1, 'f32': 4, 'f64': 8, 'i32': 4, 'i64': 8, 'u32': 4, 'u64': 8}
    record_bytes = 8 if compact_spikes else 16
    result = 40 + 8 + 16 + record_bytes * spikes + 8 * last_spikes
    event = 32 + record_bytes * spikes
    for p, instance in zip(model['definition']['populations'], model['instance']['populations'], strict=True):
        if p.get('events') != ['spike'] or p.get('event_monitors'):
            raise ValueError('budget formula requires spike-only streams and no custom monitors')
        if compact_spikes and not 0 <= p['count'] <= 2**32:
            raise ValueError('compact output neuron exceeds u32 domain')
        symbols = {v['name']: v for v in p['states'] + p['parameters'] + p.get('linked_variables', [])}
        monitor = p['monitor']
        trace = monitor['window_steps'] * len(monitor['record']) * sum(sizes[symbols[name]['dtype']] for name in monitor['variables'])
        result += 64 + trace + 8 * p['count'] + p['count'] * sum(sizes[s['dtype']] for s in p['states'])
        if instance.get('refractory') is not None:result += 9 * p['count']
        event += 8
    for s in model['definition']['synapses']:
        if s['states']:
            raise ValueError('budget formula requires stateless full-MAM synapses')
        result += 24
    return dict(results_bytes=result, events_bytes=event, combined_bytes=result + event)
