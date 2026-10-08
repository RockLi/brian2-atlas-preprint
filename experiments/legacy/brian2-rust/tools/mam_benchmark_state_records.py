"""Bounded native state conversion; no sampling, inference or simulator changes.

The caller must supply genuine recordable values and an audited sampling plan.
These helpers are not evidence that a producer has recorded all required data.
"""
import numpy as np

MAX_ROWS = 262144
VOLTAGE_DTYPE = np.dtype([('tick', '<u4'), ('cell', '<u4'), ('voltage_V', '<f8')])
FINAL_DTYPE = np.dtype([('cell', '<u4'), ('voltage_V', '<f8'), ('current_A', '<f8')])


def _float64(values, name):
    # Refuse implicit float32 promotion: lost precision cannot be reconstructed.
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype != np.dtype('float64'):
        raise ValueError(name+' must be a one-dimensional native float64 array')
    if len(values) > MAX_ROWS or not np.isfinite(values).all():
        raise ValueError(name+' exceeds the block bound or contains nonfinite values')
    return values


def _cells(senders, cell_count):
    if type(cell_count) is not int or not 0 < cell_count <= 2**32:
        raise ValueError('Invalid global cell domain')
    if (not isinstance(senders, np.ndarray) or senders.ndim != 1
            or senders.dtype.kind not in 'iu' or len(senders) > MAX_ROWS):
        raise ValueError('Senders must be a bounded integer array')
    if not np.all((senders >= 1) & (senders <= cell_count)):
        raise ValueError('Sender outside one-based global cell domain')
    return (senders-1).astype('<u4')


def voltage_records(times_ms, senders, voltage_mv, *, cell_count, last_tick):
    """Retain physical sample times; never shift state timestamps like spike labels.

    Includes an endpoint at last_tick when supplied. The caller must retain and
    separately classify an endpoint; this helper never crops to a trace window.
    """
    if type(last_tick) is not int or not 0 <= last_tick < 2**32:
        raise ValueError('Invalid physical time domain')
    times = _float64(times_ms, 'times_ms')
    voltage = _float64(voltage_mv, 'voltage_mv')
    cells = _cells(senders, cell_count)
    if not len(times) == len(voltage) == len(cells):
        raise ValueError('Voltage record lengths differ')
    ticks = np.rint(times/.1)
    if not np.all((ticks >= 0) & (ticks <= last_tick)
                  & np.isclose(times/.1, ticks, rtol=0, atol=1e-9)):
        raise ValueError('State sample is outside the exact 0.1 ms grid/domain')
    result = np.empty(len(times), dtype=VOLTAGE_DTYPE)
    result['tick'], result['cell'], result['voltage_V'] = ticks.astype('<u4'), cells, voltage*1e-3
    # Duplicate (cell,time) is an invalid capture, not something to average.
    order = np.lexsort((result['cell'], result['tick']))
    ordered = result[order]
    if len(result)>1 and np.any((ordered['tick'][1:] == ordered['tick'][:-1])
                               & (ordered['cell'][1:] == ordered['cell'][:-1])):
        raise ValueError('Duplicate voltage sample')
    return result  # preserve supplied order, including empty blocks


def final_state_records(senders, voltage_mv, current_ex_pa, current_in_pa, *,
                        cell_count, tau_syn_ex_ms, tau_syn_in_ms):
    """Combine equal-time-constant PSCs in float64; no guessed hidden state.

    The caller, not this conversion, must establish that every value is at the
    terminal physical time. Final refractory/counter fields are separate work.
    """
    if not (np.isfinite(tau_syn_ex_ms) and tau_syn_ex_ms > 0
            and tau_syn_ex_ms == tau_syn_in_ms):
        raise ValueError('Combined current requires equal positive synaptic time constants')
    voltage = _float64(voltage_mv, 'voltage_mv')
    excitation = _float64(current_ex_pa, 'current_ex_pa')
    inhibition = _float64(current_in_pa, 'current_in_pa')
    cells = _cells(senders, cell_count)
    if not len(cells) == len(voltage) == len(excitation) == len(inhibition):
        raise ValueError('Final state lengths differ')
    if len(np.unique(cells)) != len(cells):
        raise ValueError('Duplicate final state cell')
    with np.errstate(over='raise', invalid='raise'):
        current = (excitation+inhibition)*1e-12
    result = np.empty(len(cells), dtype=FINAL_DTYPE)
    result['cell'], result['voltage_V'], result['current_A'] = cells, voltage*1e-3, current
    return result
