"""Check the retained full-MAM spike and recorded-state prefix exactly.

Call only after both binary dumps pass the independent result reader. Final
states of unrecorded cells and past synaptic deliveries are not reconstructed.
"""
import numpy as np


def exact_array(left, right, *, block=131072):
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError('prefix array type or shape differs')
    for offset in range(0, len(left), block):
        if left[offset:offset+block].tobytes() != right[offset:offset+block].tobytes():
            raise ValueError('prefix array bytes differ')


def compare_populations(old, new, *, end_tick=25000):
    if type(end_tick) is not int or end_tick <= 0 or len(old) != len(new):
        raise ValueError('invalid prefix boundary or population count')
    rows=[]
    for i,(prior,current) in enumerate(zip(old,new,strict=True)):
        ticks=prior['spike_ticks']
        if np.any(ticks<0) or np.any(ticks>=end_tick):
            raise ValueError('baseline spikes exceed prefix boundary')
        end=int(np.searchsorted(current['spike_ticks'],end_tick,side='left'))
        exact_array(ticks,current['spike_ticks'][:end])
        exact_array(prior['indices'],current['indices'][:end])
        counts=np.bincount(current['indices'][:end],minlength=len(prior['counts']))
        exact_array(prior['counts'],counts.astype(prior['counts'].dtype))
        if set(prior['event_streams']) != {'spike'} or set(current['event_streams']) != {'spike'}:
            raise ValueError('requires the frozen spike-only full MAM')
        if prior['event_monitors'] or current['event_monitors']:
            raise ValueError('custom monitors require their own prefix contract')
        a,b=prior['event_streams']['spike'],current['event_streams']['spike']
        event_end=int(np.searchsorted(b['ticks'],end_tick,side='left'))
        if event_end != end:
            raise ValueError('event stream prefix count differs')
        for key in ['ticks','indices']:exact_array(a[key],b[key][:event_end])
        if set(prior['trace']) != set(current['trace']):
            raise ValueError('recorded variables differ')
        trace_values=0
        for key,values in prior['trace'].items():
            if len(values) != end_tick or len(current['trace'][key]) < end_tick:
                raise ValueError('requires full state-monitor prefix')
            exact_array(values,current['trace'][key][:end_tick])
            trace_values+=values.size
        rows.append(dict(population=i,spikes=end,recorded_state_values=trace_values))
    return dict(exact=True,end_tick_exclusive=end_tick,populations=rows,
                spikes=sum(r['spikes'] for r in rows),
                recorded_state_values=sum(r['recorded_state_values'] for r in rows),
                scope='All retained spikes, spike event streams, per-cell prefix counts and recorded state values are byte-exact. Unrecorded historical states and synaptic delivery histories are not inferred.')


def compare_populations_bounded(old, new, *, end_tick, old_release,
        new_release, old_event_release, new_event_release, block=131072):
    """Integer-exact event prefix across i64/u32 formats; byte-exact states.

    Both dumps must first pass the independent reader. Callbacks release only
    views of the corresponding result/event mapping. They must not close the
    mappings. Temporary payloads are bounded by ``block`` scalar elements;
    cell counters are bounded by the frozen population size, not spike count.
    """
    if (type(end_tick) is not int or not 0 < end_tick <= 1005000
            or len(old) != len(new) or not 1 <= len(old) <= 254
            or type(block) is not int or not 1 <= block <= 131072
            or any(not callable(f) for f in [old_release,new_release,old_event_release,new_event_release])):
        raise ValueError('invalid bounded prefix contract')
    integer_dtypes = {np.dtype('<i8'), np.dtype('<u4')}
    mixed = False

    def chunks(left,right,release_left,release_right,*,integer=False,ticks=False):
        nonlocal mixed
        if left.shape != right.shape or left.ndim not in (1,2):
            raise ValueError('prefix array shape differs')
        if integer:
            if left.ndim != 1 or left.dtype not in integer_dtypes or right.dtype not in integer_dtypes:
                raise ValueError('prefix events require i64 or u32 indices')
            mixed |= left.dtype != right.dtype
        elif left.dtype != right.dtype:
            raise ValueError('prefix array type differs')
        width = left.shape[1] if left.ndim==2 else 1
        if not 1 <= width <= block:
            raise ValueError('recorded row exceeds bounded prefix buffer')
        for offset in range(0,len(left),max(1,block//width)):
            a=left[offset:offset+block//width];b=right[offset:offset+block//width]
            try:
                if ticks and (np.any(a<0) or np.any(a>=end_tick)):
                    raise ValueError('baseline spikes exceed prefix boundary')
                if integer:
                    if np.any(a<0) or np.any(b<0) or not np.array_equal(a,b):
                        raise ValueError('prefix integer values differ')
                elif a.tobytes()!=b.tobytes():
                    raise ValueError('prefix array bytes differ')
            finally:
                for values,release in [(a,release_left),(b,release_right)]:
                    if values.ndim==2:
                        if not values.flags.c_contiguous:
                            raise ValueError('recorded state must be a contiguous reader view')
                        values=values.reshape(-1)
                    release(values)

    rows=[]
    for i,(prior,current) in enumerate(zip(old,new,strict=True)):
        if set(prior['event_streams'])!={'spike'} or set(current['event_streams'])!={'spike'}:
            raise ValueError('requires the frozen spike-only full MAM')
        if prior['event_monitors'] or current['event_monitors']:
            raise ValueError('custom monitors require their own prefix contract')
        n=len(prior['counts'])
        if not 1<=n<=4200000 or current['counts'].shape!=prior['counts'].shape or prior['counts'].dtype!=np.dtype('<i8'):
            raise ValueError('invalid full-MAM population count contract')
        # The independently validated old dump tells us the exact prefix size.
        # searchsorted can copy an entire *unaligned* binary-file field; avoid
        # reading/allocating the long tail merely to find this boundary.
        end=len(prior['spike_ticks'])
        if end<len(current['spike_ticks']) and current['spike_ticks'][end]<end_tick:
            raise ValueError('new dump has extra spikes inside the prefix')
        chunks(prior['spike_ticks'],current['spike_ticks'][:end],old_release,new_release,integer=True,ticks=True)
        chunks(prior['indices'],current['indices'][:end],old_release,new_release,integer=True)
        counts=np.zeros(n,dtype='<i8')
        for offset in range(0,end,block):
            values=current['indices'][offset:min(offset+block,end)]
            try:
                if np.any(values<0) or np.any(values>=n):raise ValueError('prefix neuron outside population')
                counts+=np.bincount(values.astype(np.int64),minlength=n)
            finally:new_release(values)
        chunks(prior['counts'],counts,old_release,lambda values:None)
        a,b=prior['event_streams']['spike'],current['event_streams']['spike']
        event_end=len(a['ticks'])
        if event_end<len(b['ticks']) and b['ticks'][event_end]<end_tick:
            raise ValueError('new dump has extra stream events inside the prefix')
        if event_end!=end:raise ValueError('event stream prefix count differs')
        for key in ['ticks','indices']:
            chunks(a[key],b[key][:event_end],old_event_release,new_event_release,integer=True,ticks=key=='ticks')
        if set(prior['trace'])!=set(current['trace']):raise ValueError('recorded variables differ')
        trace_values=0
        for key,values in prior['trace'].items():
            if len(values)!=end_tick or len(current['trace'][key])<end_tick:
                raise ValueError('requires full state-monitor prefix')
            chunks(values,current['trace'][key][:end_tick],old_release,new_release)
            trace_values+=values.size
        rows.append(dict(population=i,spikes=end,recorded_state_values=trace_values))
    return dict(exact=True,end_tick_exclusive=end_tick,populations=rows,
        spikes=sum(r['spikes'] for r in rows),recorded_state_values=sum(r['recorded_state_values'] for r in rows),
        mixed_event_index_widths=bool(mixed),event_integer_values_exact=True,
        counts_and_recorded_state_bytes_exact=True,maximum_block_elements=block,
        scope='All retained event ticks/indices are integer-exact across audited i64/u32 layouts; per-cell counts and recorded states are byte-exact. This is not whole-file byte equality or evidence about unrecorded historical states, deliveries or scientific equivalence.')
