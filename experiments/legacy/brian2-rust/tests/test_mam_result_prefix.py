import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('mam_result_prefix',Path(__file__).resolve().parents[1]/'tools/mam_result_prefix.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def population(ticks,indices,steps):
    ticks=np.array(ticks,dtype='<i8');indices=np.array(indices,dtype='<i8')
    return dict(spike_ticks=ticks,indices=indices,counts=np.bincount(indices,minlength=2).astype('<i8'),
                trace={'v':np.arange(steps,dtype='<f8').reshape(-1,1)},
                event_streams={'spike':dict(ticks=ticks.copy(),indices=indices.copy())},event_monitors={})


def test_boundary_event_is_excluded_and_all_recorded_values_checked():
    old=population([0,3],[0,1],4);new=population([0,3,4,5],[0,1,1,0],6)
    result=module.compare_populations([old],[new],end_tick=4)
    assert result['exact'] and result['spikes']==2 and result['recorded_state_values']==4


@pytest.mark.parametrize('field',['spike','event','trace','signed-zero','missing'])
def test_changed_prefix_is_rejected(field):
    old=population([0,3],[0,1],4);new=population([0,3,4,5],[0,1,1,0],6)
    if field=='spike':new['indices'][1]=0
    elif field=='event':new['event_streams']['spike']['indices'][1]=0
    elif field=='trace':new['trace']['v'][2,0]=np.nextafter(2.,3.)
    elif field=='signed-zero':new['trace']['v'][0,0]=-0.
    else:new['trace']={}
    with pytest.raises(ValueError):module.compare_populations([old],[new],end_tick=4)


def convert_indices(pop,dtype):
    for key in ['spike_ticks','indices']:pop[key]=pop[key].astype(dtype)
    for key in ['ticks','indices']:pop['event_streams']['spike'][key]=pop['event_streams']['spike'][key].astype(dtype)
    return pop


@pytest.mark.parametrize('old_dtype,new_dtype',[('<i8','<i8'),('<u4','<u4'),('<i8','<u4'),('<u4','<i8')])
@pytest.mark.parametrize('block',[1,3,131072])
def test_bounded_prefix_preserves_values_across_exact_index_widths(old_dtype,new_dtype,block):
    old=convert_indices(population([0,3],[0,1],4),old_dtype);new=convert_indices(population([0,3,4,5],[0,1,1,0],6),new_dtype)
    released=[]
    def release(values):
        assert values.ndim==1 and values.size<=block
        released.append(values.size)
    result=module.compare_populations_bounded([old],[new],end_tick=4,old_release=release,new_release=release,old_event_release=release,new_event_release=release,block=block)
    assert result['exact'] and result['event_integer_values_exact'] and result['counts_and_recorded_state_bytes_exact']
    assert result['spikes']==2 and result['recorded_state_values']==4 and released
    assert result['mixed_event_index_widths']==(old_dtype!=new_dtype)


@pytest.mark.parametrize('field',['spike','event','trace','signed-zero','count','negative','missing','float'])
def test_bounded_prefix_never_relaxes_corruption_checks(field):
    old=population([0,3],[0,1],4);new=convert_indices(population([0,3,4,5],[0,1,1,0],6),'<u4')
    if field=='spike':new['indices'][1]=0
    elif field=='event':new['event_streams']['spike']['indices'][1]=0
    elif field=='trace':new['trace']['v'][2,0]=np.nextafter(2.,3.)
    elif field=='signed-zero':new['trace']['v'][0,0]=-0.
    elif field=='count':old['counts'][0]+=1
    elif field=='negative':old['indices'][0]=-1
    elif field=='float':new['spike_ticks']=new['spike_ticks'].astype(float)
    else:new['trace']={}
    with pytest.raises(ValueError):module.compare_populations_bounded([old],[new],end_tick=4,old_release=lambda a:None,new_release=lambda a:None,old_event_release=lambda a:None,new_event_release=lambda a:None,block=1)


@pytest.mark.parametrize('end,block',[(0,1),(1005001,1),(True,1),(4,0),(4,131073),(4,True)])
def test_bounded_prefix_rejects_unadmitted_boundaries(end,block):
    old=population([0,3],[0,1],4);new=population([0,3,4,5],[0,1,1,0],6)
    with pytest.raises(ValueError):module.compare_populations_bounded([old],[new],end_tick=end,old_release=lambda a:None,new_release=lambda a:None,old_event_release=lambda a:None,new_event_release=lambda a:None,block=block)


def test_unaligned_long_tail_is_not_copied_or_searched(monkeypatch):
    import tracemalloc
    old=population([0,3],[0,1],4)
    ticks=np.r_[0,3,np.arange(4,300004)];ids=np.arange(len(ticks))%2
    new=population(ticks,ids,300004)
    storage=bytearray(len(ticks)*16+1)
    values=np.ndarray((len(ticks),),dtype=[('tick','<i8'),('id','<i8')],buffer=storage,offset=1)
    values['tick']=ticks;values['id']=ids
    new['spike_ticks']=values['tick'];new['indices']=values['id']
    new['event_streams']['spike']=dict(ticks=values['tick'],indices=values['id'])
    assert not values['tick'].flags.aligned
    def no_whole_tail_search(*a,**k):raise AssertionError('unaligned whole-tail search attempted')
    monkeypatch.setattr(np,'searchsorted',no_whole_tail_search)
    tracemalloc.start()
    try:
        result=module.compare_populations_bounded([old],[new],end_tick=4,old_release=lambda a:None,new_release=lambda a:None,old_event_release=lambda a:None,new_event_release=lambda a:None,block=3)
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert result['exact'] and result['spikes']==2
    assert peak<512*1024,peak # Copying the unaligned tick tail would need >2 MiB.


@pytest.mark.parametrize('stream',[False,True])
def test_extra_prefix_event_at_validated_baseline_length_is_rejected(stream):
    old=population([0,3],[0,1],4);new=population([0,3,3,5],[0,1,0,0],6)
    if stream:new['spike_ticks']=np.array([0,3,4,5],dtype='<i8')
    with pytest.raises(ValueError,match='extra'):
        module.compare_populations_bounded([old],[new],end_tick=4,old_release=lambda a:None,new_release=lambda a:None,old_event_release=lambda a:None,new_event_release=lambda a:None)
