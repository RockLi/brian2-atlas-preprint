"""Transport failure gates on small local fixtures; never launch remote work."""
import ast,hashlib,io,json,sys,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_t7_raw_backup as m

def expected(data):return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def reader(data):
    stream=io.BytesIO(data)
    return lambda n,t:stream.read(n)

def test_copy_then_independent_readback(tmp_path):
    data=b'neurons\0spikes\xff'*90001;item=expected(data);p=tmp_path/'results.bin';sizes=[]
    copied=m.copy_payload(reader(data),p,item,time.monotonic()+10,lambda n:sizes.append(n))
    result=m.readback(p,item,time.monotonic()+10)
    assert copied==item and result['independent_readback'] and p.read_bytes()==data
    assert max(sizes)<=m.BLOCK and len(sizes)>1

@pytest.mark.parametrize('change',['truncated','extra','wrong-hash'])
def test_payload_failure_retains_unaccepted_file(tmp_path,change):
    data=b'abcd';item=expected(data);payload=data
    if change=='truncated':payload=data[:-1]
    if change=='extra':payload=data+b'e'
    if change=='wrong-hash':item['sha256']='0'*64
    p=tmp_path/'results.bin'
    with pytest.raises(ValueError):m.copy_payload(reader(payload),p,item,time.monotonic()+10,lambda n:None)
    assert p.exists() and not (tmp_path/'complete.json').exists()

def test_no_overwrite(tmp_path):
    p=tmp_path/'results.bin';p.write_bytes(b'keep')
    with pytest.raises(FileExistsError):m.copy_payload(reader(b'x'),p,expected(b'x'),time.monotonic()+10,lambda n:None)
    assert p.read_bytes()==b'keep'

def test_reserve_failure_before_payload_write(tmp_path):
    def fail(n):raise ValueError('reserve')
    p=tmp_path/'results.bin'
    with pytest.raises(ValueError,match='reserve'):m.copy_payload(reader(b'ab'),p,expected(b'ab'),time.monotonic()+10,fail)
    assert p.read_bytes()==b''

def test_readback_detects_corruption_after_copy(tmp_path):
    p=tmp_path/'results.bin';p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='readback mismatch'):m.readback(p,expected(b'abc'),time.monotonic()+10)

def test_deadline_does_not_consume_more_input(tmp_path):
    def unexpected(n,t):pytest.fail('read after deadline')
    with pytest.raises(ValueError,match='deadline'):
        m.copy_payload(unexpected,tmp_path/'results.bin',expected(b'ab'),time.monotonic()-1,lambda n:None)

def test_remote_commands_are_bounded_and_sender_is_readonly():
    item=dict(name='results.bin',**expected(b'ab'))
    cmd,g,r,app=m.source_command(0,item,120)
    assert '--property=MemoryMax=512M' in cmd and '--property=MemorySwapMax=0' in cmd
    assert '--property=CPUQuota=100%' in cmd and '--property=RuntimeMaxSec=125' in cmd
    assert '--property=AllowedCPUs=8-9' in cmd
    tree=ast.parse(app[-1]);assert tree
    assert 'os.O_RDONLY|os.O_NOFOLLOW' in app[-1] and 'os.POSIX_FADV_DONTNEED' in app[-1]
    assert "sys.stdout.buffer.write(block)" in app[-1] and "q.open('x')" in app[-1]
    assert g.name=='b2mpi-t7-raw-rust32-v1-file0.json' and r.name==g.name

def test_oversized_reader_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='oversized'):
        m.copy_payload(lambda n,t:b'abcd',tmp_path/'results.bin',expected(b'ab'),time.monotonic()+10,lambda n:None)

def test_local_memory_headroom_must_be_observed():
    text='The system has 17179869184 (1048576 pages with a page size of 16384).\nSystem-wide memory free percentage: 45%\n'
    assert m.memory_admission(text)['estimated_free_bytes']>7*2**30
    with pytest.raises(ValueError,match='headroom'):m.memory_admission(text.replace('45%','5%'))
    with pytest.raises(ValueError,match='unrecognized'):m.memory_admission('unavailable')
