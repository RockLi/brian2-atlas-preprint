import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest
from flywire_mnist_game import handler_for


@pytest.fixture
def app(tmp_path):
    received=[]
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path,self.headers.get('Origin'),json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            body=json.dumps({'predicted_digit':7,'actual_fresh_simulation':True}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    up=f'http://127.0.0.1:{upstream.server_port}'
    (tmp_path/'index.html').write_text('<h1>FlyLab</h1>')
    (tmp_path/'private.py').write_text('private source')
    (tmp_path/'fly.bin.gz').write_bytes(gzip.compress(b'geometry',mtime=0))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(tmp_path,up,tmp_path/"recordings"))
    threads=[threading.Thread(target=s.serve_forever,daemon=True) for s in [upstream,server]]
    for t in threads:t.start()
    base=f'http://127.0.0.1:{server.server_port}'
    yield base,up,received
    for s in [server,upstream]:s.shutdown();s.server_close()
    for t in threads:t.join(timeout=2)


def request(base,path='/',value=None,headers=None):
    hs={'Origin':base,'Content-Type':'application/json',**(headers or {})}
    data=json.dumps(value).encode() if value is not None else None
    req=urllib.request.Request(base+path,data=data,headers=hs)
    try:
        with urllib.request.urlopen(req,timeout=3) as response:return response.status,response.headers,response.read()
    except urllib.error.HTTPError as e:return e.code,e.headers,e.read()


def test_proxy_preserves_pixels_and_never_sends_answer(app):
    base,up,received=app
    value={'pixels':[0]*784,'normalize':False}
    status,_,body=request(base,'/api/predict',value)
    assert status==200 and json.loads(body)['predicted_digit']==7
    assert received==[('/predict',up,value)]


def test_rejects_labels_foreign_origin_and_invalid_pixels(app):
    base,_,received=app
    for value,headers,expected in [({'pixels':[0]*784,'label':5},{},400),
            ({'pixels':[0]*784},{'Origin':'https://unrelated.example'},403),
            ({'pixels':[True]*784},{},400),({'pixels':[0]*783},{},400),
            ({'pixels':[0]*784,'normalize':'false'},{},400)]:
        assert request(base,'/api/predict',value,headers)[0]==expected
    assert not received


def test_static_paths_and_host_are_restricted(app):
    base,_,_=app
    assert request(base)[0]==200
    assert request(base,'/private.py')[0]==404
    assert request(base,'/%2e%2e/private.py')[0]==404
    assert request(base,'/',headers={'Host':'attacker.example'})[0]==403


def test_gzip_only_mesh_supports_both_clients(app):
    base,_,_=app
    status,headers,data=request(base,'/fly.bin',headers={'Accept-Encoding':'gzip'})
    assert status==200 and headers['Content-Encoding']=='gzip' and gzip.decompress(data)==b'geometry'
    status,headers,data=request(base,'/fly.bin',headers={'Accept-Encoding':'identity'})
    assert status==200 and 'Content-Encoding' not in headers and data==b'geometry'


def test_recordings_stay_local_and_use_generated_names(app):
    base,_,received=app
    payload=b'\x1aE\xdf\xa3test-recording'
    req=urllib.request.Request(base+'/api/recordings',data=payload,headers={
        'Origin':base,'Content-Type':'video/webm','X-Clip-Format':'landscape'})
    with urllib.request.urlopen(req,timeout=3) as r:
        assert r.status==201
        saved=json.loads(r.read())
    assert saved['name'].startswith('landscape-') and saved['bytes']==len(payload)
    status,_,body=request(base,saved['url'])
    assert status==200 and body==payload
    assert not received
    assert request(base,'/recordings/../../private.py')[0]==404


def test_recording_upload_rejects_foreign_origin_and_bad_media(app):
    base,_,_=app
    for origin,payload,expected in [('https://example.com',b'\x1aE\xdf\xa3x',403),
                                   (base,b'bad-data',400)]:
        req=urllib.request.Request(base+'/api/recordings',data=payload,headers={
            'Origin':origin,'Content-Type':'video/webm','X-Clip-Format':'portrait'})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(req,timeout=3)
        assert error.value.code==expected
