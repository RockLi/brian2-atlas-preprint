"""Loopback-only visual research viewer with optional fresh CPU simulations."""
import argparse
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse
import uuid
import numpy as np
from .run_experiment import array64, diagnostic, save
from .simulation import SimulationConfig, encode_movie, summarize
from .stimuli import KINDS

ALLOWED = set(KINDS) | {"blank", "flash", "bright", "dark"}


def validate_request(value):
    if not isinstance(value,dict) or set(value)!={"kind","condition"}:
        raise ValueError("expected only kind and condition")
    if not isinstance(value["kind"],str) or value["kind"] not in ALLOWED:
        raise ValueError("unknown stimulus")
    if not isinstance(value["condition"],str) or value["condition"] not in ("intact","cut"):
        raise ValueError("unknown circuit condition")
    return value["kind"],value["condition"]


def load_audit(directory, decoder, pilot):
    """Bind an optional robustness report to this exact frozen decoder study."""
    if decoder is None or pilot is None:
        raise ValueError('an audit requires its parent direction readout')
    root, parent = Path(directory), Path(pilot)
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report, protocol = read(root/'report.json'), read(root/'protocol.json')
    if (report.get('status') != 'complete' or report.get('schema') != 'flywire-robustness-audit-v1' or
            report.get('schema') != protocol.get('schema')):
        raise ValueError('audit requires a completed robustness report')
    if sha(root/'protocol.json') != report.get('protocol_sha256'):
        raise ValueError('audit protocol changed')
    if (sha(parent/'report.json') != protocol.get('parent_report_sha256') or
            sha(Path(protocol['parent_study'])/'report.json') != protocol.get('parent_report_sha256')):
        raise ValueError('audit does not belong to the displayed direction study')
    for key in ('base_sha256','binary_sha256','config','input_mode','readout_indices','input_indices'):
        if protocol.get(key) != decoder.protocol.get(key):
            raise ValueError('audit and displayed readout disagree: '+key)
    expected = protocol.get('readouts_sha256', {})
    if set(expected) != {'neural','input_neurons','encoded_input'}:
        raise ValueError('audit requires all three frozen layer readouts')
    if expected.get('neural') != decoder.readout_hash:
        raise ValueError('audit uses a different primary decoder')
    for name, digest in expected.items():
        if name not in ('neural','input_neurons','encoded_input'):
            raise ValueError('unknown audit readout')
        if (decoder.report['readouts'].get(name, {}).get('readout_sha256') != digest or
                sha(parent/(name+'-readout.npz')) != digest):
            raise ValueError('audit readout artifact changed')
    if sha(root/'features.npz') != report.get('features_sha256'):
        raise ValueError('audit features changed')
    if (set(report.get('conditions', {})) != set(protocol.get('conditions', {})) or
            report.get('groups_per_condition') != len(protocol.get('groups', [])) or
            report.get('clips_per_condition') != protocol.get('clips_per_condition')):
        raise ValueError('audit report does not cover its declared conditions')
    return report


class Experiment:
    def __init__(self, artifact, live=False, pilot=None, audit=None, causal=None, refinement=None, multispeed=None, generalization=None):
        self.artifact=Path(artifact).resolve()
        self.data=json.loads((self.artifact/"viewer-data.json").read_text())
        if self.data["manifest"]["status"]!="complete":
            raise ValueError("viewer requires completed, checked experiment data")
        self.trials={(t["condition"],t["kind"]):t for t in self.data["trials"]}
        self.lock=threading.Lock()
        self.live=live
        self.decoder=None
        self.audit=None
        self.generalization=None
        if generalization:
            from .generalization_replay import GeneralizationReplay
            self.generalization=GeneralizationReplay(generalization)
        self.multispeed=None
        if multispeed:
            from .speed_replay import SpeedReplay
            self.multispeed=SpeedReplay(multispeed)
        self.refinement=None
        if refinement:
            from .motion_replay import MotionReplay
            self.refinement=MotionReplay(refinement)
        self.causal=None
        if causal:
            from .causal_view import CausalReplay
            self.causal=CausalReplay(causal)
        if pilot:
            from .decoding import Decoder
            identity=json.loads((self.artifact/'intact/identity.json').read_text())
            self.decoder=Decoder(pilot,identity['base_sha256'],self.data['manifest']['neurons'])
            predictions=json.loads((Path(pilot)/'viewer-predictions.json').read_text())
            for trial in self.trials.values():
                prediction=predictions[trial['id']]
                trial['classification']=self.decoder.cached_prediction(prediction,trial['summary']['neural_event_sha256'])
        if audit:
            self.audit=load_audit(audit,self.decoder,pilot)
        if not live:return
        from flywire_mnist.backends.frozen_cpu import FrozenCPU
        self.cfg=SimulationConfig(**self.data["manifest"]["config"])
        self.models={condition:json.loads((self.artifact/filename).read_text())
                     for condition,filename in (("intact","model.json"),("cut","model-cut.json"))}
        self.work=self.artifact.parent/("viewer-live-"+uuid.uuid4().hex[:12])
        self.work.mkdir(exist_ok=False)
        self.runners={condition:FrozenCPU(model,self.artifact/"compile/native",self.work/condition,
                                         population="visual_input",threads=self.data["manifest"]["threads"])
                      for condition,model in self.models.items()}
        for condition,runner in self.runners.items():
            original=json.loads((self.artifact/condition/"identity.json").read_text())
            if runner.base_hash!=original["base_sha256"] or runner.binary_hash!=original["binary_sha256"]:
                raise ValueError("viewer model or executable differs from the checked experiment")
        # Store indices explicitly; do not reconstruct readouts from display labels.
        self.groups={k:np.asarray(v,dtype=int) for k,v in json.loads((self.artifact/"groups.json").read_text()).items()}

    def run(self,kind,condition):
        if not self.live:raise ValueError("server is in recorded playback mode")
        if not self.lock.acquire(blocking=False):raise BlockingIOError("已有一次仿真正在运行，请稍候重试。")
        try:
            channels=self.data["channels"]
            frames=diagnostic(kind,self.cfg.seed)
            from .refinement import encode_variant
            indices,ticks,strength=encode_variant(frames,channels,self.cfg,self.data['manifest'].get('input_mode','contrast'))
            key="live-"+uuid.uuid4().hex
            result=self.runners[condition].run(indices,ticks,key)
            summary,counts=summarize(result,self.models[condition],self.groups,channels,self.cfg)
            if np.any(counts>65535):raise ValueError("display count overflow")
            trial={"id":key,"live":True,"kind":kind,"condition":condition,"seed":self.cfg.seed,"summary":summary,
                   "input_spikes":len(indices),"input_sha256":hashlib.sha256(np.stack([ticks,indices],axis=1).astype('<i8').tobytes()).hexdigest(),
                   "movie_sha256":hashlib.sha256(frames.tobytes()).hexdigest(),
                   "frames_u8":array64(np.rint(frames*255),"u1"),"frame_shape":list(frames.shape),
                   "strength_u8":array64(np.rint(strength*255),"u1"),
                   "input_counts_u16":array64(counts,"<u2"),"count_shape":list(counts.shape)}
            if self.decoder:
                ni=next(i for i,p in enumerate(self.models[condition]['definition']['populations']) if p['name']=='flywire_neurons')
                trial['classification']=self.decoder.predict(result['populations'][ni],self.data['manifest']['neurons'])
            save(self.work/(key+".json"),trial)
            self.trials[condition,kind]=trial
            return trial
        finally:self.lock.release()


def make_handler(experiment, port):
    assets=Path(__file__).with_name("viewer")
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,data,mime="application/json; charset=utf-8"):
            if not isinstance(data,bytes):data=json.dumps(data,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url=urlparse(self.path)
            if url.path=="/favicon.ico":return self.respond(204,b"", "image/x-icon")
            files={"/motion-generalization":("generalization.html","text/html; charset=utf-8"),"/generalization.js":("generalization.js","text/javascript; charset=utf-8"),"/motion-speed":("speed.html","text/html; charset=utf-8"),"/speed.js":("speed.js","text/javascript; charset=utf-8"),"/":("index.html","text/html; charset=utf-8"),"/app.js":("app.js","text/javascript; charset=utf-8"),"/style.css":("style.css","text/css; charset=utf-8"),"/motion":("motion.html","text/html; charset=utf-8"),"/motion-v2":("motion.html","text/html; charset=utf-8"),"/motion.js":("motion.js","text/javascript; charset=utf-8"),"/motion.css":("motion.css","text/css; charset=utf-8")}
            if url.path in files:
                name,mime=files[url.path];return self.respond(200,(assets/name).read_bytes(),mime)
            if url.path=="/api/index":return self.respond(200,{"manifest":experiment.data["manifest"],"channels":experiment.data["channels"],"live":experiment.live,"pilot":experiment.decoder.report if experiment.decoder else None,"audit":experiment.audit,"causal_available":experiment.causal is not None,"refinement_available":experiment.refinement is not None,"multispeed_available":experiment.multispeed is not None,"generalization_available":experiment.generalization is not None})
            if url.path=="/api/manifest":return self.respond(200,experiment.data["manifest"])
            if url.path=="/api/readout":return self.respond(200,experiment.decoder.report if experiment.decoder else None)
            if url.path=="/api/audit":return self.respond(200,experiment.audit)
            if url.path in ("/api/causal/index","/api/causal/trial","/api/refinement/index","/api/refinement/trial","/api/multispeed/index","/api/multispeed/trial","/api/generalization/index","/api/generalization/trial"):
                replay=experiment.generalization if url.path.startswith("/api/generalization/") else (experiment.multispeed if url.path.startswith("/api/multispeed/") else (experiment.refinement if url.path.startswith("/api/refinement/") else experiment.causal))
                if replay is None:return self.respond(404,{"error":"新周期运动实验尚未载入"})
                if url.path.endswith('/index'):return self.respond(200,replay.index())
                try:
                    args=parse_qs(url.query)
                    if set(args)!={'condition','group','i','j','kind'} or any(len(v)!=1 for v in args.values()):raise ValueError('invalid causal query')
                    trial=replay.trial(args['condition'][0],int(args['group'][0]),int(args['i'][0]),int(args['j'][0]),args['kind'][0])
                    return self.respond(200,trial)
                except ValueError as error:return self.respond(400,{"error":str(error)})
            if url.path=="/api/trial":
                args=parse_qs(url.query)
                try:
                    if set(args)!={"condition","kind"} or any(len(v)!=1 for v in args.values()):raise ValueError("invalid query")
                    kind,condition=validate_request({k:v[0] for k,v in args.items()})
                    trial=experiment.trials.get((condition,kind))
                    return self.respond(200,trial) if trial else self.respond(404,{"error":"此条件尚无仿真记录"})
                except ValueError as error:return self.respond(400,{"error":str(error)})
            return self.respond(404,{"error":"not found"})

        def do_POST(self):
            if self.path!="/api/run":return self.respond(404,{"error":"not found"})
            allowed={f"http://127.0.0.1:{port}",f"http://localhost:{port}"}
            if self.headers.get("Origin") not in allowed|{None}:
                return self.respond(403,{"error":"cross-origin request rejected"})
            if self.headers.get("Content-Type","").split(";")[0]!="application/json":
                return self.respond(415,{"error":"expected application/json"})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=2048:raise ValueError("invalid request size")
                kind,condition=validate_request(json.loads(self.rfile.read(length)))
                result=experiment.run(kind,condition)
                return self.respond(200,result)
            except BlockingIOError as error:return self.respond(409,{"error":str(error)})
            except (ValueError,UnicodeError) as error:return self.respond(400,{"error":str(error)})
            except Exception as error:return self.respond(500,{"error":str(error)})
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact",type=Path,required=True)
    parser.add_argument("--port",type=int,default=18771)
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--pilot",type=Path)
    parser.add_argument("--audit",type=Path)
    parser.add_argument("--causal",type=Path)
    parser.add_argument("--refinement",type=Path)
    parser.add_argument("--multispeed",type=Path)
    parser.add_argument("--generalization",type=Path)
    args=parser.parse_args()
    experiment=Experiment(args.artifact,args.live,args.pilot,args.audit,args.causal,args.refinement,args.multispeed,args.generalization)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(experiment,args.port))
    print(f"Vision Lab http://127.0.0.1:{args.port}/ live={args.live}",flush=True)
    server.serve_forever()


if __name__=="__main__":main()
