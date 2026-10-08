"""Assemble a static, backend-free FlyLab zip while retaining remote mode."""
import argparse,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def install_model(model_dir,aot_dir,destination):
    destination.mkdir(parents=True,exist_ok=True);m=json.loads((model_dir/'model.json').read_text());build=json.loads((aot_dir/'build.json').read_text())
    assert m['files']['snapshot.bin']['sha256']==build['snapshot_sha256'] and m['source_sha256']==build['source_sha256']
    for name,info in m['files'].items():
        source=model_dir/(name+'.gz');raw=gzip.decompress(source.read_bytes());assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
        packed=name.split('.')[0]+'.packed.bin';info['url']=packed;info['encoding']='gzip';info['download_bytes']=source.stat().st_size
        shutil.copyfile(source,destination/packed)
    (destination/'pkg').mkdir(exist_ok=True)
    for name in ['flywire_browser_aot.js'] :shutil.copyfile(aot_dir/'pkg'/name,destination/'pkg'/name)
    shutil.copyfile(aot_dir/'pkg/flywire_browser_aot_bg.wasm',destination/'pkg/engine.bin')
    m['wasm_sha256']=digest(destination/'pkg/engine.bin');assert m['wasm_sha256']==build['wasm_sha256']
    (destination/'model.json').write_text(json.dumps(m,separators=(',',':')))
    shutil.copyfile(aot_dir/'build.json',destination/'build.json')
    return m

def package(model_dir,aot_dir,output):
    if output.exists():raise ValueError('Choose a new output directory')
    ui=ROOT/'validation/flywire_game'
    shutil.copytree(ui,output,ignore=shutil.ignore_patterns('browser-model','__pycache__','.gitignore','build_assets.py','digits.json'))
    m=install_model(model_dir,aot_dir,output/'browser-model')
    html=(output/'index.html').read_text().replace('<html lang="en"','<html lang="en" data-runtime="local"')
    html=html.replace('</body>','<script type="module" src="./offline-register.js"></script></body>');(output/'index.html').write_text(html)
    # Any static host can serve the anatomy; no custom gzip/proxy handler needed.
    mesh=output/'assets/fly.bin'
    if not mesh.exists():mesh.write_bytes(gzip.decompress((output/'assets/fly.bin.gz').read_bytes()))
    (output/'offline-register.js').write_text("if('serviceWorker' in navigator)navigator.serviceWorker.register('./offline-sw.js').catch(()=>{});\n")
    paths=['./']+['./'+p.relative_to(output).as_posix() for p in sorted(output.rglob('*')) if p.is_file() and not p.name.endswith('.packed.bin')]
    version=hashlib.sha256(''.join(digest(p) for p in sorted(output.rglob('*')) if p.is_file()).encode()).hexdigest()[:16]
    sw="const PREFIX='flylab-shell-'+encodeURIComponent(self.registration.scope)+'-',CACHE=PREFIX+'VERSION',ASSETS=ASSETS_JSON;\n".replace('VERSION',version).replace('ASSETS_JSON',json.dumps(paths))
    sw+='''self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS))));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key.startsWith(PREFIX)&&key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
 const url=new URL(event.request.url);
 if(event.request.method!=='GET'||url.origin!==location.origin||url.pathname.includes('/api/'))return;
 event.respondWith(fetch(event.request).catch(()=>caches.open(CACHE).then(cache=>cache.match(event.request)).then(response=>response||Response.error())));
});
'''
    (output/'offline-sw.js').write_text(sw)
    (output/'README.txt').write_text('''FlyLab — standalone browser edition

Serve this folder from any static HTTPS host (or localhost). No inference
backend, Python runtime, Rust installation, SSH connection or model API is
needed by visitors. Do not open index.html directly via file://.

Default: This device. The complete FlyWire model runs in a browser Worker.
Remote server mode remains available when hosted with the companion API.
The distribution itself contains no inference server.

The first model load downloads about 54 MB. Desktop browsers are recommended:
WASM linear memory alone is about 534 MiB, plus readout/JavaScript/rendering.
First-run compilation is slower. Performance depends on the device.

Once the page and model have loaded successfully, supported browsers cache the
assets for offline use. Browser storage eviction/private mode can remove this
cache; the unpacked distribution always remains available for static serving.
No handwriting images or prediction results are stored in the asset cache.

The model has fixed internal connections and weights. It uses artificial input
and a trained readout; the fly's movements/laser are visualization. The arcade samples all 10,000 official MNIST test images in a random order,
without repeats within a pass. Reloading starts a new shuffle. Session hits
and misses are not a completed full-test accuracy result. See
assets/mnist-test.json for source file checksums and dataset attribution.

Data attribution: see DATA-ATTRIBUTION.json (FlyWire v783, CC BY 4.0).
Licenses: see assets/LICENSE and vendor/LICENSE. Engine source is distributed
under the repository's CeCILL-2.1 license; see ENGINE-LICENSE and SOURCE-NOTICE.
''')
    for candidate in [ROOT/'LICENSE',ROOT.parent/'LICENSE']:
        if candidate.exists():shutil.copyfile(candidate,output/'ENGINE-LICENSE');break
    sources=output/'engine-source';sources.mkdir();(sources/'src').mkdir()
    shutil.copyfile(aot_dir/'crate/src/lib.rs',sources/'src/lib.rs');shutil.copyfile(aot_dir/'crate/src/browser_host.rs',sources/'src/browser_host.rs')
    shutil.copyfile(aot_dir/'crate/Cargo.toml',sources/'Cargo.toml');shutil.copyfile(aot_dir/'crate/Cargo.lock',sources/'Cargo.lock')
    provenance=json.loads((ROOT/'wasm/flywire-circuit.json').read_text())
    attribution={key:provenance[key] for key in ['release','license','source','paper','annotation_source','source_csr_sha256','original_csr_sha256']}
    attribution['modifications']='Source-major CSR conversion, signed contact weights, fixed conductance LIF model, artificial ALPN inputs and trained MNIST readout. This package contains the full graph, not the separate DM1 browser subgraph.'
    (output/'DATA-ATTRIBUTION.json').write_text(json.dumps(attribution,indent=2))
    (output/'SOURCE-NOTICE').write_text('The engine is compiled from the frozen Brian2 Rust AOT source.\nGenerated source, host adapter and build manifest are included.\nRebuild with cargo (wasm32-unknown-unknown) and wasm-bindgen-cli 0.2.100.\n')
    manifest={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'distribution.json').write_text(json.dumps(dict(protocol=m['protocol'],default_runtime='local',files=manifest),indent=2))
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(output.parent))
    print(json.dumps(dict(directory=str(output),zip=str(output.with_suffix('.zip')),zip_bytes=output.with_suffix('.zip').stat().st_size)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','aot','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();package(a.model,a.aot,a.output)
