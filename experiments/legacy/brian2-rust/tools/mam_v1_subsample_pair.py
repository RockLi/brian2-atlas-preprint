"""Two sequential primary-data analyses, one3600 s budget, no retries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def run(source,output):
    started=time.monotonic()
    protocol_hash=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest()
    assert protocol_hash=='52abbb497cb7685a73984bb87b6c1bd7cb165d4792e0b9a6d4ebd00cd53dec88'
    output.mkdir(exist_ok=False);hashed=0
    reports={}
    with (output/'stages.jsonl').open('x') as journal:
        for name in ['rust','native']:
            limit=min(1800,int(3600-(time.monotonic()-started)-30));assert limit>0
            command=[sys.executable,str(source/'tools/analyze_mam_v1_subsample.py'),
                     '--simulator',name,'--protocol',str(source/'protocol.json'),
                     '--normalization',str(source/'normalization.json'),'--output',str(output/name)]
            journal.write(json.dumps(dict(stage=name,state='started',timeout_seconds=limit,command=command))+'\n');journal.flush()
            begin=time.monotonic()
            with (output/(name+'.log')).open('x') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=limit)
            journal.write(json.dumps(dict(stage=name,state='terminal',returncode=result.returncode,seconds=time.monotonic()-begin))+'\n');journal.flush()
            if result.returncode:raise RuntimeError(name+' diagnostic failed; no retry')
            report=json.loads((output/name/'report.json').read_text())
            assert report['frozen_full_population_histograms_exact'] and report['selected_endpoint_identity_exact']
            assert report['protocol_sha256']==hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest()
            hashed+=report['hashed_input_bytes'];assert hashed<=160*2**30
            reports[name]=hashlib.sha256((output/name/'report.json').read_bytes()).hexdigest()
    assert time.monotonic()-started<3600
    assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<16*2**20
    (output/'report.json').write_text(json.dumps(dict(analysis_complete=True,report_sha256=reports,
        elapsed_seconds=time.monotonic()-started,execution_budget_seconds=3600,automatic_retry=False,
        scientific_acceptance=False,performance_cost_acceptance=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
