"""Check all unrounded official population denominators with pinned helper."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_mam_paper_time_series import Counts
from validate_mam_paper_rate_bins import HELPER_SHA


def validate(normalization,helper,output):
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA
    m=json.loads(normalization.read_text());generated=normalization.parent/m['generated_data_name']
    assert hashlib.sha256(generated.read_bytes()).hexdigest()==m['generated_data_sha256']=='8c66bb68d55cf2bff222c67716952a2b6260576d6ba0a3650cc150af49f7c2cb'
    data=json.loads(generated.read_text())
    nodes=[n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pop_rate_time_series'];assert len(nodes)==1
    env=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper),'exec'),env)
    ticks=np.array([4999,5000,5001,5004,5005,5014,5015,24999,25000,25001])
    b=Counts(1);b.add(ticks[ticks<=25000],np.zeros(np.count_nonzero(ticks<=25000),dtype=int))
    fractional=0;differences=[]
    for p in m['populations']:
        area,group=p['name'].removeprefix('mam_').rsplit('_',1);n=data['neuron_numbers'][area][group]
        assert n==p['official_normalization_neurons'] and int(n)==p['simulated_neurons'];fractional+=n!=int(n);differences.append((n-int(n))/int(n))
        actual=b.shifted[0]/(n*1./1000.)
        times=ticks*.1;times=times[(times>500)&(times<=2500)]
        expected=env['pop_rate_time_series'](np.column_stack([np.zeros(len(times)),times]),n,500.,2500.,resolution=1.)
        np.testing.assert_array_equal(actual,expected)
    assert len(m['populations'])==fractional==254
    r=dict(passed=True,fixture_populations=254,all_normalized_bins_exact=True,all_populations_fractional=True,
        maximum_relative_denominator_difference=max(differences),actual_neurons=m['actual_neurons'],unrounded_population_sum=m['unrounded_population_sum'],normalization_sha256=hashlib.sha256(normalization.read_bytes()).hexdigest(),generated_data_sha256=m['generated_data_sha256'],official_helper_sha256=HELPER_SHA,
        scope='Exact modern-helper rate-bin fixtures with each official unrounded M.N; simulation population sizes remain integer.')
    output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['normalization','helper','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();validate(a.normalization,a.helper,a.output)
