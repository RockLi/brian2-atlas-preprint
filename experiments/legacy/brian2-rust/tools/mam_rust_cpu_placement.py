"""One evidence-bound CPU relocation; old model and rank ownership stay fixed."""
from mam_primary_resources import CPUS,NODES
from mam_launch_native_primary import check,read,sha

ALTERNATE_CPUS=[1,13,25,37,49,61,73,85]
CANDIDATE_SHA='3e7b1999c43d9e2ddd7a286382a0af607f273fcbc3f53b407c1ad51b2c56197a'
WINDOW_OUTCOME_SHA='a586026bc301e6a6e33a98d99abfbcd576b705737c2a74e81ae6409f6fd608d5'

def revision():
    return dict(id='same-numa-cpu-offset-one-v3',previous_cpu_ids=CPUS,cpu_ids=ALTERNATE_CPUS,
        candidate_observation_sha256=CANDIDATE_SHA,prior_window_outcome_sha256=WINDOW_OUTCOME_SHA,
        model_and_rank_ownership_unchanged=True,worker_cpu_count_unchanged=True,
        production_services_modified=False,exclusive_cpu_reservation=False,
        maximum_launch_attempts=1,maximum_neural_runs=1,automatic_retry=False)

def admitted_cpus(admission):
    if 'cpu_placement_revision' not in admission:return CPUS
    check(admission['cpu_placement_revision']==revision(),'unapproved CPU placement revision')
    return ALTERNATE_CPUS

def candidate_topologies(evidence):
    path=evidence/'rust-cpu-placement-diagnosis-v1/candidate.json'
    check(sha(path)==CANDIDATE_SHA,'CPU candidate evidence changed')
    report=read(path)
    check(report['candidate_cpu_ids']==ALTERNATE_CPUS and report['previous_cpu_ids']==CPUS
          and report['neural_runs_started']==0 and len(report['hosts'])==4,'CPU candidate identity')
    check(sha(path.with_name('report.json'))==report['diagnosis_sha256'],'CPU diagnosis changed')
    result=[]
    for node,row in zip(NODES,report['hosts'],strict=True):
        check(row['host']==node and not row['own_units'].strip(),'CPU candidate host state')
        topology={int(x[0]):x for x in row['topology']}
        check(len({topology[c][1] for c in ALTERNATE_CPUS})==8,'distinct physical cores required')
        check(all(topology[c][2:]==topology[p][2:] and topology[c][4]=='Y'
                  for c,p in zip(ALTERNATE_CPUS,CPUS,strict=True)),'socket/NUMA placement changed')
        check(len(row['samples'])==3,'three CPU observations required')
        for sample in row['samples']:
            check(set(sample)==set(map(str,ALTERNATE_CPUS)) and max(sample.values())<50
                  and sum(sample.values())/8<25,'candidate CPU occupancy failed')
        result.append([topology[c] for c in ALTERNATE_CPUS])
    return result

def previous_window_gate(out):
    check(not (out/'alternate-cpus-v3').exists(),'CPU relocation already attempted')
    path=out/'resource-window-v2/outcome.json'
    check(sha(path)==WINDOW_OUTCOME_SHA,'previous resource window changed')
    row=read(path)
    check(row['launcher_terminal'] is True and row['admission_created'] is False
          and row['launch_started'] is False and row['neural_runs_started']==0,'previous model may have launched')
    for name,digest in row['input_sha256'].items():
        check(sha(path.parent/name)==digest,'previous resource check changed')
