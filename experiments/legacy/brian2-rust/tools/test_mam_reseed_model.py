import copy
import unittest
import numpy as np
from brian2_rust.protocol import attach_protocol
from mam_replicate_seeds import key,allocation,registry
from mam_reseed_model import reseed,audit_delta,voltage_bits


def fixture():
    p=dict(params=dict(neuron_params=dict(V0_mean=-150.,V0_sd=50.)),populations=[
        dict(name='V1-23E',count=3),dict(name='V2-23E',count=5)],
        projections=[dict(source=0,target=1,count=7),dict(source=1,target=0,count=11)])
    rng=np.random.default_rng(1729);states={}
    for pop in p['populations']:
        states['mam_'+pop['name'].replace('-','_')]=dict(initial_state=dict(v=voltage_bits(rng,p,pop['count']),current=['0000000000000000']*pop['count']),parameters=dict(weight=['3ff0000000000000']),refractory=None)
    # Exported canonical order can differ from the official draw order.
    names=['mam_V2_23E','mam_V1_23E']
    m=dict(schema='b2ir-v1',definition=dict(populations=[dict(name=n,count=len(states[n]['initial_state']['v'])) for n in names],synapses=[
        dict(name='mam_projection_1',source_population=0,target_population=1),
        dict(name='mam_projection_0',source_population=1,target_population=0)]),
        instance=dict(rng_seed=1729,populations=[states[n] for n in names],synapses=[
            dict(topology=dict(kind='fixed_total',edge_count=11,seed=1730,initializers=dict(w='same')),pathways=['same']),
            dict(topology=dict(kind='fixed_total',edge_count=7,seed=1729,initializers=dict(w='same')),pathways=['same'])]),
        run=dict(duration='4059200000000000',steps=1005000))
    return attach_protocol(m),p


class SeedTests(unittest.TestCase):
    def test_registry_no_batch_or_legacy_key_reuse(self):
        r=registry();self.assertEqual(r['keys'],5*(8344+2));self.assertEqual(r['legacy_key_overlap'],0)
        self.assertEqual(r,registry())

    def test_domains_and_replicates_differ(self):
        self.assertEqual(len({key(1750,'initial_voltage'),key(1750,'runtime_input'),key(1750,'projection',0),key(1751,'projection',0)}),4)
        with self.assertRaises(ValueError):key(1729,'runtime_input')
        with self.assertRaises(ValueError):key(1750,'projection',True)

    def test_reseed_preserves_definition_and_run(self):
        old,p=fixture();saved=copy.deepcopy(old);new,report=reseed(old,p,1750)
        self.assertEqual(old,saved);self.assertEqual(new['definition'],old['definition']);self.assertEqual(new['run'],old['run'])
        self.assertNotEqual(new['protocol']['layers']['instance'],old['protocol']['layers']['instance'])
        self.assertTrue(report['legacy_voltage_reproduced_exactly'])
        self.assertEqual(new,reseed(old,p,1750)[0]);self.assertNotEqual(new['instance'],reseed(old,p,1751)[0]['instance'])

    def test_wrong_legacy_voltage_is_rejected(self):
        old,p=fixture();old['instance']['populations'][0]['initial_state']['v'][0]='0000000000000000';attach_protocol(old)
        with self.assertRaisesRegex(ValueError,'legacy voltage'):reseed(old,p,1750)

    def test_wrong_legacy_recipe_is_rejected(self):
        old,p=fixture();old['instance']['synapses'][0]['topology']['seed']=900;attach_protocol(old)
        with self.assertRaisesRegex(ValueError,'legacy projection'):reseed(old,p,1750)

    def test_undeclared_changes_rejected(self):
        old,p=fixture();base,report=reseed(old,p,1750)
        for mutate in [lambda m:m['run'].update(steps=100),
                       lambda m:m['instance']['populations'][0]['initial_state']['current'].__setitem__(0,'changed'),
                       lambda m:m['instance']['synapses'][0]['pathways'].append('changed'),
                       lambda m:m['instance']['populations'].append(copy.deepcopy(m['instance']['populations'][0])),
                       lambda m:m['instance']['synapses'][0]['topology'].update(seed=42),
                       lambda m:m['instance']['populations'][0]['initial_state']['v'].__setitem__(0,'changed')]:
            new=copy.deepcopy(base);mutate(new)
            with self.assertRaises(ValueError):audit_delta(old,new,p,report['keys'])

    def test_wrong_projection_mapping_rejected(self):
        old,p=fixture();old['definition']['synapses'][0]['source_population']=1;attach_protocol(old)
        with self.assertRaisesRegex(ValueError,'endpoint'):reseed(old,p,1750)


if __name__=='__main__':unittest.main()
