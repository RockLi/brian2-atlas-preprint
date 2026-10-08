"""Cross-replicate identity and deployment isolation using completed artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import mam_confirmation_identity as identity
import mam_deploy_confirmation_artifact as deploy

E=Path(__file__).resolve().parents[1]/'mpi-evidence'

class NextReplicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=identity.identity(E)
        cls.new=identity.identity(E,1751)

    def tearDown(self):deploy.configure(self.old)

    def test_full_geometry_definition_and_run_unchanged_inputs_distinct(self):
        for key in ('definition_sha256','run_sha256'):
            self.assertEqual(self.old[key],self.new[key])
        # The plan embeds the new instance hash; its own digest must change.
        for key in ('model_sha256','instance_sha256','plan_sha256','executable_sha256'):
            self.assertNotEqual(self.old[key],self.new[key])
        for seed in (1750,1751):
            prepared=json.loads((E/f'confirmation-artifact-v1-seed{seed}'/'prepared.json').read_text())
            # Both full workers compared every non-instance plan field to the
            # identical pinned legacy plan before emitting these bound receipts.
            self.assertTrue(prepared['plan_only_instance_identity_changed'])
            self.assertTrue(prepared['definition_and_run_exact'])
        a=self.old['random_keys'];b=self.new['random_keys']
        self.assertTrue(set([a['initial_voltage'],a['runtime_input'],*a['projections']]).isdisjoint([b['initial_voltage'],b['runtime_input'],*b['projections']]))
        seen=[]
        for i in range(4):
            c=identity.deployment_catalog(self.new,i)
            ranks=sorted(int(x['path'].removeprefix('mpi/instance.rank-').removesuffix('.bin')) for x in c['files'] if x['path'].startswith('mpi/instance.rank-'))
            self.assertEqual(ranks,list(range(i*8,(i+1)*8)));seen.extend(ranks)
            self.assertEqual(len(c['files']),16)
        self.assertEqual(seen,list(range(32)))
        for i in range(32):
            n=f'mpi/instance.rank-{i}.bin'
            self.assertNotEqual(self.old['files'][n]['sha256'],self.new['files'][n]['sha256'])

    def test_renamed_old_completion_cannot_authorize_new_replica(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'confirmation-artifact-v1-seed1751';p.mkdir()
            (p/'complete.json').write_bytes((E/identity.CASE/'complete.json').read_bytes())
            with self.assertRaisesRegex(ValueError,'digest'):identity.identity(Path(tmp),1751)
        with self.assertRaisesRegex(ValueError,'not pinned'):identity.identity(E,1752)

    def test_deployment_preflight_uses_only_new_artifact_and_targets(self):
        deploy.configure(self.new)
        c=identity.deployment_catalog(self.new,1)
        with patch.object(deploy,'remote',lambda node,code:code):
            src=deploy.preflight(0,c,True);dst=deploy.preflight(1,c,False)
        self.assertIn(self.new['source_artifact'],src)
        self.assertIn(self.new['project'],dst)
        self.assertNotIn('seed1750',src+dst+' '.join(deploy.CONTROLS))
        self.assertEqual(len(set(deploy.OUTPUTS)),2) # resolved brick path on23, same home path elsewhere
        with self.assertRaisesRegex(ValueError,'restricted to 1750'):
            deploy.run(E,Path('/unused-output'),E/'historical-prior',1751)

if __name__=='__main__':unittest.main()
