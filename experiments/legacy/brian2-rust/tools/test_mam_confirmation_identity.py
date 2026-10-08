import copy
import json
from pathlib import Path
import tempfile
import unittest
from mam_confirmation_identity import identity,deployment_catalog,CASE

EVIDENCE=Path(__file__).resolve().parents[1]/'mpi-evidence'


class IdentityTests(unittest.TestCase):
    def test_real_full_build_identity_and_owned_shards(self):
        value=identity(EVIDENCE);seen=[]
        self.assertEqual(value['replicate'],1750)
        self.assertNotEqual(value['random_keys']['runtime_input'],1750)
        self.assertFalse(value['neural_launch_admitted'])
        for host in range(4):
            catalog=deployment_catalog(value,host)
            self.assertEqual(len(catalog['files']),16)
            ranks=sorted(int(x['path'].removeprefix('mpi/instance.rank-').removesuffix('.bin')) for x in catalog['files'] if x['path'].startswith('mpi/instance.rank-'))
            self.assertEqual(ranks,list(range(host*8,(host+1)*8)));seen.extend(ranks)
            self.assertNotIn('model.json',[x['path'] for x in catalog['files']])
        self.assertEqual(seen,list(range(32)))

    def test_unknown_host_index_rejected(self):
        for index in [-1,4,True,'0']:
            with self.assertRaises(ValueError):deployment_catalog({},index)

    def test_changed_completion_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/CASE;p.mkdir()
            c=json.loads((EVIDENCE/CASE/'complete.json').read_text());c['replicate']=1729
            (p/'complete.json').write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'digest'):identity(Path(temp))

    def test_changed_audit_cannot_pass_same_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/CASE;p.mkdir();base=EVIDENCE/CASE
            c=json.loads((base/'complete.json').read_text())
            for name in ['complete.json',*c['evidence_sha256']]:
                (p/name).write_bytes((base/name).read_bytes())
            (p/'random-input-audit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence changed'):identity(Path(temp))


if __name__=='__main__':unittest.main()
