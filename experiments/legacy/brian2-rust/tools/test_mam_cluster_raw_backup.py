import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from mam_cluster_raw_backup import readback, command, SOURCE, DEST, SCHEMA


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.data=b'bounded backup test'*4096
        (self.root/'events.bin').write_bytes(self.data)
        self.catalog=dict(schema=SCHEMA,files=[dict(path='events.bin',bytes=len(self.data),
            sha256=hashlib.sha256(self.data).hexdigest())])

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):return readback(self.root,self.catalog)

    def test_full_readback(self):
        result=self.check()
        self.assertTrue(result['full_destination_readback'])
        self.assertEqual(result['files'],self.catalog['files'])

    def test_same_size_corruption_rejected(self):
        (self.root/'events.bin').write_bytes(b'x'+self.data[1:])
        with self.assertRaisesRegex(ValueError,'digest'):self.check()

    def test_truncated_rejected(self):
        (self.root/'events.bin').write_bytes(self.data[:-1])
        with self.assertRaisesRegex(ValueError,'size'):self.check()

    def test_symlink_rejected(self):
        (self.root/'events.bin').rename(self.root/'other.bin')
        (self.root/'events.bin').symlink_to(self.root/'other.bin')
        with self.assertRaisesRegex(ValueError,'symlink'):self.check()

    def test_path_escape_rejected(self):
        self.catalog['files'][0]['path']='../events.bin'
        with self.assertRaises(ValueError):self.check()

    def test_resource_and_volume_bounds(self):
        for node,vol,reserve,allow in [(SOURCE,'/data/brick2','1280',False),(DEST,'/','128',True)]:
            cmd=command(node,'test',1800,['python3','test.py'])
            self.assertEqual(cmd[cmd.index('--volume')+1],vol)
            self.assertEqual(cmd[cmd.index('--min-free-gib')+1],reserve)
            self.assertEqual('--allow-root-volume' in cmd,allow)
            self.assertIn('--property=MemorySwapMax=0',cmd)
            self.assertIn('--property=MemoryMax=4096M',cmd)
            self.assertIn('--property=RuntimeMaxSec=1805',cmd)
            self.assertEqual(cmd[-3:],['--','python3','test.py'])


if __name__=='__main__':unittest.main()
