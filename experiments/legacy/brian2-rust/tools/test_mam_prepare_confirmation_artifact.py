"""Ensure a selected replicate reaches the remote worker without old-path reuse."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import mam_prepare_confirmation_artifact as prepare

class DispatchCaptured(Exception):
    pass

class PreparationSelectionTest(unittest.TestCase):
    def tearDown(self):
        prepare.configure(1750)

    def test_invalid_selection_preserves_existing_selection(self):
        prepare.configure(1751)
        for invalid in (True, '1751', 1752, 0):
            with self.assertRaises(ValueError):
                prepare.configure(invalid)
            self.assertEqual(prepare.REPLICATE,1751)
            self.assertTrue(str(prepare.STAGE).endswith('seed1751'))

    def dispatch(self,replicate):
        prepare.configure(replicate)
        calls=[]
        def remote(code,data=None):
            calls.append(code)
            if data is None:
                return {'snapshot':'synthetic-dispatch-test'}
            bundle=json.loads(data)
            return {n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()}
        def launch(command,**kwargs):
            self.assertIn('root@'+prepare.NODE,command)
            shell=command[-1]
            self.assertIn('--unit=b2mpi-mam-confirmation-artifact-v1-seed'+str(replicate)+'-build',shell)
            self.assertIn('/mam-confirmation-artifact-v1-seed'+str(replicate)+'/tools/mam_prepare_confirmation_artifact.py',shell)
            if replicate==1751:
                self.assertTrue(shell.endswith('worker --replicate 1751'),shell)
                self.assertNotIn('seed1750',shell)
            else:
                self.assertTrue(shell.endswith('worker'),shell)
                self.assertNotIn('--replicate',shell)
            raise DispatchCaptured()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'once'
            with patch.object(prepare,'remote',remote),patch.object(prepare.subprocess,'run',launch):
                with self.assertRaises(DispatchCaptured):prepare.run(output)
                intent=json.loads((output/'intent.json').read_text())
                self.assertEqual(intent['replicate'],replicate)
                self.assertEqual(intent['maximum_full_preparations'],1)
                self.assertEqual(intent['neural_simulations'],0)
                self.assertEqual(intent['worker_seconds'],1500)
                self.assertIn('seed'+str(replicate),calls[0])
                with self.assertRaisesRegex(ValueError,'attempt already exists'):
                    prepare.run(output)
                self.assertEqual(len(calls),2)

    def test_1751_remote_command_and_one_attempt_gate(self):self.dispatch(1751)
    def test_1750_default_remote_command_retained(self):self.dispatch(1750)

if __name__=='__main__':unittest.main()
