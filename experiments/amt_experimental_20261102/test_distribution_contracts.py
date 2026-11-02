"""Check fail-closed runtime admission and non-destructive output import."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runtime_attestation as runtime
from remote_task import merge


class RuntimeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = runtime.read(runtime.ROOT/'distributed/expected-runtime.json')

    def test_hardware_and_install_paths_are_recorded_but_not_version_changes(self):
        other=copy.deepcopy(self.expected)
        other['gpu_inventory']=['a different approved GPU and driver']
        other['cuda_visible_devices']='1'
        other['actual_tracking_imports']['holo_linker']['path']='/private/worker/holo_linker.py'
        other['modules']['torch']['path']='/private/worker/torch/__init__.py'
        self.assertEqual(runtime.canonical(self.expected),runtime.canonical(other))

    def test_stale_holo_linker_is_rejected(self):
        other=copy.deepcopy(self.expected)
        other['actual_tracking_imports']['holo_linker']['sha256']='stale'
        with patch.object(runtime,'runtime_identity',return_value=other):
            with self.assertRaisesRegex(AssertionError,'differs'):
                runtime.check()

    def test_dependency_or_execution_adapter_changes_are_rejected(self):
        for field in ['modules','execution_adapter_files','numerical_helper_files']:
            other=copy.deepcopy(self.expected)
            if field=='modules':other[field]['torch']['version']='old'
            else:other[field][next(iter(other[field]))]='changed'
            with self.subTest(field=field),patch.object(runtime,'runtime_identity',return_value=other):
                with self.assertRaisesRegex(AssertionError,'differs'):
                    runtime.check()

    def test_missing_pinned_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(runtime,'runtime_identity',return_value=self.expected):
            with self.assertRaisesRegex(AssertionError,'missing'):
                runtime.check(Path(temp)/'absent.json')


class ImportContracts(unittest.TestCase):
    def test_different_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'new';target=Path(temp)/'existing'
            source.write_text('new output');target.write_text('existing research output')
            with self.assertRaises(AssertionError):merge(source,target)
            self.assertEqual(target.read_text(),'existing research output')
            self.assertTrue(source.exists())

    def test_matching_files_and_new_files_can_be_merged(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'new';target=Path(temp)/'existing'
            source.mkdir();target.mkdir()
            (source/'same').write_text('same');(target/'same').write_text('same')
            (source/'added').write_text('new')
            merge(source,target)
            self.assertEqual((target/'same').read_text(),'same')
            self.assertEqual((target/'added').read_text(),'new')


if __name__=='__main__':unittest.main()
