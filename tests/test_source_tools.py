import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_public_tree import audit, source_paths
from export_public_source import export_source


class SourceToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_reports_reference_and_credential_without_echoing_value(self):
        private_url = 'https://service.' + 'capital' + 'one.com'
        token = 'ghp_' + 'a' * 36
        (self.root / 'config.py').write_text(f'URL = "{private_url}"\nTOKEN = "{token}"\n')
        findings = audit(self.root)
        self.assertEqual({f['rule'] for f in findings}, {'organization-reference', 'credential-format'})
        self.assertNotIn(token, json.dumps(findings))
        self.assertNotIn(private_url, json.dumps(findings))

    def test_reports_notebook_outputs_and_metadata(self):
        notebook = {'nbformat': 4, 'metadata': {'widgets': {}}, 'cells': [
            {'cell_type': 'code', 'source': [], 'execution_count': 1, 'outputs': [{'text': 'session'}]}]}
        (self.root / 'session.ipynb').write_text(json.dumps(notebook))
        self.assertEqual({f['rule'] for f in audit(self.root)},
                         {'saved-notebook-output', 'notebook-session-metadata'})

    def test_allows_selected_public_dataset_without_allowing_other_references(self):
        organization = 'capital' + 'one'
        public = f'https://huggingface.co/datasets/{organization}/PRES'
        (self.root / 'README.md').write_text(public + '\n')
        (self.root / 'download.py').write_text(f'repo_id = "{organization}/PRES"\n')
        self.assertEqual(audit(self.root), [])
        (self.root / 'README.md').write_text(public + '\nhttps://private.' + organization + '.com\n')
        self.assertEqual([(f['line'], f['rule']) for f in audit(self.root)], [(2, 'organization-reference')])

    def test_export_uses_edited_source_and_excludes_git_and_ignored_files(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / '.gitignore').write_text('private/\n*.local.yaml\n')
        (self.root / 'main.py').write_text('old = True\n')
        subprocess.run(['git', '-C', str(self.root), 'add', '.gitignore', 'main.py'], check=True)
        (self.root / 'main.py').write_text('current = True\n')
        (self.root / 'private').mkdir()
        (self.root / 'private/data.txt').write_text('private session')
        (self.root / 'job.local.yaml').write_text('site configuration')
        (self.root / 'new.py').write_text('new = True\n')
        output = self.root.parent / (self.root.name + '.zip')
        self.addCleanup(lambda: output.unlink(missing_ok=True))
        self.assertEqual(export_source(self.root, output), 3)
        with ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()),
                             {'event-seq-graph/.gitignore', 'event-seq-graph/main.py', 'event-seq-graph/new.py'})
            self.assertEqual(archive.read('event-seq-graph/main.py'), b'current = True\n')

    def test_rejects_dirty_source_even_if_git_index_is_clean(self):
        (self.root / 'bad.py').write_text('value = "' + 'capital' + 'one"\n')
        output = self.root / 'source.zip'
        with self.assertRaises(ValueError):
            export_source(self.root, output)
        self.assertFalse(output.exists())

    def test_rejects_symlink_instead_of_exporting_target(self):
        (self.root / 'main.py').write_text('valid = True\n')
        (self.root / 'link.py').symlink_to(self.root / 'main.py')
        self.assertIn('source-symlink', {f['rule'] for f in audit(self.root)})


if __name__ == '__main__':
    unittest.main()
