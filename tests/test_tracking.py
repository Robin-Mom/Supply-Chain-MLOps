"""Local-file authentication and DVC environment inheritance, without a server."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import experiment
import tracking


class TrackingTests(unittest.TestCase):
    def test_loads_dagshub_credentials_without_executing_or_copying_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env.dagshub.local'
            path.write_text('MLFLOW_TRACKING_URI=https://dagshub.com/example/repo.mlflow\n'
                            'MLFLOW_TRACKING_USERNAME=example\n'
                            'DAGSHUB_USER_TOKEN="fake-$(do-not-execute)-token"\n'
                            'UNRELATED_SETTING=must-not-load\n')
            original = path.read_bytes()
            fake_mlflow = SimpleNamespace(set_tracking_uri=Mock())
            with patch.dict(sys.modules, {'mlflow': fake_mlflow}):
                uri = experiment.tracking({'tracking': {'env_file': str(path), 'uri': ''}})
            self.assertEqual(uri, 'https://dagshub.com/example/repo.mlflow')
            self.assertEqual(os.environ['MLFLOW_TRACKING_PASSWORD'], 'fake-$(do-not-execute)-token')
            self.assertEqual(os.environ['MLFLOW_TRACKING_USERNAME'], 'example')
            self.assertNotIn('UNRELATED_SETTING', os.environ)
            self.assertEqual(path.read_bytes(), original)
            fake_mlflow.set_tracking_uri.assert_called_once_with(uri)
            # The ignored file need not exist in the isolated workspace. Each
            # stage inherits the URI, the authentication and the original path.
            child = Path(directory) / 'isolated'
            child.mkdir()
            result = subprocess.run([sys.executable, '-c',
                'import os; from tracking import configure_tracking; '
                'from unittest.mock import patch; from types import SimpleNamespace; '
                'from unittest.mock import Mock; '
                'expected = os.environ["DAGSHUB_USER_TOKEN"]; '
                'assert not __import__("pathlib").Path(".env.dagshub.local").exists(); '
                'fake = SimpleNamespace(set_tracking_uri=Mock()); '
                '\nwith patch.dict(__import__("sys").modules, {"mlflow": fake}):\n'
                ' assert configure_tracking() == os.environ["MLFLOW_TRACKING_URI"]\n'
                ' assert os.environ["MLFLOW_TRACKING_PASSWORD"] == expected\n'],
                cwd=child, env={**os.environ, 'PYTHONPATH': str(Path(tracking.__file__).parent)},
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, 'Isolated tracking configuration failed.')

    def test_exports_and_explicit_uri_take_priority(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env.dagshub.local'
            path.write_text('MLFLOW_TRACKING_URI=https://dagshub.com/file/repo.mlflow\n'
                            'MLFLOW_TRACKING_USERNAME=file-user\nDAGSHUB_USER_TOKEN=file-token\n')
            os.environ.update(MLFLOW_TRACKING_URI='https://dagshub.com/export/repo.mlflow',
                              MLFLOW_TRACKING_USERNAME='export-user', MLFLOW_TRACKING_PASSWORD='export-token')
            with patch.dict(sys.modules, {'mlflow': SimpleNamespace(set_tracking_uri=Mock())}):
                self.assertEqual(tracking.configure_tracking(env_file=path), os.environ['MLFLOW_TRACKING_URI'])
                self.assertEqual(tracking.configure_tracking('sqlite:////tmp/offline.db', env_file=path),
                                 'sqlite:////tmp/offline.db')
            self.assertEqual(os.environ['MLFLOW_TRACKING_USERNAME'], 'export-user')
            self.assertEqual(os.environ['MLFLOW_TRACKING_PASSWORD'], 'export-token')

    def test_rejects_missing_credentials_and_uri_embedded_credentials(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            missing = Path(directory) / 'missing.env'
            with self.assertRaisesRegex(ValueError, 'DagsHub requires'):
                tracking.configure_tracking('https://dagshub.com/example/repo.mlflow', env_file=missing)
            with self.assertRaises(ValueError) as context:
                tracking.configure_tracking('https://user:fake-secret@dagshub.com/example/repo.mlflow', env_file=missing)
            self.assertNotIn('fake-secret', str(context.exception))
            with self.assertRaisesRegex(ValueError, 'HTTPS'):
                tracking.configure_tracking('http://dagshub.com/example/repo.mlflow', env_file=missing)

    def test_connection_check_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / 'local.env'
            path.write_text('MLFLOW_TRACKING_URI=https://dagshub.com/example/repo.mlflow\n'
                            'MLFLOW_TRACKING_USERNAME=example\nDAGSHUB_USER_TOKEN=fake-token\n')
            client = Mock()
            fake_mlflow = SimpleNamespace(set_tracking_uri=Mock(), MlflowClient=Mock(return_value=client))
            with patch.dict(sys.modules, {'mlflow': fake_mlflow}):
                tracking.main(['--check', '--env-file', str(path)])
            client.search_experiments.assert_called_once_with(max_results=1)
            self.assertEqual(len(client.mock_calls), 1)


if __name__ == '__main__':
    unittest.main()
