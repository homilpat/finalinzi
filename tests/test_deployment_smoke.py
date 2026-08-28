import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "giukhaji"


class DeploymentSmokeTests(unittest.TestCase):
    def test_app_health_access_gate_and_private_phone_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "smoke.sqlite3"
            env = os.environ.copy()
            env.update(
                SECRET_KEY="test-secret",
                PHONE_HASH_SALT="test-phone-salt",
                MOCA_DB_PATH=str(db_path),
            )
            code = """
import os
import app
from core import database

client = app.app.test_client()
assert client.get('/health').get_json()['ok'] is True
assert client.get('/').status_code == 200
os.environ['ACCESS_PASSWORD'] = 'demo-password'
assert client.get('/health').status_code == 200
assert client.get('/').status_code == 302
assert client.post('/login', data={
    'password': 'demo-password', 'next': '/'
}).status_code == 302
assert client.get('/').status_code == 200

phone = '01012345678'
member_id, _, member_code, created = database.get_or_create_member(phone, 'high')
assert created and member_code.startswith('ID_A')
assert database.find_member_by_phone(phone)['id'] == member_id
assert phone.encode() not in open(database.DB_PATH, 'rb').read()
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=APP_DIR,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cloud_and_native_microphone_configuration(self):
        render = (ROOT / "render.yaml").read_text(encoding="utf-8")
        gradle = (ROOT / "FinalProjectApp/app/build.gradle.kts").read_text(
            encoding="utf-8"
        )
        kotlin = (ROOT / "FinalProjectApp/app/src/main/java/com/example/finalprojectapp/MainActivity.kt").read_text(
            encoding="utf-8"
        )
        javascript = (ROOT / "giukhaji/static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("healthCheckPath: /health", render)
        self.assertIn("PHONE_HASH_SALT", render)
        self.assertIn("https://finalinzi.onrender.com/", gradle)
        self.assertIn("fun startTestStt()", kotlin)
        self.assertIn("fun stopPengteuStt()", kotlin)
        self.assertIn("stopTestStt()\n        stopPengteuTts()", kotlin)
        self.assertIn("override fun onStop(utteranceId", kotlin)
        self.assertIn("window.TestSpeechNative", javascript)
        self.assertIn("window.AndroidBridge.startTestStt()", javascript)
        self.assertIn("window.AndroidBridge.stopTestStt()", javascript)
        self.assertIn("window.AndroidBridge.stopPengteuStt()", javascript)


if __name__ == "__main__":
    unittest.main()
