import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "giukhaji"
sys.path.insert(0, str(APP_DIR))

from modeling.gait_axis_aligned_processor import predict_daily_gait_csv
from pengteu.rag_engine import LocalRagEngine


def synthetic_gait_csv() -> str:
    rows = [
        "Timestamp_ns,Acc_X,Acc_Y,Acc_Z,"
        "Gyro_Clean_X,Gyro_Clean_Y,Gyro_Clean_Z"
    ]
    for index in range(2001):
        seconds = index / 100.0
        ml = 0.22 * math.sin(2 * math.pi * 1.8 * seconds)
        vertical = 9.80665 + 0.35 * math.sin(2 * math.pi * 3.6 * seconds)
        ap = 0.18 * math.sin(2 * math.pi * 1.8 * seconds + 0.5)
        rows.append(
            f"{index * 10_000_000},{ml},{vertical},{ap},0,0,0"
        )
    return "\n".join(rows)


class SystemFlowTests(unittest.TestCase):
    def test_gait_model_accepts_android_csv_shape(self):
        result = predict_daily_gait_csv(
            io.StringIO(synthetic_gait_csv()), APP_DIR / "models"
        )
        self.assertIn(result["prediction"], (0, 1))
        self.assertTrue(0 <= result["probability"] <= 1)
        self.assertEqual(result["window"]["duration_sec"], 20.0)
        for name in (
            "v_jerk_rms_median",
            "v_jerk_rms_iqr",
            "v_harmonic_ratio_iqr",
        ):
            self.assertTrue(math.isfinite(result["features"][name]))

    def test_rag_returns_paper_citation_and_excludes_references(self):
        engine = LocalRagEngine(APP_DIR / "pengteu" / "knowledge")
        results = engine.search("고령자 스마트폰 가속도계 보행 정확도", top_k=8)
        evidence = [item for item in results if item.get("doi")]
        self.assertTrue(evidence)
        self.assertTrue(any(item.get("pages") for item in evidence))
        self.assertTrue(any("Keppler" in item.get("citation", "") for item in evidence))
        self.assertFalse(any(item["title"].lower() == "references" for item in results))

    def test_flask_upload_rag_and_safety_routes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "synthetic_gait.csv"
            csv_path.write_text(synthetic_gait_csv(), encoding="utf-8")
            env = os.environ.copy()
            env.update(
                SECRET_KEY="flow-secret",
                PHONE_HASH_SALT="flow-phone-salt",
                MOCA_DB_PATH=str(Path(temp_dir) / "flow.sqlite3"),
                FLOW_CSV_PATH=str(csv_path),
            )
            code = """
import io
import os
import app
from core import database

client = app.app.test_client()
csv_text = open(os.environ['FLOW_CSV_PATH'], encoding='utf-8').read()
response = client.post('/gait/upload-csv', data={
    'member_phone': '01098765432',
    'education_level': 'high',
    'file': (io.BytesIO(csv_text.encode()), 'apk_gait.csv'),
}, content_type='multipart/form-data')
assert response.status_code == 200, response.get_data(as_text=True)
payload = response.get_json()
assert payload['ok'] and len(payload['features']) == 3

rag = client.get('/assistant/rag/search?q=고령자 스마트폰 보행 정확도')
assert rag.status_code == 200
results = rag.get_json()['results']
assert any(item.get('doi') and item.get('pages') for item in results)

member = database.find_member_by_phone('01098765432')
with client.session_transaction() as session:
    session['member_id'] = member['id']
chat = client.post('/assistant/chat', json={'message': '보행 센서 결과는 진단인가요?'})
assert chat.status_code == 200
reply = chat.get_json()['reply']
assert '확진' not in reply and '진단입니다' not in reply

bad = client.post('/gait/upload-csv', data={
    'file': (io.BytesIO(b'bad'), 'bad.csv'),
}, content_type='multipart/form-data')
assert bad.status_code == 400
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


if __name__ == "__main__":
    unittest.main()
