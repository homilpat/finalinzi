"""
MoCA-K 시연용 Flask 웹앱
폰 브라우저에서 접속: http://<서버IP>:5000
"""

import os, io, base64, json, hmac, secrets, urllib.request
from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4


def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()

from flask import Flask, render_template, request, jsonify, session, send_from_directory, redirect, url_for
import numpy as np
import cv2
from exercise_sensor_processor import analyze_exercise_csv
from gait_axis_aligned_processor import predict_daily_gait_csv
from rag_engine import retrieve_knowledge
from database import (
    EDUCATION_LEVELS,
    complete_assessment,
    create_assessment,
    education_label,
    find_member_by_code_or_name,
    find_member_by_phone,
    get_exercise_summary,
    get_guardian_dashboard,
    get_guardian_cheers,
    get_latest_assessment,
    get_latest_physical_result,
    get_member,
    get_member_context_bundle,
    get_or_create_assistant_profile,
    get_or_create_guardian,
    get_recent_assessment_summaries,
    get_or_create_member,
    init_db,
    link_guardian_member,
    normalize_phone,
    phone_hash,
    phone_last4,
    save_assistant_message,
    save_exercise_record,
    save_guardian_cheer,
    save_physical_result,
    save_sensor_calibration,
    update_assistant_profile,
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'moca-demo-2026-dev')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
)
init_db()


def _access_password():
    return os.environ.get('ACCESS_PASSWORD', '').strip()


@app.before_request
def require_access_password():
    password = _access_password()
    if not password:
        return None
    if request.endpoint in ('login', 'api_mobile_remember_login'):
        return None
    if session.get('access_granted') is True:
        return None
    if request.path == '/favicon.ico':
        return '', 204
    return redirect(url_for('login', next=request.path))


@app.route('/pengt.png')
def pengteu_image():
    return send_from_directory(app.root_path, 'pengt.png')

# 메모리 세션 저장소 (시연용)
_store = {}  # uid → { 'sess': MoCASession, 'raw': dict, 'location': str, 'sigungu': str }
_gait_result_store = {}

# ── CNN 모델 (학습 완료 시 자동 로드, 없으면 룰베이스 폴백) ──
_cnn_cube        = None   # (model, device) from cube_cnn_inference_v2.load_model
_cnn_clock       = None   # dict: {'deepc': model, 'deeph': model, 'deepn': model}
_cnn_clock_dev   = None   # torch.device
_gait_models     = None

GAIT_FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]
GAIT_REQUIRED_FEATURES = GAIT_FEATURES


def _load_gait_artifact(model_name, metadata_name):
    model_path = os.path.join(app.root_path, "models", model_name)
    metadata_path = os.path.join(app.root_path, "models", metadata_name)
    metadata = None
    model = None

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if os.path.exists(model_path):
        try:
            import joblib
            model = joblib.load(model_path)
            if model is not None:
                from sklearn.pipeline import Pipeline
                from sklearn.impute import SimpleImputer
                if isinstance(model, dict) and "pipeline" in model:
                    new_pipe = Pipeline(model["pipeline"].steps)
                    for _, step_obj in new_pipe.steps:
                        if isinstance(step_obj, SimpleImputer):
                            if not hasattr(step_obj, "_fill_dtype") and hasattr(step_obj, "_fit_dtype"):
                                step_obj._fill_dtype = step_obj._fit_dtype
                    model["pipeline"] = new_pipe
        except Exception as e:
            print(f"[gait model load error] {e}")
            model = None

    return model, metadata


def _load_gait_models():
    """Load the final daily acc-only gait model lazily."""
    global _gait_models
    if _gait_models is not None:
        return _gait_models

    daily_model, daily_metadata = _load_gait_artifact(
        "gait_daily_clinical_3feat.joblib",
        "gait_daily_clinical_3feat_metadata.json",
    )
    _gait_models = {
        "daily": {"model": daily_model, "metadata": daily_metadata},
    }
    return _gait_models


def _gait_model_summary():
    daily_meta_path = os.path.join(app.root_path, "models", "gait_daily_clinical_3feat_metadata.json")
    daily_meta = {}
    if os.path.exists(daily_meta_path):
        try:
            with open(daily_meta_path, "r", encoding="utf-8") as f:
                daily_meta = json.load(f)
        except Exception:
            pass
    daily_available = os.path.exists(os.path.join(app.root_path, "models", "gait_daily_clinical_3feat.joblib"))

    oof = daily_meta.get("oof", {})
    train = daily_meta.get("train", {})
    return {
        "available": daily_available,
        "daily_model_available": daily_available,
        "threshold": daily_meta.get("threshold", 0.470),
        "threshold_strategy": daily_meta.get("threshold_strategy", "fixed_screening_threshold_sensitivity_prioritized"),
        "n_subjects": train.get("n_subjects"),
        "features": daily_meta.get("features", GAIT_FEATURES),
        "metrics": {
            "subject_auc": oof.get("auc"),
            "sensitivity": oof.get("sens"),
            "specificity": oof.get("spec"),
        },
        "label_source": daily_meta.get("label_source"),
        "realtime_model_available": daily_available,
    }


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _has_final_gait_features(features):
    for feature in GAIT_REQUIRED_FEATURES:
        try:
            value = float(features.get(feature))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(value):
            return False
    return True


def _gait_feature_insights(features):
    checks = [
        {
            "key": "v_jerk_rms_median",
            "label": "수직 충격/추진 대표값",
            "value": _safe_float(features.get("v_jerk_rms_median")),
            "unit": " g/s",
            "problem": "걸음의 수직 충격과 추진 수준이 낮아 보폭이나 보행 힘을 확인할 필요가 있습니다.",
            "ok": "걸음의 수직 충격과 추진 수준이 비교적 양호합니다.",
            "risk_when": "low",
            "cut": 1.20,
        },
        {
            "key": "v_jerk_rms_iqr",
            "label": "수직 충격 변동성",
            "value": _safe_float(features.get("v_jerk_rms_iqr")),
            "unit": " g/s",
            "problem": "걸음 중 충격과 추진의 흔들림 폭이 커져 보행 일관성 확인이 필요합니다.",
            "ok": "걸음 중 충격과 추진의 변동 폭이 비교적 안정적입니다.",
            "risk_when": "high",
            "cut": 0.80,
        },
        {
            "key": "v_harmonic_ratio_iqr",
            "label": "수직 리듬 변동성",
            "value": _safe_float(features.get("v_harmonic_ratio_iqr")),
            "unit": "",
            "problem": "보행 리듬의 구간별 변동이 커져 일정한 보행 리듬 확인이 필요합니다.",
            "ok": "보행 리듬의 구간별 변동이 비교적 낮습니다.",
            "risk_when": "high",
            "cut": 0.10,
        },
    ]
    for item in checks:
        item["is_problem"] = item["value"] < item["cut"] if item["risk_when"] == "low" else item["value"] > item["cut"]
        item["message"] = item["problem"] if item["is_problem"] else item["ok"]
    return checks

def _gait_explainability(model_artifact, features):
    names = model_artifact.get("features", GAIT_FEATURES)
    try:
        import pandas as pd
        frame = pd.DataFrame([[features[name] for name in names]], columns=names)
        pipeline = model_artifact["pipeline"]
        transformed = pipeline[:-1].transform(frame)[0]
        coefs = pipeline[-1].coef_[0]
        raw = [
            {
                "key": name,
                "label": {
                    "v_jerk_rms_median": "수직 충격/추진 대표값",
                    "v_jerk_rms_iqr": "수직 충격 변동성",
                    "v_harmonic_ratio_iqr": "수직 리듬 변동성",
                }.get(name, name),
                "value": float(features[name]),
                "contribution": float(coef * val),
            }
            for name, coef, val in zip(names, coefs, transformed)
        ]
    except Exception as e:
        print(f"[gait explainability error] {e}")
        raw = [
            {"key": name, "label": name, "value": _safe_float(features.get(name)), "contribution": 0.0}
            for name in names
        ]

    max_abs = max([abs(item["contribution"]) for item in raw] + [1.0])
    for item in raw:
        item["direction"] = "risk" if item["contribution"] > 0 else "protective"
        item["width"] = int(min(100, abs(item["contribution"]) / max_abs * 100))
    return raw


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _gait_visual_profile(features):
    jerk_median = _safe_float(features.get("v_jerk_rms_median"))
    hr_iqr = _safe_float(features.get("v_harmonic_ratio_iqr"))
    vertical = _safe_float(features.get("v_amp_pool_median"), max(0.02, min(1.0, jerk_median / 5.0)))
    lateral = _safe_float(features.get("ml_amp_pool_iqr"), max(0.02, hr_iqr))
    regularity = _safe_float(features.get("base_v_stride_regularity"), max(0.0, 1.0 - hr_iqr * 5))
    roll = _safe_float(features.get("roll_amp_pool_iqr"), 10 + hr_iqr * 20)

    step_lift = _clamp(vertical / 0.18)
    lateral_sway = _clamp(lateral / 0.14)
    rhythm = _clamp(regularity)
    trunk_rotation = _clamp(roll / 28.0)
    instability = _clamp((lateral_sway * 0.35) + ((1 - rhythm) * 0.40) + (trunk_rotation * 0.25))
    lateral_blend = int(lateral_sway * 100)
    rhythm_blend = int((1 - rhythm) * 100)
    trunk_blend = int(trunk_rotation * 100)
    normal_blend = int(_clamp(1 - ((lateral_sway + (1 - rhythm) + trunk_rotation) / 3.0)) * 100)

    return {
        "step_lift_pct": int(30 + step_lift * 70),
        "lateral_sway_pct": int(10 + lateral_sway * 90),
        "rhythm_pct": int(rhythm * 100),
        "trunk_rotation_pct": int(10 + trunk_rotation * 90),
        "instability_pct": int(instability * 100),
        "animation_sec": round(1.45 - (rhythm * 0.55), 2),
        "sway_px": round(3 + lateral_sway * 16, 1),
        "bob_px": round(2 + step_lift * 10, 1),
        "rotation_deg": round(4 + trunk_rotation * 18, 1),
        "stride_px": int(16 + step_lift * 42),
        "normal_blend_pct": normal_blend,
        "lateral_blend_pct": lateral_blend,
        "rhythm_blend_pct": rhythm_blend,
        "trunk_blend_pct": trunk_blend,
    }


def _extract_signal_preview(csv_bytes: bytes, window_info: dict, model_dir: str) -> dict | None:
    """CSV 바이트 → 3초 bandpass V/ML 신호 (15Hz) 아바타 애니메이션용."""
    try:
        import joblib
        from gait_axis_aligned_core import (
            load_sensor_csv_with_metadata, _acc_columns,
            align_to_vmlap, resample_array_to_100hz,
            transform_signal, bandpass, TARGET_FS_HZ,
        )
        df, meta = load_sensor_csv_with_metadata(io.BytesIO(csv_bytes))
        acc, already_vmlap, _, _ = _acc_columns(df, meta)
        t = df["Timestamp_ns"].to_numpy(float)
        dur = (float(t.max()) - float(t.min())) / 1e9
        obs_fs = float(len(df) / dur) if dur > 0 else TARGET_FS_HZ
        aligned, _ = align_to_vmlap(acc, already_vmlap=already_vmlap, fs=obs_fs)
        vmlap = resample_array_to_100hz(aligned, obs_fs)   # (N, 3)

        art = joblib.load(os.path.join(model_dir, "gait_daily_clinical_3feat.joblib"))
        alpha = float(art.get("signal_correction", {}).get("alpha", 1.0))
        corrected = transform_signal(vmlap, alpha, 1.0)

        v_bp  = bandpass(corrected[:, 0], TARGET_FS_HZ)
        ml_bp = bandpass(corrected[:, 1], TARGET_FS_HZ)

        start_sec = float(window_info.get("start_sec", 0) or 0)
        s0  = max(0, int(start_sec * TARGET_FS_HZ))
        win = int(3 * TARGET_FS_HZ)
        v_w  = v_bp[s0: s0 + win]
        ml_w = ml_bp[s0: s0 + win]

        DISP_FS = 15
        step = max(1, int(TARGET_FS_HZ / DISP_FS))
        v_d  = v_w[::step].tolist()
        ml_d = ml_w[::step].tolist()

        v_max  = max(max(abs(x) for x in v_d),  1e-6)
        ml_max = max(max(abs(x) for x in ml_d), 1e-6)
        return {
            "v":     [round(x / v_max,  3) for x in v_d],
            "ml":    [round(x / ml_max, 3) for x in ml_d],
            "dt_ms": int(round(1000 / DISP_FS)),
        }
    except Exception as e:
        print(f"[signal preview error] {e}")
        return None


def _get_cognitive_result():
    uid = session.get('uid')
    if not uid or uid not in _store:
        return None
    entry = _store[uid]
    score_result = _compute_score(entry['raw'], entry, entry['sess'])
    if entry.get('assessment_id') and not entry.get('result_saved'):
        complete_assessment(entry['assessment_id'], entry['raw'], score_result)
        entry['result_saved'] = True
    return {
        "score": score_result,
        "version": entry['sess'].version,
        "raw": entry['raw'],
        "participant": {
            "member_code": entry.get('member_code', ''),
            "education_label": entry.get('education_label', ''),
            "phone_last4": entry.get('phone_last4', ''),
        },
    }


CARE_TYPE_MAP = {
    # cognitive_flag: 0=양호, 1=저하 / physical_flag: 0=양호, 1=저하
    (0, 0): {
        "code": "A",
        "name": "건강유지형",
        "title": "A 유형",
        "sub": "(신체 양호 · 인지 양호)",
        "summary": "인지기능과 신체기능이 모두 양호한 상태입니다. 현재의 건강한 상태를 유지하고 향상시키기 위한 예방 중심의 운동을 권장합니다.",
        "focus": "현재 상태 유지",
        "cognitive_status": "양호",
        "physical_status": "양호",
    },
    (0, 1): {
        "code": "C",
        "name": "신체강화형",
        "title": "C 유형",
        "sub": "(신체 저하 · 인지 양호)",
        "summary": "인지기능은 양호하지만 신체기능 관리가 필요한 상태입니다. 안전한 맞춤 운동으로 하체 근력과 균형 능력을 함께 길러보겠습니다.",
        "focus": "보행 및 근력 관리",
        "cognitive_status": "양호",
        "physical_status": "저하",
    },
    (1, 0): {
        "code": "B",
        "name": "인지강화형",
        "title": "B 유형",
        "sub": "(신체 양호 · 인지 저하)",
        "summary": "신체기능은 양호하지만 인지기능 변화 확인과 관리가 필요한 상태입니다. 지금부터 꾸준히 두뇌 운동을 하면 인지 건강을 관리하는 데 도움이 됩니다.",
        "focus": "인지 훈련 중심 통합 관리",
        "cognitive_status": "저하",
        "physical_status": "양호",
    },
    (1, 1): {
        "code": "D",
        "name": "통합강화형",
        "title": "D 유형",
        "sub": "(신체 저하 · 인지 저하)",
        "summary": "인지기능과 신체기능 모두 세심한 관리가 필요한 상태입니다. 보호자와 함께 안전하게 운동하며 몸과 두뇌를 함께 움직여 보세요.",
        "focus": "인지 및 신체 동시 관리",
        "cognitive_status": "저하",
        "physical_status": "저하",
    },
}


def _binary_flag(value):
    if value is None:
        return None
    try:
        return 1 if int(value) else 0
    except (TypeError, ValueError):
        return None


def _classify_care_type(cognitive=None, gait=None):
    cognitive_flag = None
    physical_flag = None

    if cognitive:
        cognitive_flag = _binary_flag(
            cognitive.get("score", {}).get("mci", {}).get("label")
        )
    if gait:
        physical_flag = _binary_flag(gait.get("prediction"))

    if cognitive_flag is None or physical_flag is None:
        return {
            "ready": False,
            "code": "",
            "name": "평가 대기",
            "title": "평가 대기",
            "summary": "인지기능평가와 신체평가를 모두 완료하면 A/B/C/D 유형이 표시됩니다.",
            "focus": "두 평가 완료 필요",
            "cognitive_flag": cognitive_flag,
            "physical_flag": physical_flag,
            "cognitive_status": "대기" if cognitive_flag is None else CARE_TYPE_MAP[(cognitive_flag, 0)]["cognitive_status"],
            "physical_status": "대기" if physical_flag is None else CARE_TYPE_MAP[(0, physical_flag)]["physical_status"],
        }

    result = dict(CARE_TYPE_MAP[(cognitive_flag, physical_flag)])
    result.update({
        "ready": True,
        "cognitive_flag": cognitive_flag,
        "physical_flag": physical_flag,
    })
    return result


def _basic_pengteu_reply(message, context, knowledge=None):
    member = (context or {}).get("member") or {}
    assessment = (context or {}).get("latest_assessment") or {}
    physical = (context or {}).get("latest_physical") or {}
    exercise = (context or {}).get("exercise_summary") or {}
    profile = (context or {}).get("assistant_profile") or {}
    knowledge = knowledge or []

    name = profile.get("persona_name") or "펭트"
    member_code = member.get("member_code") or "회원님"
    final_score = assessment.get("final_score")
    gait_prediction = None
    raw_gait = physical.get("raw_json") if physical else {}
    if isinstance(raw_gait, dict):
        gait_prediction = raw_gait.get("prediction")
    streak = _safe_int(exercise.get("streak_days"))
    evidence_hint = ""
    if knowledge:
        titles = []
        for item in knowledge[:2]:
            title = item.get("title") or item.get("source") or ""
            if title and title not in titles:
                titles.append(title)
        if titles:
            evidence_hint = f" 제가 참고한 기준은 {', '.join(titles)} 쪽이에요."

    lowered = (message or "").lower()
    if "보행" in message or "걷" in message:
        if gait_prediction == 1:
            return f"{name}가 볼 때 {member_code}의 최근 보행 결과는 신체기능 관리가 필요한 신호가 있어요. 오늘은 빠르게 걷기보다 허리에 스마트폰을 잘 고정하고, 천천히 균형을 지키는 운동부터 해볼게요.{evidence_hint}"
        if gait_prediction == 0:
            return f"{name}가 확인했어요. {member_code}의 최근 보행 결과는 정상 범위 가능성이 높아요. 그래도 매일 조금씩 걷기와 균형 운동을 이어가면 좋아요.{evidence_hint}"
        return f"{name}가 아직 최신 보행 결과를 찾지 못했어요. 스마트폰을 허리에 고정하고 20초 이상 평소처럼 걸어서 먼저 측정해볼게요.{evidence_hint}"

    if "인지" in message or "moca" in lowered or "점수" in message:
        if final_score is not None:
            return f"{name}가 최근 인지평가를 확인했어요. MoCA 최종 점수는 {final_score}점이에요. 점수 하나로 단정하지 않고 기억력, 주의력, 실행기능 흐름을 같이 보면서 설명해드릴게요.{evidence_hint}"
        return f"{name}가 아직 완료된 인지평가를 찾지 못했어요. 먼저 MoCA 평가를 끝내면 결과를 바탕으로 쉽게 설명해드릴게요.{evidence_hint}"

    if "운동" in message or "오늘" in message:
        if streak > 0:
            return f"{name}가 응원합니다. 지금 {streak}일 연속 운동 기록이 있어요. 오늘은 무리하지 말고 화면 안내에 맞춰 천천히 이어가면 됩니다.{evidence_hint}"
        return f"{name}가 오늘 운동을 같이 도와드릴게요. 먼저 현재 유형에 맞는 운동을 시작하고, 센서 기준값이 준비되면 동작 성공 여부도 자동으로 확인할 수 있어요.{evidence_hint}"

    if "기여도" in message or "xai" in lowered or "shap" in lowered:
        return f"{name}가 쉽게 말해드릴게요. 모델 기여도는 이번 보행 판단에서 어떤 보행 피처가 위험 쪽으로 밀었고, 어떤 피처가 정상 쪽으로 도왔는지 보여주는 설명이에요. 지금은 로지스틱 회귀의 표준화 피처와 계수를 이용한 설명이고, 나중에 SHAP을 붙이면 더 정식 XAI로 보여줄 수 있어요.{evidence_hint}"

    if "스펙트럼" in message or "주파수" in message:
        return f"{name}가 설명해드릴게요. 스펙트럼은 허리 가속도 원신호가 시간에 따라 어떤 리듬과 주파수 패턴을 보였는지 보여주는 보조 그림이에요. 모델은 최종 3개 보행 피처로 판단하고, 스펙트럼은 그 판단을 이해하기 쉽게 돕는 시각 자료예요.{evidence_hint}"

    if knowledge:
        if "tts" in lowered or "bgm" in lowered or "mp3" in lowered or "음악" in message or "소리" in message:
            return f"{name}예요. 운동 음악은 그대로 배경음으로 두고, 제가 말할 때만 운동 안내음과 효과음을 잠깐 낮추거나 멈추게 할게요. 제 말이 끝나면 운동 화면의 음악은 원래 볼륨으로 돌아가요."
        return f"{name}예요. 관련 자료는 제가 참고만 했고, 화면에는 회원님께 필요한 내용만 짧게 말할게요. 더 자세히 알고 싶은 부분을 말해주시면 보행, 인지, 운동 기록에 맞춰 쉽게 풀어드릴게요."
        top = knowledge[0]
        return f"{name}예요. 질문과 가까운 자료를 찾아봤어요. {top.get('text', '')} 이 내용을 바탕으로 {member_code}에게 맞게 더 쉽게 설명해드릴게요."

    return f"{name}예요. 저는 {member_code}의 인지평가, 보행평가, 운동기록을 함께 보면서 상황에 맞게 설명하고 안내할 준비가 되어 있어요."


def _pengteu_local_answer_ready(message, knowledge=None):
    text = (message or "").lower()
    keywords = (
        "보행", "걷", "걸음", "운동", "오늘", "moca", "인지", "점수",
        "낙상", "센서", "보정", "보호자", "글씨", "볼륨", "속도",
        "기준", "모델", "라벨", "기여도", "스펙트럼", "주파수", "xai", "shap",
        "threshold", "gait", "fall", "sensor", "exercise", "score",
    )
    return any(keyword in text for keyword in keywords)


def _compact_pengteu_context(context, knowledge=None):
    context = context or {}
    member = context.get("member") or {}
    assessment = context.get("latest_assessment") or {}
    physical = context.get("latest_physical") or {}
    exercise = context.get("exercise_summary") or {}
    profile = context.get("assistant_profile") or {}
    raw_gait = physical.get("raw_json") if isinstance(physical, dict) else {}
    if not isinstance(raw_gait, dict):
        raw_gait = {}
    return {
        "member_code": member.get("member_code"),
        "moca_final_score": assessment.get("final_score"),
        "gait_prediction": raw_gait.get("prediction"),
        "gait_probability": raw_gait.get("probability"),
        "exercise_streak_days": exercise.get("streak_days"),
        "exercise_present_days": exercise.get("present_days"),
        "assistant_profile": {
            "voice_rate": profile.get("voice_rate"),
            "tts_volume": profile.get("tts_volume"),
            "text_scale": profile.get("text_scale"),
            "high_contrast": profile.get("high_contrast"),
        },
        "retrieved_knowledge": [
            {
                "source": item.get("source"),
                "title": item.get("title"),
                "text": (item.get("text") or "")[:700],
            }
            for item in (knowledge or [])[:3]
        ],
    }


def _extract_response_text(payload):
    if not isinstance(payload, dict):
        return ""
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()
    texts = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in ("output_text", "text") and content.get("text"):
                texts.append(str(content.get("text")))
    return "\n".join(texts).strip()


def _openai_pengteu_fallback(message, context, knowledge=None):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.environ.get("OPENAI_ASSISTANT_MODEL", "gpt-4.1-mini").strip()
    system_prompt = (
        "너는 고령 사용자를 돕는 펭트 AI 어시스턴트다. "
        "진단을 확정하지 말고 선별/주의 표현을 사용한다. "
        "답변은 한국어로 2~4문장, 쉽고 따뜻하게 말한다. "
        "사용자 기록과 검색 지식 안에서만 개인 결과를 설명하고, 모르는 것은 모른다고 말한다."
    )
    body = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": json.dumps({
                        "question": message,
                        "context": _compact_pengteu_context(context, knowledge),
                    }, ensure_ascii=False),
                }],
            },
        ],
        "max_output_tokens": 220,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return _extract_response_text(payload) or None
    except Exception as exc:
        print(f"[pengteu openai fallback error] {exc}")
        return None


def _clean_pengteu_reply(reply, message=""):
    text = (reply or "").strip()
    blocked = (
        "질문과 가까운 자료를 찾아봤어요",
        "이 내용을 바탕으로",
        "retrieved_knowledge",
        "RAG",
        "/static/audio",
        "`/static/audio`",
    )
    if any(token in text for token in blocked):
        lowered = (message or "").lower()
        if "tts" in lowered or "bgm" in lowered or "mp3" in lowered or "음악" in message or "소리" in message:
            return "펭트예요. 운동 음악은 배경음으로 그대로 두고, 제가 말할 때만 운동 안내음과 효과음을 잠깐 낮추거나 멈출게요. 제 말이 끝나면 음악은 다시 원래 볼륨으로 돌아가요."
        return "펭트예요. 자료는 제가 참고만 하고, 화면에는 필요한 말만 짧게 설명할게요. 궁금한 부분을 한 번 더 말해주시면 쉽게 풀어드릴게요."
    if len(text) > 320:
        text = text[:317].rstrip() + "..."
    return text


def _save_gait_result(gait_result):
    gait_result_id = session.get('gait_result_id') or uuid4().hex
    _gait_result_store[gait_result_id] = gait_result
    session['gait_result_id'] = gait_result_id
    session.pop('gait_result', None)
    member_id = _current_member_id()
    if member_id:
        probability = _safe_float(gait_result.get("probability"))
        save_physical_result(member_id, _current_assessment_id(), {
            "gait_type": gait_result.get("model_mode") or "daily_gait",
            "gait_level": gait_result.get("label") or "",
            "gait_score": max(0, min(100, round((1.0 - probability) * 100))),
            "measured_at": datetime.now().isoformat(timespec="seconds"),
            **gait_result,
        })


def _get_gait_result():
    gait_result_id = session.get('gait_result_id')
    if gait_result_id and gait_result_id in _gait_result_store:
        return _gait_result_store[gait_result_id]
    return session.get('gait_result')


def _current_member_id():
    member_id = session.get('member_id')
    uid = session.get('uid')
    if not member_id and uid in _store:
        member_id = _store[uid].get('member_id')
        if member_id:
            session['member_id'] = member_id
    return member_id


def _current_assessment_id():
    uid = session.get('uid')
    if uid in _store:
        return _store[uid].get('assessment_id')
    return session.get('assessment_id')


def _ensure_member_from_phone(phone, education_level="high"):
    normalized = normalize_phone(phone)
    if len(normalized) < 9:
        raise ValueError("전화번호를 다시 확인해 주세요.")
    existing = find_member_by_phone(normalized)
    if existing:
        member_id = existing["id"]
        member_code = existing.get("member_code") or existing.get("name")
        edu = existing.get("education_years")
        level = existing.get("education_level") or education_level
    else:
        member_id, edu, member_code, _ = get_or_create_member(normalized, education_level)
        level = education_level
    session["member_id"] = member_id
    session["member_code"] = member_code
    session["education_years"] = edu
    session["education_level"] = level
    session["phone_last4"] = phone_last4(normalized)
    return member_id, int(edu), member_code, level


def _restore_member_session(member, education_level=None):
    level = education_level or member.get("education_level") or "high"
    item = EDUCATION_LEVELS.get(level, EDUCATION_LEVELS["high"])
    session["access_granted"] = True
    session["member_id"] = member["id"]
    session["member_code"] = member["name"]
    session["education_years"] = int(member.get("education_years") or item["years"])
    session["education_level"] = level
    session["education_label"] = education_label(level)
    session["phone_last4"] = member.get("phone_last4", "")
    session["is_new_member"] = False


def _exercise_mock_data():
    user_name = session.get('user_name') or '어르신'
    return {
        'user': {
            'name': user_name,
            'label': '시니어',
            'health_score': 82,
            'stars': 4,
        },
        'today': {
            'type': 'C유형 맞춤',
            'mission_title': '오늘의 미션',
            'mission': '걷기 + 기억력 게임',
            'duration_min': 15,
            'exercise_name': '제자리 걷기',
            'sets': '3세트',
            'set_duration': '각 2분',
            'notice': '휴대폰을 허리에 고정하고 가볍게 걷는 동작을 유지하세요.',
        },
        'attendance': {
            'streak_days': 4,
            'days': [
                {'label': '월', 'state': 'empty'},
                {'label': '화', 'state': 'done'},
                {'label': '수', 'state': 'done'},
                {'label': '목', 'state': 'done'},
                {'label': '금', 'state': 'done'},
                {'label': '토', 'state': 'fire'},
                {'label': '일', 'state': 'empty'},
            ],
            'present_days': 16,
            'missed_days': 3,
            'best_streak': 4,
        },
        'growth': {
            'tree_level': 'Lv.3',
            'title': '건강 나무',
            'message': '신체나이가 7일 젊어졌어요',
            'progress_pct': 70,
            'remaining': '3일 더하면 레벨업',
        },
        'report': {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_score': 82,
            'score_delta': 7,
            'cognitive_score': 60,
            'cognitive_delta': 3,
            'gait_score': 75,
            'gait_delta': 5,
            'exercise_minutes': 47,
            'exercise_goal_minutes': 60,
            'badges': 4,
        },
    }


def _overlay_personal_records(exercise):
    member_id = _current_member_id()
    member = get_member(member_id)
    latest_assessment = get_latest_assessment(member_id)
    latest_physical = get_latest_physical_result(member_id)
    exercise_summary = get_exercise_summary(member_id)

    if member:
        exercise['user']['name'] = member.get('member_code') or member.get('name') or exercise['user']['name']
        exercise['user']['label'] = education_label(member.get('education_level', 'high'))

    if latest_assessment:
        final_score = _safe_int(latest_assessment.get('final_score'))
        cognitive_score = round(final_score / 30 * 100) if final_score <= 30 else final_score
        exercise['report']['cognitive_score'] = max(0, min(100, cognitive_score))
        exercise['report']['moca_score'] = final_score
        exercise['report']['date'] = (
            latest_assessment.get('completed_at') or
            latest_assessment.get('started_at') or
            exercise['report']['date']
        )[:10]

    if latest_physical:
        gait_score = _safe_int(latest_physical.get('gait_score'))
        cognitive_score = latest_physical.get('cognitive_score')
        exercise['report']['gait_score'] = max(0, min(100, gait_score))
        if cognitive_score and not latest_assessment:
            exercise['report']['cognitive_score'] = max(0, min(100, _safe_int(cognitive_score)))
        exercise['report']['gait_type'] = latest_physical.get('gait_type') or ''
        exercise['report']['gait_level'] = latest_physical.get('gait_level') or ''
        exercise['report']['date'] = (latest_physical.get('measured_at') or exercise['report']['date'])[:10]

    if exercise_summary:
        present_days = _safe_int(exercise_summary.get('present_days'))
        streak_days = _safe_int(exercise_summary.get('streak_days'))
        total_minutes = _safe_int(exercise_summary.get('total_minutes'))
        exercise['attendance']['present_days'] = present_days or exercise['attendance']['present_days']
        exercise['attendance']['streak_days'] = streak_days
        exercise['attendance']['best_streak'] = max(exercise['attendance'].get('best_streak', 0), streak_days)
        exercise['report']['exercise_minutes'] = total_minutes or exercise['report']['exercise_minutes']

    cog = _safe_int(exercise['report'].get('cognitive_score'))
    gait = _safe_int(exercise['report'].get('gait_score'))
    minutes = _safe_int(exercise['report'].get('exercise_minutes'))
    goal = max(1, _safe_int(exercise['report'].get('exercise_goal_minutes'), 60))
    exercise_score = min(100, round(minutes / goal * 100))
    exercise['report']['total_score'] = round((cog + gait + exercise_score) / 3)
    exercise['user']['health_score'] = exercise['report']['total_score']
    exercise['user']['stars'] = max(1, min(5, round(exercise['report']['total_score'] / 20)))
    return exercise


def _personal_exercise_data():
    return _overlay_personal_records(_exercise_mock_data())


def _record_exercise_completion(exercise):
    member_id = _current_member_id()
    if not member_id:
        return
    today = exercise.get('today', {})
    save_exercise_record(member_id, _current_assessment_id(), {
        'exercise_name': today.get('exercise_name') or 'exercise',
        'type': today.get('type') or today.get('mission'),
        'duration_min': today.get('duration_min') or 0,
    })


def _physical_mock_data():
    return {
        'title': '보행 측정',
        'instruction': '스마트폰을 허리에 고정한 상태로 안전하게 걸어주세요.',
        'duration_seconds': 20,
        'result': {
            'testType': 'physical_gait',
            'gaitType': 'C유형',
            'gaitLevel': '활력 증진형',
            'gaitScore': 75,
            'cognitiveScore': 60,
            'walkingSpeed': 0.82,
            'stepCount': 12,
        },
    }


def _load_cnn_models():
    """앱 최초 채점 시 1회 호출 — .pth 파일 존재 여부에 따라 자동 로드"""
    global _cnn_cube, _cnn_clock, _cnn_clock_dev

    base = os.path.dirname(os.path.abspath(__file__))

    # cube CNN: 합성 데이터로만 학습되어 실제 손그림 오판 → 룰베이스 사용
    # (실제 QuickDraw/손그림 데이터 확보 후 재활성화)

    if _cnn_clock is None:
        paths = {
            "deepc": os.path.join(base, "deepc.pth"),
            "deeph": os.path.join(base, "deeph.pth"),
            "deepn": os.path.join(base, "deepn.pth"),
        }
        if all(os.path.exists(p) for p in paths.values()):
            try:
                from clock_cnn_inference import load_models
                _cnn_clock, _cnn_clock_dev = load_models(
                    paths["deepc"], paths["deeph"], paths["deepn"]
                )
                print(f"[CNN] clock 모델 로드: {list(_cnn_clock.keys())}")
            except Exception as e:
                print(f"[CNN] clock 로드 실패: {e}")

ITEM_TITLES = {
    'trail_making':     '길 만들기',
    'cube':             '도형 따라 그리기',
    'clock':            '시계 그리기',
    'naming':           '이름 말하기',
    'memory_immediate': '단어 기억하기',
    'forward_digits':   '숫자 따라 말하기',
    'backward_digits':  '숫자 거꾸로 말하기',
    'clapping':         '화면 치기',
    'serial_7':         '빼기 계산',
    'sentence_repeat':  '따라 말하기',
    'verbal_fluency':   '단어 말하기',
    'abstraction':      '공통점 찾기',
    'delayed_recall':   '단어 기억 떠올리기',
    'orientation':      '날짜와 장소',
}

ITEM_TYPE = {
    'trail_making':     'drawing',
    'cube':             'drawing',
    'clock':            'drawing',
    'naming':           'naming',
    'memory_immediate': 'memory',
    'forward_digits':   'voice',
    'backward_digits':  'voice',
    'clapping':         'clapping',
    'serial_7':         'voice',
    'sentence_repeat':  'voice_multi',
    'verbal_fluency':   'voice',
    'abstraction':      'voice_multi',
    'delayed_recall':   'voice',
    'orientation':      'orientation',
}

# 빈 응답 초기값
def _empty_raw():
    return {
        'trail_touch_points': [],
        'canvas_width': 512, 'canvas_height': 512,
        'cube_score': 0,
        'clock_contour': 0, 'clock_numbers': 0, 'clock_hands': 0,
        'animal1_stt': '', 'animal2_stt': '', 'animal3_stt': '',
        'immediate1_stt': '', 'immediate2_stt': '',
        'delayed_recall_stt': '',
        'forward_stt': '', 'backward_stt': '',
        'tapped_indices': [],
        'serial7_stt': '',
        'sentence1_stt': '', 'sentence2_stt': '',
        'fluency_stt': '',
        'abstraction_pair1_stt': '', 'abstraction_pair2_stt': '',
        'year_stt': '', 'month_stt': '', 'day_stt': '',
        'weekday_stt': '', 'place_stt': '', 'sigungu_stt': '',
    }


def _tts_urls(item_name, version):
    v = 'moca_k' if version == 'MoCA-K' else 'k_moca'
    mp = {
        'trail_making':     ['trail_making_inst.mp3'],
        'cube':             ['cube_inst.mp3'],
        'clock':            ['clock_inst.mp3'],
        'naming':           ['naming_inst.mp3'],
        'memory_immediate': ['memory_inst.mp3'] + [f'{v}_word{i}.mp3' for i in range(1, 6)],
        'forward_digits':   ['digit_forward_inst.mp3', 'digit_forward_seq.mp3'],
        'backward_digits':  ['digit_backward_inst.mp3', 'digit_backward_seq.mp3'],
        'clapping':         [f'clap_inst_{v}.mp3'],
        'serial_7':         ['serial7_inst.mp3'],
        'sentence_repeat':  ['repeat_inst.mp3'],
        'verbal_fluency':   [f'fluency_inst_{v}.mp3'],
        'abstraction':      ['abstract_inst.mp3'],
        'delayed_recall':   ['delayed_recall_inst.mp3'],
        'orientation':      ['orientation_date_inst.mp3', 'orientation_year.mp3'],
    }
    return [f'/audio/{f}' for f in mp.get(item_name, [])]


# ── 라우트 ────────────────────────────────────

@app.route('/')
def home():
    if request.args.get('registered') == '1' and not request.args.get('select'):
        return redirect(url_for('main_home_page'))
    is_new = bool(request.args.get('registered')) or session.get('is_new_member', False)
    template = 'home_new.html' if is_new else 'home.html'
    return render_template(
        template,
        education_levels=EDUCATION_LEVELS,
        error=request.args.get('error', ''),
        assessment_phase=request.args.get('phase', ''),
        profile_ready=bool(session.get('member_id')),
        registered=request.args.get('registered', ''),
    )


@app.route('/main_home')
def main_home_page():
    return render_template(
        'main_home.html',
        exercise=_personal_exercise_data(),
        cognitive=_get_cognitive_result(),
        gait=_get_gait_result()
    )


@app.route('/physical/guide')
def physical_guide_page():
    return render_template('physical_guide.html', measure_url='/physical/measure')


@app.route('/physical/measure')
def physical_measure_page():
    return render_template('physical_measure.html', physical=_physical_mock_data())


@app.route('/physical/result')
def physical_result_page():
    return render_template('physical_result.html')


@app.route('/physical/save', methods=['POST'])
def physical_save_page():
    member_id = _current_member_id()
    if not member_id:
        return jsonify({'ok': False, 'error': 'member_not_found'}), 400
    data = request.get_json(silent=True) or {}
    save_physical_result(member_id, _current_assessment_id(), data)
    return jsonify({'ok': True})


@app.route('/exercise')
def exercise_redirect():
    return redirect(url_for('exercise_today_page'))


@app.route('/exercise/today')
def exercise_today_page():
    return render_template('exercise_today.html', exercise=_personal_exercise_data())


@app.route('/exercise/active')
def exercise_active_page():
    return render_template('exercise_active.html', exercise=_personal_exercise_data())


@app.route('/exercise/complete')
def exercise_complete_page():
    exercise = _personal_exercise_data()
    _record_exercise_completion(exercise)
    exercise = _personal_exercise_data()
    return render_template('exercise_complete.html', exercise=exercise)


@app.route('/exercise/sensor/analyze', methods=['POST'])
def exercise_sensor_analyze():
    upload = request.files.get('file')
    exercise_type = request.form.get('exercise_type') or request.args.get('exercise_type')
    if upload is None:
        data = request.get_json(silent=True) or {}
        exercise_type = exercise_type or data.get('exercise_type')
        csv_text = data.get('csv')
        if not csv_text:
            return jsonify({'ok': False, 'error': 'missing_csv_file'}), 400
        from io import StringIO
        source = StringIO(csv_text)
    else:
        source = upload.stream

    if not exercise_type:
        return jsonify({'ok': False, 'error': 'missing_exercise_type'}), 400

    try:
        result = analyze_exercise_csv(source, exercise_type)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True, 'result': result})


@app.route('/report/detail')
def report_detail_page():
    return render_template(
        'report_detail.html',
        exercise=_personal_exercise_data(),
        back_url='/exercise/complete',
        cognitive=_get_cognitive_result(),
        gait=_get_gait_result(),
        panel=request.args.get('panel', 'cognitive')
    )


@app.route('/mypage')
def mypage_page():
    cog = _get_cognitive_result()
    gait_res = _get_gait_result()
    return render_template(
        'mypage.html',
        exercise=_personal_exercise_data(),
        back_url='/main_home',
        cognitive=cog,
        gait=gait_res,
        care_type=_classify_care_type(cog, gait_res),
        panel=request.args.get('panel', 'cognitive')
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    password = _access_password()
    if not password:
        return redirect(url_for('home'))

    error = None
    next_url = request.args.get('next') or url_for('home')
    if request.method == 'POST':
        submitted = request.form.get('password', '')
        next_url = request.form.get('next') or url_for('home')
        if hmac.compare_digest(submitted, password):
            session['access_granted'] = True
            return redirect(next_url if next_url.startswith('/') else url_for('home'))
        error = '비밀번호가 맞지 않습니다.'

    return render_template('login.html', error=error, next_url=next_url)


@app.route('/gait')
def gait_page():
    return render_template('gait.html', model=_gait_model_summary())


def _predict_gait_from_payload(data):
    models = _load_gait_models()
    model_artifact = models["daily"]["model"]
    if model_artifact is None:
        return None, ({'ok': False, 'error': '최종 보행 평가 모델을 불러오지 못했습니다.'}, 503)

    def parse_feature(feature):
        value = data.get(feature)
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    required_values = {feature: parse_feature(feature) for feature in GAIT_REQUIRED_FEATURES}
    if any(value is None for value in required_values.values()):
        return None, ({'ok': False, 'error': '최종 보행 feature 3개가 부족합니다. CSV 측정 또는 최종 피처 payload를 사용해 주세요.'}, 400)

    model_features = model_artifact.get('features', GAIT_FEATURES)
    try:
        values = [float(required_values[feature]) for feature in model_features]
    except (KeyError, TypeError, ValueError):
        return None, ({'ok': False, 'error': '보행 feature 조합이 최종 모델과 맞지 않습니다. 다시 측정해 주세요.'}, 400)

    import pandas as pd
    frame = pd.DataFrame([values], columns=model_features)
    probability = float(model_artifact['pipeline'].predict_proba(frame)[:, 1][0])
    threshold = float(model_artifact.get('threshold', 0.5))
    prediction = int(probability >= threshold)
    features = dict(zip(model_features, values))
    model_mode = model_artifact.get('model_mode', 'daily_subwindow_clinical_acc_only_3feat')
    threshold_strategy = model_artifact.get('threshold_strategy', 'fixed_screening_threshold_sensitivity_prioritized')
    gait_result = {
        'probability': probability,
        'threshold': threshold,
        'prediction': prediction,
        'label': '이동기능 저하 가능성' if prediction else '이동기능 정상 범위 가능성',
        'threshold_strategy': threshold_strategy,
        'model_mode': model_mode,
        'features': features,
        'insights': _gait_feature_insights(features),
        'explainability': _gait_explainability(model_artifact, features),
        'visual': _gait_visual_profile(features),
        'window': data.get('_window') or {},
    }
    _save_gait_result(gait_result)
    response = {
        'ok': True,
        'probability': probability,
        'threshold': threshold,
        'prediction': prediction,
        'label': '이동기능 저하 가능성' if prediction else '이동기능 정상 범위 가능성',
        'threshold_strategy': threshold_strategy,
        'model_mode': model_mode,
        'features': features,
        'window': data.get('_window') or {},
        'redirect_url': url_for('physical_to_cognitive_page') if session.get('is_new_member') else url_for('gait_avatar_page'),
    }
    return gait_result, (response, 200)


@app.route('/gait/predict', methods=['POST'])
def gait_predict():
    data = request.get_json(silent=True) or {}
    _, result = _predict_gait_from_payload(data)
    body, status = result
    return jsonify(body), status


@app.route('/gait/upload-csv', methods=['POST'])
def gait_upload_csv():
    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({'ok': False, 'error': 'CSV 파일을 선택해 주세요.'}), 400
    member_phone = request.form.get("member_phone") or request.form.get("phone") or request.args.get("member_phone") or ""
    if member_phone:
        try:
            _ensure_member_from_phone(member_phone, request.form.get("education_level") or "high")
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
    model_dir = os.path.join(app.root_path, "models")
    daily_model_path = os.path.join(model_dir, "gait_daily_clinical_3feat.joblib")

    if os.path.exists(daily_model_path):
        csv_bytes = upload.read()
        try:
            result = predict_daily_gait_csv(io.BytesIO(csv_bytes), model_dir)
        except Exception as e:
            print(f"[gait csv preprocessing error] {e}")
            return jsonify({
                'ok': False,
                'error': '보행 데이터가 충분하지 않아 평가할 수 없습니다. 스마트폰을 허리에 고정하고 평소처럼 20초 이상 걸어 다시 측정해 주세요.'
            }), 400

        feats    = result['features']
        if not _has_final_gait_features(feats):
            return jsonify({
                'ok': False,
                'error': '보행 데이터가 충분하지 않아 평가할 수 없습니다. 스마트폰을 허리에 고정하고 평소처럼 20초 이상 걸어 다시 측정해 주세요.'
            }), 400

        jerk_med = _safe_float(feats.get('v_jerk_rms_median'))
        jerk_iqr = _safe_float(feats.get('v_jerk_rms_iqr'))
        hr_iqr   = _safe_float(feats.get('v_harmonic_ratio_iqr'))
        model_artifact = (_load_gait_models().get("daily") or {}).get("model")
        gait_result = {
            'probability':        result['probability'],
            'threshold':          result['threshold'],
            'prediction':         result['prediction'],
            'label':              result['label'],
            'threshold_strategy': result['threshold_strategy'],
            'model_mode':         result['model_mode'],
            'features':           feats,
            'insights': [
                {
                    'key':        'v_jerk_rms_median',
                    'label':      '수직 충격/추진 대표값',
                    'value':      jerk_med,
                    'unit':       ' g/s',
                    'ref_normal': '≥ 1.2',
                    'is_problem': jerk_med < 1.2,
                    'message':    '걸음의 수직 충격과 추진 수준이 낮아 보폭이나 보행 힘을 확인할 필요가 있습니다.' if jerk_med < 1.2 else '걸음의 수직 충격과 추진 수준이 비교적 양호합니다.',
                },
                {
                    'key':        'v_jerk_rms_iqr',
                    'label':      '수직 충격 변동성',
                    'value':      jerk_iqr,
                    'unit':       ' g/s',
                    'ref_normal': '≤ 0.80',
                    'is_problem': jerk_iqr > 0.80,
                    'message':    '걸음 중 충격과 추진의 흔들림 폭이 커져 보행 일관성 확인이 필요합니다.' if jerk_iqr > 0.80 else '걸음 중 충격과 추진의 변동 폭이 비교적 안정적입니다.',
                },
                {
                    'key':        'v_harmonic_ratio_iqr',
                    'label':      '수직 리듬 변동성',
                    'value':      hr_iqr,
                    'unit':       '',
                    'ref_normal': '≤ 0.10',
                    'is_problem': hr_iqr > 0.10,
                    'message':    '보행 리듬의 구간별 변동이 커져 일정한 보행 리듬 확인이 필요합니다.' if hr_iqr > 0.10 else '보행 리듬의 구간별 변동이 비교적 낮습니다.',
                },
            ],
            'explainability': _gait_explainability(model_artifact, feats) if model_artifact else [],
            'visual': _gait_visual_profile({
                'v_amp_pool_median':        max(0.02, min(1.0, jerk_med / 5.0)),
                'ml_amp_pool_iqr':          max(0.02, hr_iqr),
                'base_v_stride_regularity': max(0.0,  1.0 - hr_iqr * 5),
                'roll_amp_pool_iqr':        10 + hr_iqr * 20,
            }),
            'window':         result['window'],
            'signal_preview': _extract_signal_preview(csv_bytes, result.get('window', {}), model_dir),
        }
        _save_gait_result(gait_result)
        return jsonify({
            'ok': True,
            'probability':        result['probability'],
            'threshold':          result['threshold'],
            'prediction':         result['prediction'],
            'label':              result['label'],
            'threshold_strategy': result['threshold_strategy'],
            'model_mode':         result['model_mode'],
            'features':           feats,
            'window':             result['window'],
            'extracted_features': feats,
            'redirect_url':       url_for('gait_avatar_page'),
        }), 200

    return jsonify({'ok': False, 'error': '최종 보행 모델 파일을 찾지 못했습니다.'}), 503


@app.route('/gait/avatar')
def gait_avatar_page():
    gait_result = _get_gait_result()
    if not gait_result:
        return redirect(url_for('gait_page'))
    return render_template('gait_avatar.html', gait=gait_result)


@app.route('/phone/send-code', methods=['POST'])
def send_phone_code():
    data = request.get_json(silent=True) or {}
    phone = normalize_phone(data.get('phone', ''))
    if len(phone) < 9:
        return jsonify({'ok': False, 'error': '전화번호를 다시 확인해 주세요.'}), 400

    code = f"{secrets.randbelow(1000000):06d}"
    expires_at = datetime.now() + timedelta(minutes=5)
    session['phone_verification'] = {
        'phone_hash': phone_hash(phone),
        'phone_last4': phone_last4(phone),
        'code': code,
        'expires_at': expires_at.isoformat(),
        'verified': False,
    }

    # 시연용: 실제 SMS 연동 전까지 화면에서 인증번호를 확인한다.
    return jsonify({
        'ok': True,
        'message': '인증번호가 발급되었습니다.',
        'demo_code': code,
        'expires_in': 300,
    })


@app.route('/phone/verify-code', methods=['POST'])
def verify_phone_code():
    data = request.get_json(silent=True) or {}
    phone = normalize_phone(data.get('phone', ''))
    code = (data.get('code') or '').strip()
    saved = session.get('phone_verification') or {}

    if not saved or saved.get('phone_hash') != phone_hash(phone):
        return jsonify({'ok': False, 'error': '인증번호를 다시 발급해 주세요.'}), 400

    try:
        expires_at = datetime.fromisoformat(saved.get('expires_at', ''))
    except ValueError:
        return jsonify({'ok': False, 'error': '인증번호를 다시 발급해 주세요.'}), 400

    if datetime.now() > expires_at:
        return jsonify({'ok': False, 'error': '인증번호가 만료되었습니다.'}), 400

    if not hmac.compare_digest(saved.get('code', ''), code):
        return jsonify({'ok': False, 'error': '인증번호가 맞지 않습니다.'}), 400

    saved['verified'] = True
    session['phone_verification'] = saved
    return jsonify({'ok': True, 'message': '전화번호 인증이 완료되었습니다.'})


def _start_assessment(member_id, edu, member_code, education_level, loc='', sgg='', phone_last=''):
    from session_manager import create_session

    uid = f"{member_code}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    sess = create_session(uid, edu)
    sess.start()
    assessment_id = create_assessment(uid, member_id, sess.version, loc, sgg)

    _store[uid] = {
        'sess': sess,
        'raw': _empty_raw(),
        'location': loc,
        'sigungu': sgg,
        'member_id': member_id,
        'assessment_id': assessment_id,
        'member_code': member_code,
        'education_level': education_level,
        'education_label': education_label(education_level),
        'phone_last4': phone_last,
        'result_saved': False,
    }
    session['uid'] = uid
    session['member_id'] = member_id
    session['assessment_id'] = assessment_id

    return redirect(url_for('item_page'))


@app.route('/profile', methods=['POST'])
def save_profile():
    phone = normalize_phone(request.form.get('phone', ''))
    education_level = request.form.get('education_level', 'high')
    loc = request.form.get('location', '').strip()
    sgg = request.form.get('sigungu', '').strip()

    verified = session.get('phone_verification') or {}
    if not verified.get('verified') or verified.get('phone_hash') != phone_hash(phone):
        return redirect(url_for('home', error='전화번호 인증을 먼저 완료해 주세요.'))

    try:
        member_id, edu, member_code, is_new = get_or_create_member(phone, education_level)
    except ValueError as e:
        return redirect(url_for('home', error=str(e)))

    user_name = request.form.get('user_name', '').strip()
    if user_name:
        session['user_name'] = user_name

    session['member_id'] = member_id
    session['member_code'] = member_code
    session['education_years'] = edu
    session['education_level'] = education_level
    session['education_label'] = education_label(education_level)
    session['phone_last4'] = phone_last4(phone)
    session['location'] = loc
    session['sigungu'] = sgg
    session['is_new_member'] = is_new

    return redirect(url_for('home', registered='1', select='1'))


@app.route('/start_saved')
def start_saved():
    member_id = session.get('member_id')
    member_code = session.get('member_code')
    edu = session.get('education_years')
    education_level = session.get('education_level', 'high')
    if not member_id or not member_code or edu is None:
        return redirect(url_for('home', error='기본정보를 먼저 입력해 주세요.'))

    return _start_assessment(
        member_id,
        int(edu),
        member_code,
        education_level,
        session.get('location', ''),
        session.get('sigungu', ''),
        session.get('phone_last4', ''),
    )


@app.route('/start', methods=['POST'])
def start():
    phone = normalize_phone(request.form.get('phone', ''))
    education_level = request.form.get('education_level', 'high')
    loc = request.form.get('location', '').strip()
    sgg = request.form.get('sigungu', '').strip()

    verified = session.get('phone_verification') or {}
    if not verified.get('verified') or verified.get('phone_hash') != phone_hash(phone):
        return redirect(url_for('home', error='전화번호 인증을 먼저 완료해 주세요.'))

    try:
        member_id, edu, member_code, is_new = get_or_create_member(phone, education_level)
    except ValueError as e:
        return redirect(url_for('home', error=str(e)))

    session['member_id'] = member_id
    session['member_code'] = member_code
    session['education_years'] = edu
    session['education_level'] = education_level
    session['education_label'] = education_label(education_level)
    session['phone_last4'] = phone_last4(phone)
    session['location'] = loc
    session['sigungu'] = sgg
    session['is_new_member'] = is_new

    return _start_assessment(member_id, edu, member_code, education_level, loc, sgg, phone_last4(phone))


@app.route('/item')
def item_page():
    uid = session.get('uid')
    if not uid or uid not in _store:
        return redirect(url_for('home'))

    s    = _store[uid]['sess']
    info = s._get_current_item_info()
    item = info['item']
    ver  = s.version
    v    = 'moca_k' if ver == 'MoCA-K' else 'k_moca'

    # 항목별 추가 데이터
    if item == 'naming':
        from version_manager import get_version_config
        animals_cfg = get_version_config(ver)['animals']
        info['animals'] = [
            {'key': a['key'], 'label': a['label'], 'stt_key': f'animal{i+1}_stt'}
            for i, a in enumerate(animals_cfg)
        ]
    elif item == 'clapping':
        info['clap_audio_seq'] = f'/audio/clap_seq_{v}.mp3'
    elif item == 'sentence_repeat':
        info['sentences_audio'] = [
            f'/audio/repeat1_{v}.mp3',
            f'/audio/repeat2_{v}.mp3',
        ]
    elif item == 'abstraction':
        info['pairs_audio'] = [
            f'/audio/abstract1_{v}.mp3',
            f'/audio/abstract2_{v}.mp3',
        ]
    elif item == 'orientation':
        info['sub_questions'] = [
            {'key': 'year',    'label': '몇 년도인가요?',          'audio': '/audio/orientation_year.mp3'},
            {'key': 'month',   'label': '몇 월인가요?',            'audio': '/audio/orientation_month.mp3'},
            {'key': 'day',     'label': '며칠인가요?',             'audio': '/audio/orientation_date.mp3'},
            {'key': 'weekday', 'label': '무슨 요일인가요?',         'audio': '/audio/orientation_day.mp3'},
            {'key': 'place',   'label': '지금 계신 동네 이름은?',   'audio': '/audio/orientation_place_inst.mp3'},
            {'key': 'sigungu', 'label': '지금 계신 시군구는?',      'audio': '/audio/orientation_city_inst.mp3'},
        ]
    elif item == 'memory_immediate':
        info['trial2_audio'] = [f'/audio/memory_inst2.mp3'] + [f'/audio/{v}_word{i}.mp3' for i in range(1, 6)]

    from session_manager import ITEM_SEQUENCE
    progress_pct = int((s.item_index / len(ITEM_SEQUENCE)) * 100)

    return render_template('item.html',
        item         = info,
        item_type    = ITEM_TYPE.get(item, 'voice'),
        title        = ITEM_TITLES.get(item, item),
        tts_urls     = _tts_urls(item, ver),
        version      = ver,
        progress_pct = progress_pct,
        current_step = s.item_index + 1,
        total_steps  = len(ITEM_SEQUENCE),
    )


@app.route('/submit', methods=['POST'])
def submit():
    uid = session.get('uid')
    if not uid or uid not in _store:
        return jsonify({'error': 'no session'}), 400

    entry = _store[uid]
    s     = entry['sess']
    raw   = entry['raw']
    data  = request.get_json(silent=True) or {}
    data  = data.get('response', data)   # JS: JSON.stringify({response: App.responses}) 언래핑
    item  = s.current_item

    _store_response(raw, item, data)

    result = s.next_item(response=data)

    if result.get('status') == 'waiting':
        return jsonify({'next': 'waiting', 'wait_seconds': result['wait_seconds']})
    elif result.get('status') == 'completed':
        if session.get('is_new_member'):
            return jsonify({'next': 'final-result'})
        else:
            return jsonify({'next': 'result'})
    else:
        return jsonify({'next': 'item'})


@app.route('/waiting')
def waiting_page():
    uid = session.get('uid')
    if not uid or uid not in _store:
        return redirect(url_for('home'))
    wait = int(_store[uid]['sess']._check_delayed_recall_wait())
    return render_template('waiting.html', wait_seconds=wait)


@app.route('/continue_wait', methods=['POST'])
def continue_wait():
    uid = session.get('uid')
    if not uid or uid not in _store:
        return jsonify({'error': 'no session'}), 400
    s = _store[uid]['sess']

    remaining = s._check_delayed_recall_wait()
    if remaining > 0:
        return jsonify({'next': 'waiting', 'wait_seconds': int(remaining)})

    result = s.next_item()
    if result.get('status') == 'completed':
        if session.get('is_new_member'):
            return jsonify({'next': 'final-result'})
        else:
            return jsonify({'next': 'result'})
    return jsonify({'next': 'item'})


@app.route('/skip_to_delayed')
def skip_to_delayed():
    """시연용: 5분 대기 건너뛰고 즉시 지연회상으로 이동"""
    uid = session.get('uid')
    if not uid or uid not in _store:
        return redirect(url_for('home'))
    s = _store[uid]['sess']
    from session_manager import ITEM_SEQUENCE, SessionState
    dr_idx = ITEM_SEQUENCE.index('delayed_recall')
    s.item_index   = dr_idx
    s.current_item = 'delayed_recall'
    s.state        = SessionState.IN_PROGRESS
    return redirect(url_for('item_page'))


@app.route('/result')
def result_page():
    cognitive = _get_cognitive_result()
    if cognitive is None:
        return redirect(url_for('home'))

    return render_template('result.html',
        score   = cognitive["score"],
        version = cognitive["version"],
        participant = cognitive["participant"],
    )


@app.route('/start_new_member_flow')
def start_new_member_flow():
    session['is_new_member'] = True
    return redirect(url_for('gait_page'))


@app.route('/physical-to-cognitive')
def physical_to_cognitive_page():
    return render_template('physical_to_cognitive.html')


@app.route('/final-result')
def final_result():
    panel = request.args.get('panel', 'summary')
    cognitive = _get_cognitive_result()
    gait_result = _get_gait_result()
    care_type = _classify_care_type(cognitive, gait_result)
    
    # Clear is_new_member flag as they have successfully completed the flow and viewed the final result
    session.pop('is_new_member', None)
    
    return render_template(
        'final_result.html',
        panel=panel,
        cognitive=cognitive,
        gait=gait_result,
        care_type=care_type,
    )


@app.route('/care-type')
def care_type_api():
    cognitive = _get_cognitive_result()
    gait_result = _get_gait_result()
    return jsonify(_classify_care_type(cognitive, gait_result))


@app.route('/api/mobile/moca/score', methods=['POST'])
def mobile_moca_score_api():
    data = request.get_json(silent=True) or {}
    phone = data.get("member_phone") or data.get("phone") or ""
    education_level = data.get("education_level") or "high"
    version = data.get("version") or "MoCA-K"
    loc = (data.get("location") or "").strip()
    sgg = (data.get("sigungu") or "").strip()

    try:
        member_id, edu, member_code, level = _ensure_member_from_phone(phone, education_level)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    raw = _empty_raw()
    raw.update(data.get("raw") or {})
    responses = data.get("responses") or {}
    for item, response in responses.items():
        if isinstance(response, dict):
            _store_response(raw, item, response)

    uid = f"{member_code}_mobile_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    assessment_id = create_assessment(uid, member_id, version, loc, sgg)
    score = _compute_score(
        raw,
        {"location": loc, "sigungu": sgg},
        SimpleNamespace(education_years=edu, version=version),
    )
    complete_assessment(assessment_id, raw, score)
    session["assessment_id"] = assessment_id
    return jsonify({
        "ok": True,
        "member_id": member_id,
        "member_code": member_code,
        "assessment_id": assessment_id,
        "version": version,
        "score": score,
        "redirect_url": url_for("final_result", panel="cognitive"),
    })


@app.route('/api/mobile/remember-login', methods=['POST'])
def api_mobile_remember_login():
    data = request.get_json(silent=True) or {}
    phone = normalize_phone(data.get("member_phone") or data.get("phone") or "")
    if len(phone) < 9:
        return jsonify({"ok": False, "error": "phone_required"}), 400

    member = find_member_by_phone(phone)
    if not member:
        return jsonify({"ok": False, "error": "member_not_found"}), 404

    _restore_member_session(member, data.get("education_level") or member.get("education_level"))
    return jsonify({
        "ok": True,
        "member_id": member["id"],
        "member_code": member["name"],
        "redirect_url": url_for("main_home_page"),
    })


@app.route('/guardian/login')
def guardian_login_page():
    return render_template('guardian_login.html')


@app.route('/guardian/send-code', methods=['POST'])
def guardian_send_code():
    data = request.get_json(silent=True) or {}
    member_phone = normalize_phone(data.get("member_phone") or data.get("phone") or "")
    member = find_member_by_phone(member_phone)
    if not member:
        return jsonify({"ok": False, "error": "등록된 사용자 번호를 찾지 못했습니다."}), 404

    code = f"{secrets.randbelow(1000000):06d}"
    expires_at = datetime.now() + timedelta(minutes=5)
    session["guardian_verification"] = {
        "member_phone_hash": phone_hash(member_phone),
        "member_id": member["id"],
        "code": code,
        "expires_at": expires_at.isoformat(),
        "verified": False,
    }
    return jsonify({
        "ok": True,
        "message": "인증번호가 발급되었습니다.",
        "demo_code": code,
        "expires_in": 300,
    })


@app.route('/guardian/verify-code', methods=['POST'])
def guardian_verify_code():
    data = request.get_json(silent=True) or {}
    member_phone = normalize_phone(data.get("member_phone") or data.get("phone") or "")
    code = (data.get("code") or "").strip()
    saved = session.get("guardian_verification") or {}
    if not saved or saved.get("member_phone_hash") != phone_hash(member_phone):
        return jsonify({"ok": False, "error": "인증번호를 다시 발급해 주세요."}), 400
    try:
        expires_at = datetime.fromisoformat(saved.get("expires_at", ""))
    except ValueError:
        return jsonify({"ok": False, "error": "인증번호를 다시 발급해 주세요."}), 400
    if datetime.now() > expires_at:
        return jsonify({"ok": False, "error": "인증번호가 만료되었습니다."}), 400
    if not hmac.compare_digest(saved.get("code", ""), code):
        return jsonify({"ok": False, "error": "인증번호가 맞지 않습니다."}), 400

    saved["verified"] = True
    session["guardian_verification"] = saved
    session["guardian_member_id"] = saved["member_id"]
    guardian_id = get_or_create_guardian(member_phone, name="보호자")
    session["guardian_id"] = guardian_id
    link_guardian_member(guardian_id, saved["member_id"])
    return jsonify({
        "ok": True,
        "message": "보호자 인증이 완료되었습니다.",
        "redirect_url": url_for("guardian_page"),
    })


def _resolve_guardian_member(parent_name="", member_phone=""):
    member_id = _current_member_id()
    if member_id:
        return member_id
    member_id = session.get("guardian_member_id")
    if member_id:
        return member_id
    member = find_member_by_phone(member_phone)
    if member:
        return member["id"]
    member = find_member_by_code_or_name(parent_name)
    if member:
        return member["id"]
    return None


@app.route('/guardian')
def guardian_page():
    parent_name = (request.args.get('parent_name') or request.args.get('parentName') or '').strip()
    member_phone = request.args.get('member_phone') or request.args.get('guardian_phone') or request.args.get('guardianPhone') or ''
    guardian_id = session.get('guardian_id')
    member_id = _resolve_guardian_member(parent_name, member_phone)
    verification = session.get("guardian_verification") or {}
    if member_phone and (
        not verification.get("verified")
        or verification.get("member_phone_hash") != phone_hash(member_phone)
    ):
        return redirect(url_for("guardian_login_page"))
    if member_id:
        session["guardian_member_id"] = member_id

    if member_phone:
        try:
            guardian_id = get_or_create_guardian(member_phone, name="보호자")
            session['guardian_id'] = guardian_id
            if member_id:
                link_guardian_member(guardian_id, member_id)
        except ValueError:
            guardian_id = session.get('guardian_id')

    dashboard = get_guardian_dashboard(member_id=member_id, limit=5)
    return render_template('guardian.html', dashboard=dashboard)


@app.route('/guardian/cheer', methods=['POST'])
def guardian_cheer():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    member_id = _current_member_id() or session.get("guardian_member_id")
    if not member_id:
        dashboard = get_guardian_dashboard(limit=1)
        member = dashboard.get("member")
        member_id = member.get("id") if member else None
    if not member_id:
        return jsonify({"ok": False, "error": "member_not_found"}), 400

    cheer = save_guardian_cheer(member_id, message, guardian_id=session.get('guardian_id'))
    cheers = get_guardian_cheers(member_id, limit=5)
    return jsonify({"ok": True, "message": cheer["message"], "count": len(cheers), "cheers": cheers})


@app.route('/assistant/profile', methods=['GET', 'POST'])
def assistant_profile_api():
    member_id = _current_member_id()
    if not member_id:
        return jsonify({"ok": False, "error": "member_not_found"}), 400
    if request.method == 'GET':
        return jsonify({"ok": True, "profile": get_or_create_assistant_profile(member_id)})

    data = request.get_json(silent=True) or {}
    profile = update_assistant_profile(
        member_id,
        voice_rate=data.get("voice_rate"),
        tts_volume=data.get("tts_volume"),
        text_scale=data.get("text_scale"),
        high_contrast=1 if data.get("high_contrast") else 0,
        reduced_motion=1 if data.get("reduced_motion") else 0,
        situation_json=data.get("situation") or {},
    )
    return jsonify({"ok": True, "profile": profile})


@app.route('/assistant/context')
def assistant_context_api():
    member_id = _current_member_id()
    if not member_id:
        return jsonify({"ok": False, "error": "member_not_found"}), 400
    return jsonify({"ok": True, "context": get_member_context_bundle(member_id)})


@app.route('/assistant/rag/search')
def assistant_rag_search_api():
    query = request.args.get("q") or ""
    return jsonify({"ok": True, "query": query, "results": retrieve_knowledge(query, top_k=5)})


@app.route('/assistant/chat', methods=['POST'])
def assistant_chat_api():
    member_id = _current_member_id()
    if not member_id:
        return jsonify({"ok": False, "error": "member_not_found"}), 400
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "empty_message"}), 400
    context = get_member_context_bundle(member_id) or {}
    knowledge = retrieve_knowledge(user_message, top_k=4)
    message_context = {**context, "retrieved_knowledge": knowledge}
    save_assistant_message(member_id, "user", user_message, context=message_context)
    reply_source = "local"
    if _pengteu_local_answer_ready(user_message, knowledge):
        reply = _basic_pengteu_reply(user_message, context, knowledge=knowledge)
    else:
        reply = _openai_pengteu_fallback(user_message, context, knowledge=knowledge)
        if reply:
            reply_source = "openai_fallback"
        else:
            reply_source = "local_fallback"
            reply = _basic_pengteu_reply(user_message, context, knowledge=knowledge)
    reply = _clean_pengteu_reply(reply, user_message)
    save_assistant_message(member_id, "assistant", reply, context=message_context)
    return jsonify({
        "ok": True,
        "reply": reply,
        "reply_source": reply_source,
        "context": context,
        "knowledge": knowledge,
    })


@app.route('/exercise/sensor/calibration', methods=['POST'])
def exercise_sensor_calibration_api():
    member_id = _current_member_id()
    if not member_id:
        return jsonify({"ok": False, "error": "member_not_found"}), 400
    data = request.get_json(silent=True) or {}
    exercise_type = data.get("exercise_type") or data.get("type") or "default"
    calibration = data.get("calibration") or data
    save_sensor_calibration(member_id, exercise_type, calibration)
    return jsonify({"ok": True})


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(os.path.join(app.root_path, 'assets', 'tts'), filename)


# ── 응답 저장 ──────────────────────────────────

def _store_response(raw, item, data):
    if item == 'trail_making':
        raw['trail_touch_points'] = data.get('points', [])
        raw['canvas_width']  = data.get('width', 512)
        raw['canvas_height'] = data.get('height', 512)

    elif item == 'cube':
        res = _score_canvas(data.get('image', ''), 'cube')
        if res:
            raw['cube_score'] = res.get('score', 0)

    elif item == 'clock':
        res = _score_canvas(data.get('image', ''), 'clock')
        if res:
            raw['clock_contour'] = res.get('contour', {}).get('score', 0)
            raw['clock_numbers'] = res.get('numbers', {}).get('score', 0)
            raw['clock_hands']   = res.get('hands',   {}).get('score', 0)

    elif item == 'naming':
        raw['animal1_stt'] = data.get('animal1_stt', '')
        raw['animal2_stt'] = data.get('animal2_stt', '')
        raw['animal3_stt'] = data.get('animal3_stt', '')

    elif item == 'memory_immediate':
        raw['immediate1_stt'] = data.get('trial1_stt', '')
        raw['immediate2_stt'] = data.get('trial2_stt', '')

    elif item == 'forward_digits':
        raw['forward_stt'] = data.get('stt', '')

    elif item == 'backward_digits':
        raw['backward_stt'] = data.get('stt', '')

    elif item == 'clapping':
        raw['tapped_indices'] = data.get('tapped_indices', [])

    elif item == 'serial_7':
        raw['serial7_stt'] = data.get('stt', '')

    elif item == 'sentence_repeat':
        raw['sentence1_stt'] = data.get('stt1', '')
        raw['sentence2_stt'] = data.get('stt2', '')

    elif item == 'verbal_fluency':
        raw['fluency_stt'] = data.get('stt', '')

    elif item == 'abstraction':
        raw['abstraction_pair1_stt'] = data.get('stt1', '')
        raw['abstraction_pair2_stt'] = data.get('stt2', '')

    elif item == 'delayed_recall':
        raw['delayed_recall_stt'] = data.get('stt', '')

    elif item == 'orientation':
        for k in ('year', 'month', 'day', 'weekday', 'place', 'sigungu'):
            raw[f'{k}_stt'] = data.get(k, '')


def _score_canvas(img_b64, kind):
    try:
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_data = base64.b64decode(img_b64)
        nparr    = np.frombuffer(img_data, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        _load_cnn_models()  # 최초 1회만 실제 로드

        if kind == 'cube':
            if _cnn_cube is not None:
                from cube_cnn_inference_v2 import score_cube_final
                r = score_cube_final(img, model=_cnn_cube[0], device=_cnn_cube[1])
                return {'score': r['total']}
            from cube import score_cube
            return score_cube(img)

        elif kind == 'clock':
            if _cnn_clock is not None:
                from clock_cnn_inference import score_clock_final
                return score_clock_final(img, models=_cnn_clock, device=_cnn_clock_dev)
            from clock import score_clock
            return score_clock(img)

    except Exception as e:
        print(f'[캔버스 채점 오류] {kind}: {e}')
    return None


def _compute_score(raw, entry, s):
    from total_scorer import score_total
    now = datetime.now()
    weekday_map = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

    location_key = {
        '장소':  entry.get('location', ''),
        '시군구': entry.get('sigungu', ''),
    }
    try:
        result = score_total(
            trail_touch_points = raw['trail_touch_points'],
            canvas_width       = raw['canvas_width'],
            canvas_height      = raw['canvas_height'],
            cube_score         = raw['cube_score'],
            clock_contour      = raw['clock_contour'],
            clock_numbers      = raw['clock_numbers'],
            clock_hands        = raw['clock_hands'],
            naming_stts        = [raw['animal1_stt'], raw['animal2_stt'], raw['animal3_stt']],
            immediate1_stt     = raw['immediate1_stt'],
            immediate2_stt     = raw['immediate2_stt'],
            delayed_recall_stt = raw['delayed_recall_stt'],
            forward_stt        = raw['forward_stt'],
            backward_stt       = raw['backward_stt'],
            tapped_indices     = raw['tapped_indices'],
            serial7_stt        = raw['serial7_stt'],
            sentence1_stt      = raw['sentence1_stt'],
            sentence2_stt      = raw['sentence2_stt'],
            fluency_stt        = raw['fluency_stt'],
            abstraction_pair1_stt = raw['abstraction_pair1_stt'],
            abstraction_pair2_stt = raw['abstraction_pair2_stt'],
            year_stt           = raw['year_stt'],
            month_stt          = raw['month_stt'],
            day_stt            = raw['day_stt'],
            weekday_stt        = raw['weekday_stt'],
            place_stt          = raw['place_stt'],
            sigungu_stt        = raw['sigungu_stt'],
            location_key       = location_key,
            education_years    = s.education_years,
            version            = s.version,
        )
        return result
    except Exception as e:
        print(f'[총점 계산 오류] {e}')
        return {'final_score': 0, 'raw_score': 0, 'sections': {}, 'mci': {'label': 1, 'interpretation': '오류'}, 'education_correction': 0}


if __name__ == '__main__':
    os.makedirs('assets/tts', exist_ok=True)
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
