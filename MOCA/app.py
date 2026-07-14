"""
MoCA-K 시연용 Flask 웹앱
폰 브라우저에서 접속: http://<서버IP>:5000
"""

import os, base64, json, hmac, secrets
from datetime import datetime, timedelta
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, session, send_from_directory, redirect, url_for
import numpy as np
import cv2
from database import (
    EDUCATION_LEVELS,
    complete_assessment,
    create_assessment,
    education_label,
    get_exercise_summary,
    get_latest_assessment,
    get_latest_physical_result,
    get_member,
    get_recent_assessment_summaries,
    get_or_create_member,
    init_db,
    normalize_phone,
    phone_hash,
    phone_last4,
    save_exercise_record,
    save_physical_result,
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
    if request.endpoint in ('login',):
        return None
    if session.get('access_granted') is True:
        return None
    if request.path == '/favicon.ico':
        return '', 204
    return redirect(url_for('login', next=request.path))

# 메모리 세션 저장소 (시연용)
_store = {}  # uid → { 'sess': MoCASession, 'raw': dict, 'location': str, 'sigungu': str }

# ── CNN 모델 (학습 완료 시 자동 로드, 없으면 룰베이스 폴백) ──
_cnn_cube        = None   # (model, device) from cube_cnn_inference_v2.load_model
_cnn_clock       = None   # dict: {'deepc': model, 'deeph': model, 'deepn': model}
_cnn_clock_dev   = None   # torch.device
_gait_model      = None
_gait_metadata   = None

GAIT_FEATURES = [
    "v_amp_pool_median",
    "ml_amp_pool_iqr",
    "base_v_stride_regularity",
    "roll_amp_pool_iqr",
]


def _load_gait_model():
    """Load the nested Youden gait model lazily for the physical assessment page."""
    global _gait_model, _gait_metadata
    if _gait_model is not None:
        return _gait_model, _gait_metadata

    model_path = os.path.join(app.root_path, "models", "gait_nested_youden.joblib")
    metadata_path = os.path.join(app.root_path, "models", "gait_nested_youden_metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            _gait_metadata = json.load(f)

    if os.path.exists(model_path):
        try:
            import joblib
            _gait_model = joblib.load(model_path)
        except Exception as e:
            print(f"[gait model load error] {e}")
            _gait_model = None

    return _gait_model, _gait_metadata


def _gait_model_summary():
    model, metadata = _load_gait_model()
    meta = metadata or {}
    return {
        "available": model is not None,
        "threshold_strategy": meta.get("threshold_strategy", "nested_inner_oof_youden"),
        "threshold": meta.get("threshold"),
        "n_subjects": meta.get("n_subjects"),
        "features": meta.get("features", GAIT_FEATURES),
        "metrics": meta.get("cv_metrics", {}),
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


def _gait_feature_insights(features):
    checks = [
        {
            "key": "v_amp_pool_median",
            "label": "수직 보행 활력",
            "value": _safe_float(features.get("v_amp_pool_median")),
            "unit": "",
            "problem": "수직 움직임이 작아 보폭/추진력이 약한 패턴",
            "ok": "수직 움직임이 비교적 안정적",
            "risk_when": "low",
            "cut": 0.10,
        },
        {
            "key": "ml_amp_pool_iqr",
            "label": "좌우 흔들림",
            "value": _safe_float(features.get("ml_amp_pool_iqr")),
            "unit": "",
            "problem": "좌우 흔들림 변동성이 커 균형 안정성 확인 필요",
            "ok": "좌우 흔들림 변동성이 낮은 편",
            "risk_when": "high",
            "cut": 0.08,
        },
        {
            "key": "base_v_stride_regularity",
            "label": "보행 리듬 규칙성",
            "value": _safe_float(features.get("base_v_stride_regularity")),
            "unit": "",
            "problem": "걸음 리듬이 불규칙한 패턴",
            "ok": "걸음 리듬이 비교적 규칙적",
            "risk_when": "low",
            "cut": 0.72,
        },
        {
            "key": "roll_amp_pool_iqr",
            "label": "몸통 회전 안정성",
            "value": _safe_float(features.get("roll_amp_pool_iqr")),
            "unit": "deg/s",
            "problem": "몸통 회전 변동성이 커 자세 안정성 확인 필요",
            "ok": "몸통 회전 변동성이 낮은 편",
            "risk_when": "high",
            "cut": 18.0,
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
                    "v_amp_pool_median": "수직 보행 활력",
                    "ml_amp_pool_iqr": "좌우 흔들림",
                    "base_v_stride_regularity": "보행 리듬",
                    "roll_amp_pool_iqr": "몸통 회전",
                }.get(name, name),
                "value": float(features[name]),
                "contribution": float(coef * val),
            }
            for name, coef, val in zip(names, coefs, transformed)
        ]
    except Exception:
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
    vertical = _safe_float(features.get("v_amp_pool_median"))
    lateral = _safe_float(features.get("ml_amp_pool_iqr"))
    regularity = _safe_float(features.get("base_v_stride_regularity"))
    roll = _safe_float(features.get("roll_amp_pool_iqr"))

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
        "name": "유지형",
        "title": "A 유형",
        "summary": "인지기능과 신체기능이 모두 양호한 상태입니다.",
        "focus": "현재 상태 유지",
        "cognitive_status": "양호",
        "physical_status": "양호",
    },
    (0, 1): {
        "code": "B",
        "name": "신체관리형",
        "title": "B 유형",
        "summary": "인지기능은 양호하지만 신체기능 관리가 필요한 상태입니다.",
        "focus": "보행 및 근력 관리",
        "cognitive_status": "양호",
        "physical_status": "저하",
    },
    (1, 0): {
        "code": "C",
        "name": "통합관리형",
        "title": "C 유형",
        "summary": "신체기능은 양호하지만 인지기능 변화 확인과 통합 관리가 필요한 상태입니다.",
        "focus": "인지 훈련 중심 통합 관리",
        "cognitive_status": "저하",
        "physical_status": "양호",
    },
    (1, 1): {
        "code": "D",
        "name": "인지관리형",
        "title": "D 유형",
        "summary": "인지기능과 신체기능 모두 세심한 관리가 필요한 상태입니다.",
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


def _exercise_mock_data():
    return {
        'user': {
            'name': '사용자',
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
    'clapping':         '손뼉치기',
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
        'orientation':      ['orientation_date_inst.mp3'],
    }
    return [f'/audio/{f}' for f in mp.get(item_name, [])]


# ── 라우트 ────────────────────────────────────

@app.route('/')
def home():
    return render_template(
        'home.html',
        education_levels=EDUCATION_LEVELS,
        error=request.args.get('error', ''),
        assessment_phase=request.args.get('phase', ''),
        profile_ready=bool(session.get('member_id')),
        registered=request.args.get('registered', ''),
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


@app.route('/report/detail')
def report_detail_page():
    return render_template('report_detail.html', exercise=_personal_exercise_data(), back_url='/exercise/complete')


@app.route('/mypage')
def mypage_page():
    return render_template('report_detail.html', exercise=_personal_exercise_data(), back_url='/?registered=1')


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


@app.route('/gait/predict', methods=['POST'])
def gait_predict():
    model_artifact, metadata = _load_gait_model()
    if model_artifact is None:
        return jsonify({'ok': False, 'error': '보행 평가 모델을 불러오지 못했습니다.'}), 503

    data = request.get_json(silent=True) or {}
    values = []
    try:
        for feature in model_artifact.get('features', GAIT_FEATURES):
            values.append(float(data[feature]))
    except (KeyError, TypeError, ValueError):
        return jsonify({'ok': False, 'error': '4개 보행 feature 값을 모두 숫자로 입력해 주세요.'}), 400

    import pandas as pd
    frame = pd.DataFrame([values], columns=model_artifact.get('features', GAIT_FEATURES))
    probability = float(model_artifact['pipeline'].predict_proba(frame)[:, 1][0])
    threshold = float(model_artifact.get('threshold', 0.5))
    prediction = int(probability >= threshold)
    features = dict(zip(model_artifact.get('features', GAIT_FEATURES), values))
    gait_result = {
        'probability': probability,
        'threshold': threshold,
        'prediction': prediction,
        'label': '이동기능 저하 가능성' if prediction else '이동기능 정상 범위 가능성',
        'threshold_strategy': model_artifact.get('threshold_strategy', 'nested_inner_oof_youden'),
        'features': features,
        'insights': _gait_feature_insights(features),
        'explainability': _gait_explainability(model_artifact, features),
        'visual': _gait_visual_profile(features),
        'window': data.get('_window') or {},
    }
    session['gait_result'] = gait_result

    return jsonify({
        'ok': True,
        'probability': probability,
        'threshold': threshold,
        'prediction': prediction,
        'label': '이동기능 저하 가능성' if prediction else '이동기능 정상 범위 가능성',
        'threshold_strategy': model_artifact.get('threshold_strategy', 'nested_inner_oof_youden'),
        'redirect_url': url_for('gait_avatar_page'),
    })


@app.route('/gait/avatar')
def gait_avatar_page():
    gait_result = session.get('gait_result')
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
        member_id, edu, member_code = get_or_create_member(phone, education_level)
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

    return redirect(url_for('home', registered='1'))


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
        member_id, edu, member_code = get_or_create_member(phone, education_level)
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
            {'key': 'year',    'label': '몇 년도인가요?',          'audio': '/audio/orientation_date_inst.mp3'},
            {'key': 'month',   'label': '몇 월인가요?',            'audio': None},
            {'key': 'day',     'label': '며칠인가요?',             'audio': None},
            {'key': 'weekday', 'label': '무슨 요일인가요?',         'audio': None},
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


@app.route('/final-result')
def final_result():
    panel = request.args.get('panel', 'summary')
    cognitive = _get_cognitive_result()
    gait_result = session.get('gait_result')
    care_type = _classify_care_type(cognitive, gait_result)
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
    gait_result = session.get('gait_result')
    return jsonify(_classify_care_type(cognitive, gait_result))


@app.route('/guardian')
def guardian_page():
    recent_assessments = get_recent_assessment_summaries(limit=5)
    gait_result = session.get('gait_result')
    cheers = session.get('guardian_cheers') or []
    dashboard = {
        "cognitive_done": any(item["is_completed"] for item in recent_assessments),
        "gait_done": gait_result is not None,
        "latest_cognitive": recent_assessments[0] if recent_assessments else None,
        "latest_gait": gait_result,
        "recent_assessments": recent_assessments,
        "cheers": cheers[-5:],
    }
    return render_template('guardian.html', dashboard=dashboard)


@app.route('/guardian/cheer', methods=['POST'])
def guardian_cheer():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        message = '오늘도 정말 잘하고 있어요. 천천히 같이 해봐요!'
    cheers = session.get('guardian_cheers') or []
    cheers.append({
        "message": message[:120],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    session['guardian_cheers'] = cheers[-20:]
    return jsonify({"ok": True, "message": message[:120], "count": len(session['guardian_cheers'])})


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
        return {'final_score': 0, 'raw_score': 0, 'sections': {}, 'mci': {'interpretation': '오류'}, 'education_correction': 0}


if __name__ == '__main__':
    os.makedirs('assets/tts', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
