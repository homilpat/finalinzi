import os
import json
import urllib.request


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

    if "기여도" in message or "판단 근거" in message or "xai" in lowered or "shap" in lowered:
        return f"{name}가 쉽게 말해드릴게요. 이번 보행 판단은 수직 추진/충격 크기, 수직 움직임 변동성, 보행 리듬 변동성 세 가지를 함께 본 결과예요. 막대는 각 지표가 이번 결과를 주의 쪽으로 설명하는지, 안정 쪽으로 설명하는지 쉽게 보여주는 자료예요.{evidence_hint}"

    if "스펙트럼" in message or "주파수" in message:
        return f"{name}가 설명해드릴게요. 스펙트럼은 허리 가속도 원신호가 시간에 따라 어떤 리듬과 주파수 패턴을 보였는지 보여주는 보조 그림이에요. 모델은 최종 3개 보행 피처로 판단하고, 스펙트럼은 그 판단을 이해하기 쉽게 돕는 시각 자료예요.{evidence_hint}"

    if knowledge:
        if "tts" in lowered or "bgm" in lowered or "mp3" in lowered or "음악" in message or "소리" in message:
            return f"{name}예요. 운동 음악은 그대로 배경음으로 두고, 제가 말할 때만 운동 안내음과 효과음을 잠깐 낮추거나 멈추게 할게요. 제 말이 끝나면 운동 화면의 음악은 원래 볼륨으로 돌아가요."
        return f"{name}예요. 관련 자료는 제가 참고만 했고, 화면에는 회원님께 필요한 내용만 짧게 말할게요. 더 자세히 알고 싶은 부분을 말해주시면 보행, 인지, 운동 기록에 맞춰 쉽게 풀어드릴게요."

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


def _safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default
