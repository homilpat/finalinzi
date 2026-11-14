"""펭트 LangChain RAG 생성 체인.

검색된 지식(knowledge)과 사용자 컨텍스트를 프롬프트에 주입해 ChatOpenAI로
펭트 답변을 생성한다. LangChain/키가 없거나 호출이 실패하면 None을 반환하고,
호출부(app.py)가 기존 폴백 경로로 넘어간다.
"""

from __future__ import annotations

import json
import os

from .pengteu import _compact_pengteu_context


PENGTEU_SYSTEM_PROMPT = (
    "너는 고령 사용자를 돕는 펭트 AI 어시스턴트다. "
    "진단을 확정하지 말고 선별/주의 표현을 사용한다. "
    "답변은 한국어로 2~4문장, 쉽고 따뜻하게 말한다. "
    "사용자 기록(context)과 검색 지식(retrieved_knowledge) 안에서만 개인 결과를 설명하고, "
    "모르는 것은 모른다고 말한다. 검색 지식은 근거로만 참고하고, 문서 제목이나 "
    "'retrieved_knowledge' 같은 내부 표현을 화면에 그대로 노출하지 않는다. "
    "인지검사(MoCA) 진행 중에는 문항의 정답, 기억 단어, 그림·시계 정답, 힌트를 "
    "절대 알려주지 않는다. 정답을 물으면 '검사 중에는 답을 알려드릴 수 없어요'라고 "
    "부드럽게 거절하고, 대신 편하게 검사에 집중하도록 격려만 한다."
)


def _model_name() -> str:
    return os.environ.get("OPENAI_ASSISTANT_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"


def langchain_pengteu_reply(message, context, knowledge=None):
    """LangChain 체인으로 펭트 답변 생성. 실패 시 None."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return None
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=_model_name(),
            temperature=0.3,
            max_tokens=260,
            timeout=15,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PENGTEU_SYSTEM_PROMPT),
                ("human", "{payload}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        payload = json.dumps(
            {
                "question": message,
                "context": _compact_pengteu_context(context, knowledge),
            },
            ensure_ascii=False,
        )
        reply = chain.invoke({"payload": payload})
        return (reply or "").strip() or None
    except Exception as exc:
        print(f"[pengteu langchain reply error] {exc}")
        return None
