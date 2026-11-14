"""Build the missing presentation appendix charts (4, 5, 7, and 8)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis_outputs" / "presentation_appendix_pack"
MODEL_PATH = ROOT / "backend" / "models" / "gait_daily_clinical_3feat.joblib"
PILOT_PATH = (
    ROOT
    / "analysis_outputs"
    / "service_sample_qc_audit"
    / "labeled_pilot_metrics.json"
)
KNOWLEDGE_DIR = ROOT / "backend" / "pengteu" / "knowledge"

INK = "#16212C"
MUTED = "#5B6775"
BLUE = "#2F80C3"
GREEN = "#2CA25F"
ORANGE = "#E98B2A"
CYAN = "#31B7C8"
LIGHT = "#EDF3F7"


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
        linespacing=1.35,
    )


def arrow(ax, x1: float, x2: float, y: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": MUTED},
    )


def plot_sensor_calibration() -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.04,
        0.93,
        "스마트폰 센서 보정 및 해부학축 정렬",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.04,
        0.875,
        "기종별 완전 교정이 아니라 Android 공장 보정값을 기반으로 한 정지 bias 제거와 축 정렬",
        fontsize=12,
        color=MUTED,
    )

    xs = [0.04, 0.235, 0.43, 0.625, 0.82]
    titles = [
        "공장 보정 센서",
        "3초 정지 수집",
        "정지 보정",
        "V·ML·AP 변환",
        "20초 측정 저장",
    ]
    bodies = [
        "TYPE_ACCELEROMETER\nTYPE_GYROSCOPE",
        "중력 평균\n자이로 평균",
        "V축 = 중력 방향\ngyro bias 제거",
        "AP = 착용 방향\nML = V × AP",
        "100 Hz 요청\nraw·clean·해부학축",
    ]
    colors = [BLUE, CYAN, GREEN, ORANGE, BLUE]
    for index, x in enumerate(xs):
        box(ax, x, 0.48, 0.15, 0.25, titles[index], bodies[index], colors[index])
        if index < len(xs) - 1:
            arrow(ax, x + 0.153, xs[index + 1] - 0.006, 0.605)

    formulas = (
        r"$\mathrm{gyro}_{clean}=\mathrm{gyro}_{raw}-"
        r"\overline{\mathrm{gyro}_{stationary}}$"
        "\n\n"
        r"$a_V=\mathbf{a}\cdot\mathbf{e}_V/9.80665,\quad"
        r"a_{ML}=\mathbf{a}\cdot\mathbf{e}_{ML}/9.80665,\quad"
        r"a_{AP}=\mathbf{a}\cdot\mathbf{e}_{AP}/9.80665$"
    )
    ax.text(0.5, 0.30, formulas, ha="center", va="center", fontsize=15, color=INK)
    ax.text(
        0.04,
        0.08,
        "주의: 3초 정지는 중력축 정렬과 gyro zero-bias 제거 단계이며, "
        "6-position scale calibration이나 기종 간 완전한 진폭 일치는 수행하지 않음.",
        fontsize=11,
        color=MUTED,
    )
    fig.savefig(
        OUT_DIR / "04_sensor_calibration_axis_alignment.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_rag_pipeline() -> None:
    n_docs = len(list(KNOWLEDGE_DIR.glob("*.md")))
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.04,
        0.93,
        "펭트 RAG·LLM 응답 파이프라인",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.04,
        0.875,
        f"Markdown 지식문서 {n_docs}개 · 의미검색 우선 · 실패 시 로컬 TF-IDF 폴백",
        fontsize=12,
        color=MUTED,
    )

    xs = [0.035, 0.205, 0.375, 0.545, 0.715, 0.865]
    widths = [0.125, 0.125, 0.125, 0.125, 0.125, 0.10]
    titles = ["사용자 질문", "회원 컨텍스트", "RAG 검색", "규칙 판단", "LLM fallback", "응답"]
    bodies = [
        "질문·대화",
        "MoCA·보행\n운동·접근성",
        "Embedding top-4\nTF-IDF fallback",
        "로컬 템플릿\n처리 가능?",
        "Responses API\ngpt-4.1-mini",
        "안전 필터\n2~4문장",
    ]
    colors = [BLUE, CYAN, GREEN, ORANGE, BLUE, GREEN]
    for index, x in enumerate(xs):
        box(ax, x, 0.49, widths[index], 0.25, titles[index], bodies[index], colors[index])
        if index < len(xs) - 1:
            arrow(ax, x + widths[index] + 0.004, xs[index + 1] - 0.006, 0.615)

    ax.text(
        0.04,
        0.30,
        "검색 백엔드",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.17,
        0.30,
        "OpenAI text-embedding-3-small + InMemoryVectorStore"
        "  →  API 키·호출 실패 시 char_wb TF-IDF(2~5gram) + cosine similarity",
        fontsize=11.5,
        color=MUTED,
    )
    ax.text(
        0.04,
        0.20,
        "생성 설정",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.17,
        0.20,
        "Responses API · 최대 220 tokens · timeout 12초 · 오류 시 로컬 응답",
        fontsize=11.5,
        color=MUTED,
    )
    ax.text(
        0.04,
        0.10,
        "안전 원칙",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.17,
        0.10,
        "진단 확정 금지 · 검색 근거 밖 추정 금지 · MoCA 정답·힌트 제공 금지",
        fontsize=11.5,
        color=MUTED,
    )
    fig.savefig(
        OUT_DIR / "05_pengteu_rag_llm_pipeline.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_limitations() -> None:
    rows = [
        ["학습 대상 71명", "split 변동·선택 편향 가능", "독립 대상자 확대"],
        ["서비스 파일 10개", "저하군 2개인 파일럿", "전향적 앱 수집"],
        ["OR 보행저하 라벨", "낙상 진단 라벨이 아님", "실제 낙상 추적"],
        ["회전 QC 초기 기준", "기종·착용 위치별 미검증", "외부 기기 검증"],
        ["확률 calibration", "절대위험도로 해석 불가", "외부 재보정"],
    ]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.axis("off")
    ax.text(
        0.03,
        0.94,
        "한계 및 향후 검증",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    table = ax.table(
        cellText=rows,
        colLabels=["현재 한계", "해석", "향후 검증"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.27, 0.39, 0.28],
        bbox=[0.03, 0.22, 0.94, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#C7D1DA")
        cell.set_linewidth(1)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT)
            cell.get_text().set_color(INK)
    ax.text(
        0.03,
        0.10,
        "최종 해석: 낙상을 진단하는 모델이 아니라 20초 자유보행에서 "
        "운동기능 저하 가능성을 선별하여 추가 평가 또는 재측정을 안내하는 모델",
        transform=ax.transAxes,
        fontsize=12,
        color=MUTED,
    )
    fig.savefig(
        OUT_DIR / "07_limitations_future_validation.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_service_sample_correction() -> None:
    artifact = joblib.load(MODEL_PATH)
    correction = artifact["signal_correction"]
    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    before = pilot["uncorrected"]
    after = pilot["corrected"]

    fig = plt.figure(figsize=(13.33, 7.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.24)
    ax_flow = fig.add_subplot(grid[0])
    ax_bar = fig.add_subplot(grid[1])
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    ax_flow.text(
        0.03,
        0.93,
        "실사용 보행 SAMPLE의 신호 보정 및 파일럿 점검",
        fontsize=23,
        fontweight="bold",
        color=INK,
    )

    xs = [0.03, 0.23, 0.43, 0.63, 0.83]
    titles = ["앱 CSV", "신호 진폭 보정", "피처 추출", "LR 추론", "QC"]
    bodies = [
        "공장 보정\nV·ML·AP 정렬",
        f"α={correction['alpha']:.4f}\nτ={correction['tau']:.1f}",
        "10초 window 6개\nmedian·IQR",
        f"threshold\n{artifact['threshold']:.3f}",
        "회전·결측·OOD\njackknife",
    ]
    colors = [BLUE, ORANGE, CYAN, GREEN, BLUE]
    for index, x in enumerate(xs):
        box(ax_flow, x, 0.28, 0.14, 0.40, titles[index], bodies[index], colors[index])
        if index < len(xs) - 1:
            arrow(ax_flow, x + 0.145, xs[index + 1] - 0.008, 0.48)
    ax_flow.text(
        0.03,
        0.08,
        "현재 α는 정상 SAMPLE 6개의 V-band RMS 중앙값(0.09838)을 "
        "PhysioNet 정상 기준(0.19386)에 맞춘 비율이다. τ=1.0으로 시간축은 변경하지 않는다.",
        fontsize=10.5,
        color=MUTED,
    )

    labels = ["민감도", "특이도", "정확도", "F1"]
    before_values = [
        before["sensitivity"],
        before["specificity"],
        before["accuracy"],
        before["f1"],
    ]
    after_values = [
        after["sensitivity"],
        after["specificity"],
        after["accuracy"],
        after["f1"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    bars_before = ax_bar.bar(
        x - width / 2, before_values, width, color="#A7B4C2", label="보정 전"
    )
    bars_after = ax_bar.bar(
        x + width / 2, after_values, width, color=BLUE, label="보정 후"
    )
    ax_bar.bar_label(bars_before, fmt="%.2f", padding=2)
    ax_bar.bar_label(bars_after, fmt="%.2f", padding=2)
    ax_bar.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1.12),
        ylabel="파일럿 지표",
        title="서비스 SAMPLE 파일럿: 보정 전·후 비교",
    )
    ax_bar.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.legend(loc="lower right")
    fig.text(
        0.985,
        0.015,
        "n=10(발다침 2, 정상 8) · 반복 파일 독립성 불명 · "
        "보정 SAMPLE과 평가 SAMPLE이 겹쳐 기술적 파일럿으로만 해석",
        ha="right",
        fontsize=9.5,
        color=MUTED,
    )
    fig.savefig(
        OUT_DIR / "08_service_sample_correction_audit.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    setup()
    plot_sensor_calibration()
    plot_rag_pipeline()
    plot_limitations()
    plot_service_sample_correction()
    for path in [
        OUT_DIR / "04_sensor_calibration_axis_alignment.png",
        OUT_DIR / "05_pengteu_rag_llm_pipeline.png",
        OUT_DIR / "07_limitations_future_validation.png",
        OUT_DIR / "08_service_sample_correction_audit.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()
