/* ── MoCA-K 앱 JS ──────────────────────────────
   TTS 재생 → 끝나면 타이머 시작
   타이머 종료 시 자동 제출
   ─────────────────────────────────────────── */

'use strict';

// ────────────────────────────────────────────
// 전역 상태
// ────────────────────────────────────────────
const App = {
  ttsUrls:       [],
  ttsIndex:      0,
  duration:      30,
  timerInterval: null,
  timerRemain:   0,
  timerTotal:    0,
  recognition:   null,
  recording:     false,
  micStopRequested: false,
  stepAdvancing: false,
  responses:     {},       // 최종 제출 데이터
  itemType:      '',
  itemName:      '',
  timerStarted:  false,
  activeAudio:   null,
  answerBuffer:  '',       // 현재 답변 턴에 누적된 음성 텍스트(침묵으로 확정하지 않음)

  // 다중 응답 (naming, sentence_repeat 등)
  multiStep:     0,
  multiAnswers:  {},
  orientationLocationRequested: false,
};

// ────────────────────────────────────────────
// 마이크 소유권 + 오디오 재생 중재 (half-duplex 상태머신)
//  - 검사 답변 STT와 펭트 STT는 동시에 켜지지 않는다 (마이크 뮤텍스).
//  - TTS/문항 음성 재생 중이면 audioBusy=true → STT가 그 소리를 주워듣지 않게 게이트.
//  - 펭트가 마이크를 가져갈 때 문항 TTS는 일시정지하되 검사 타이머는 건드리지 않는다.
// ────────────────────────────────────────────
const MicBus = {
  owner: 'none',        // 'test' | 'pengteu' | 'none'
  audioBusy: false,
  setAudioBusy(v) { this.audioBusy = Boolean(v); },
  // 펭트가 마이크를 가져감: 문항 TTS 일시정지(타이머 유지) + 검사 STT 정지
  grantToPengteu() {
    if (App.activeAudio) { try { App.activeAudio.pause(); } catch (e) {} }
    if (App.recognition && App.recording) {
      App.micStopRequested = true;
      App.recording = false;
      try { App.recognition.stop(); } catch (e) {}
    }
    this.owner = 'pengteu';
    this.audioBusy = false;  // 문항 음성을 멈췄으므로 게이트 해제 (펭트가 말하면 직후 다시 true)
  },
  // 펭트 종료 → 마이크 반납: 일시정지됐던 문항 TTS 이어재생
  releaseFromPengteu() {
    if (this.owner !== 'pengteu') return;
    this.owner = 'none';
    if (App.activeAudio && App.activeAudio.paused && !App.activeAudio.ended) {
      try { App.activeAudio.play(); } catch (e) {}
    }
  },
};
window.MicBus = MicBus;

// ────────────────────────────────────────────
// 초기화
// ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initPhysicalPages();

  const cfg = window.ITEM_CONFIG;
  if (!cfg) return;

  App.ttsUrls  = cfg.ttsUrls || [];
  App.duration = cfg.duration || 30;
  App.itemType = cfg.type;
  App.itemName = cfg.item;
  App.timerStarted = false;

  initItemUI();
  playNextTTS();
});

// ────────────────────────────────────────────
// 항목별 UI 초기화
// ────────────────────────────────────────────
const PHYSICAL_GAIT_STORAGE_KEY = 'physical_gait_result';

function startPhysicalTest(measureUrl) {
  window.location.href = measureUrl || '/physical/measure';
}

function saveGaitResult(result) {
  localStorage.setItem(PHYSICAL_GAIT_STORAGE_KEY, JSON.stringify(normalizeGaitResult(result)));
}

function completePhysicalMeasurement(result) {
  saveGaitResult(result);
  window.location.href = '/physical/result';
}

window.startPhysicalTest = startPhysicalTest;
window.saveGaitResult = saveGaitResult;
window.completePhysicalMeasurement = completePhysicalMeasurement;

function normalizeGaitResult(result) {
  const source = result || {};
  const gaitScore = clampScore(source.gaitScore ?? source.score ?? source.physicalScore ?? 75);
  const cognitiveScore = clampScore(source.cognitiveScore ?? source.mocaScore ?? source.balanceScore ?? 60);

  return {
    testType: source.testType || 'physical_gait',
    gaitScore,
    cognitiveScore,
    gaitLevel: source.gaitLevel || source.level || '활력 증진형',
    gaitType: source.gaitType || source.type || 'C유형',
    walkingSpeed: source.walkingSpeed,
    stepCount: source.stepCount,
    measuredAt: source.measuredAt || new Date().toISOString(),
  };
}

function clampScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, Math.round(number)));
}

function syncGaitResult(result) {
  if (!result || result.syncedToServer) return;
  fetch('/physical/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(result),
  })
    .then((res) => res.ok ? res.json() : null)
    .then((data) => {
      if (data && data.ok) {
        const synced = {...result, syncedToServer: true};
        localStorage.setItem(PHYSICAL_GAIT_STORAGE_KEY, JSON.stringify(synced));
      }
    })
    .catch((error) => console.warn('physical save failed:', error));
}

function loadGaitResult() {
  const raw = localStorage.getItem(PHYSICAL_GAIT_STORAGE_KEY);
  if (!raw) return null;

  try {
    return normalizeGaitResult(JSON.parse(raw));
  } catch (e) {
    console.warn('failed to load physical gait result:', e);
    return null;
  }
}

function initPhysicalPages() {
  const screen = document.querySelector('[data-physical-result-page]');
  if (!screen) return;

  const result = loadGaitResult();
  if (!result) {
    setText('physicalGaitMessage', '저장된 보행 결과가 없습니다. 검사를 먼저 시작해 주세요.');
    return;
  }

  setText('physicalGaitType', result.gaitType);
  setText('physicalGaitLevel', `(${result.gaitLevel})`);
  setText('physicalGaitMessage', getGaitMessage(result));
  setText('physicalBalanceScore', result.cognitiveScore);
  setText('physicalGaitScore', result.gaitScore);
  setText('physicalWalkingSpeed', result.walkingSpeed ?? '-');
  setText('physicalStepCount', result.stepCount ?? '-');
  setText('physicalMeasuredAt', `측정 시간 ${formatPhysicalDate(result.measuredAt)}`);

  const balanceBar = document.getElementById('physicalBalanceBar');
  if (balanceBar) balanceBar.style.width = `${clampScore(result.cognitiveScore)}%`;

  const gaitBar = document.getElementById('physicalGaitBar');
  if (gaitBar) gaitBar.style.width = `${clampScore(result.gaitScore)}%`;

  syncGaitResult(result);
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function getGaitMessage(result) {
  if (result.gaitLevel === '정상' || result.gaitLevel === '안정형') {
    return '현재 보행 안정성이 양호합니다. 지금의 활동 습관을 유지해 주세요.';
  }
  return '보행 안정감을 높이기 위한 하체 근력 강화가 필요합니다.';
}

function formatPhysicalDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || '-';
  return date.toLocaleString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function initItemUI() {
  const type = App.itemType;

  if (type === 'drawing') {
    initCanvas();
  } else if (type === 'clapping') {
    initClapping();
  } else if (type === 'naming') {
    initNaming();
  } else if (type === 'memory') {
    initMemory();
  } else if (type === 'voice_multi') {
    initVoiceMulti();
  } else if (type === 'orientation') {
    initOrientation();
  }
}

// ────────────────────────────────────────────
// TTS 순차 재생 (ttsUrls 배열을 순서대로)
// 마지막 파일 재생 완료 후 타이머 시작
// ────────────────────────────────────────────
function playNextTTS() {
  if (App.ttsIndex >= App.ttsUrls.length) {
    App.activeAudio = null;
    MicBus.setAudioBusy(false);
    onTTSComplete();
    return;
  }

  const url   = App.ttsUrls[App.ttsIndex];
  const audio = new Audio(url);
  App.activeAudio = audio;
  MicBus.setAudioBusy(true);
  audio.onended = () => {
    App.ttsIndex++;
    playNextTTS();
  };
  audio.onerror = () => {
    App.ttsIndex++;
    playNextTTS();
  };
  audio.play().catch(() => {
    App.ttsIndex++;
    playNextTTS();
  });
}

// TTS 전부 끝 → 타이머 시작
function onTTSComplete() {
  const waves = document.getElementById('ttsWaves');
  const txt = document.getElementById('ttsText');
  const replayBtn = document.getElementById('replayBtn');
  
  if (waves) waves.style.display = 'none';
  if (txt) txt.style.display = 'none';
  if (replayBtn) replayBtn.style.display = 'flex';

  if (App.itemType === 'orientation' && App.multiStep === 0) {
    App.ttsUrls = ['/audio/orientation_year.mp3'];
  }

  if (App.itemType === 'voice_multi' && App.multiStep === 0 && App.voiceMultiParts && App.voiceMultiParts.length > 0) {
    const au = new Audio(App.voiceMultiParts[0]);
    App.activeAudio = au;
    if (window.MicBus) MicBus.setAudioBusy(true);   // 자동 마이크가 이 음성을 주워듣지 않게 게이트
    const done = () => { App.activeAudio = null; if (window.MicBus) MicBus.setAudioBusy(false); };
    au.onended = done;
    au.onerror = done;
    au.play().catch(() => done());
  }

  if (App.timerStarted) {
    return;
  }
  App.timerStarted = true;

  // 손뼉치기: 시퀀스 오디오 따로 재생 + 시퀀스 애니메이션 시작
  if (App.itemType === 'clapping') {
    startClappingSequence();
  }

  startTimer(App.duration);

  // 입력 활성화
  const micBtn = document.getElementById('micBtn');
  if (micBtn) micBtn.disabled = false;
  enableSubmit();
}

window.replayTTS = function() {
  if (App.activeAudio) {
    try {
      App.activeAudio.pause();
      App.activeAudio.currentTime = 0;
    } catch(e){}
    App.activeAudio = null;
  }
  
  // 만약 마이크가 켜져 있으면 녹음 충돌 방지를 위해 꺼줌
  if (App.recording) {
    toggleMic();
  }
  
  App.ttsIndex = 0;
  
  const waves = document.getElementById('ttsWaves');
  const txt = document.getElementById('ttsText');
  const replayBtn = document.getElementById('replayBtn');
  
  if (waves) waves.style.display = 'flex';
  if (txt) txt.style.display = 'inline';
  if (replayBtn) replayBtn.style.display = 'none';
  
  playNextTTS();
};

// ────────────────────────────────────────────
// 타이머
// ────────────────────────────────────────────
function startTimer(seconds) {
  const wrap = document.getElementById('timerWrap');
  const num  = document.getElementById('timerNum');
  const arc  = document.getElementById('timerArc');
  if (!wrap) return;

  wrap.style.display = 'flex';
  App.timerTotal  = seconds;
  App.timerRemain = seconds;
  const circumference = 163.4;

  num.textContent = seconds;
  arc.style.strokeDashoffset = '0';
  arc.classList.remove('warn', 'danger');

  App.timerInterval = setInterval(() => {
    App.timerRemain--;
    num.textContent = App.timerRemain;

    const offset = circumference * (1 - App.timerRemain / App.timerTotal);
    arc.style.strokeDashoffset = offset;

    if (App.timerRemain <= 10) {
      arc.classList.add('danger');
      arc.classList.remove('warn');
    } else if (App.timerRemain <= 20) {
      arc.classList.add('warn');
    }

    if (App.timerRemain <= 0) {
      clearInterval(App.timerInterval);
      submitItem();
    }
  }, 1000);

  // 자동 마이크: 답변 차례가 시작되면(음성 문항) 문항 음성이 끝난 뒤 자동 청취.
  // → 노인이 마이크 버튼을 누를 필요 없이 듣고 말하기만 하면 된다. 그리기/손뼉은 제외.
  if (!['drawing', 'clapping'].includes(App.itemType)) {
    App.answerBuffer = '';
    setTimeout(() => autoStartMicSafe(), 200);
  }
}

function stopTimer() {
  clearInterval(App.timerInterval);
}

// ────────────────────────────────────────────
// 제출
// ────────────────────────────────────────────
function enableSubmit() {
  const btn = document.getElementById('submitBtn');
  if (btn) btn.disabled = false;
}

async function submitItem() {
  if (App.stepAdvancing) {
    setTimeout(submitItem, 300);
    return;
  }

  clearTimeout(App.answerIdleTimer);
  // 현재 답변 버퍼를 확정 저장한다(음성 "다음"·버튼·타이머 어느 경로로 왔든 동일).
  const _buf = (App.answerBuffer || '').trim();
  if (_buf) {
    const _k = getCurrentSTTKey();
    App.responses[_k] = _buf;
    App.multiAnswers[_k] = _buf;
  }
  App.answerBuffer = '';

  // 진행 시 마이크를 잠깐 끈다: 다음 단계 질문 음성을 마이크가 주워듣지 않게.
  // (다음 단계의 startTimer가 음성 종료 후 autoStartMicSafe로 다시 켠다.)
  stopTestMic();

  // 다중 단계 진행 도중이고 마지막 스텝이 아니면 다음 스텝으로(누적 답변을 전달).
  if (App.itemType === 'naming') {
    if (App.multiStep < (App.namingAnimals || []).length - 1) {
      advanceMultiStep(_buf);
      return;
    }
  }
  if (App.itemType === 'orientation') {
    if (App.multiStep < (App.orientQuestions || []).length - 1) {
      advanceMultiStep(_buf);
      return;
    }
  }
  if (App.itemType === 'voice_multi') {
    if (App.multiStep < (App.voiceMultiParts || []).length - 1) {
      onVoiceMultiStep(_buf);
      return;
    }
  }
  if (App.itemType === 'memory') {
    if (App.multiStep < 1) {
      onMemoryStep(_buf);
      return;
    }
  }

  if (App.recording && App.recognition) {
    App.micStopRequested = true;
    App.recording = false;
    try {
      App.recognition.stop();
    } catch (e) {
      console.warn('STT 종료 오류:', e.message);
    }
  }

  stopTimer();
  const btn = document.getElementById('submitBtn');
  if (btn) { btn.disabled = true; btn.textContent = '처리 중...'; }

  // 그리기: 캔버스 이미지 첨부
  if (App.itemType === 'drawing') {
    const canvas = document.getElementById('drawCanvas');
    App.responses.image  = canvas.toDataURL('image/png');
    App.responses.points = App.strokePoints || [];
    App.responses.width  = canvas.width;
    App.responses.height = canvas.height;
  }

  // 손뼉치기: 탭 인덱스
  if (App.itemType === 'clapping') {
    App.responses.tapped_indices = App.tappedIndices || [];
  }

  // 다중 단계 응답 병합
  Object.assign(App.responses, App.multiAnswers);

  try {
    const res = await fetch('/submit', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ response: App.responses }),
    });
    const data = await res.json();

    if (data.error) {
      alert('세션이 만료되었습니다. 처음부터 다시 시작해 주세요.');
      window.location.href = '/';
      return;
    }
    if (data.next === 'waiting') {
      sessionStorage.setItem('wait_seconds', data.wait_seconds);
      window.location.href = '/waiting';
    } else if (data.next === 'final-result') {
      window.location.href = '/final-result';
    } else if (data.next === 'result') {
      window.location.href = '/result';
    } else {
      window.location.href = '/item';
    }
  } catch (e) {
    console.error(e);
    if (btn) { btn.disabled = false; btn.textContent = '다음'; }
  }
}

// ────────────────────────────────────────────
// Web Speech API STT
// ────────────────────────────────────────────
// 진행/명령 판별 — 답변 끝에 붙는 "다음/넘어가/완료/끝" 류를 진행 명령으로 인식.
// (검사 답변에는 거의 안 나오는 표현이라 오판 위험 낮음. 명령은 답변에서 잘라낸다.)
const NAV_CMD_RE = /\s*(다음\s*(문제|이요|이오|으로|요)?|넘어\s*가(요|주세요)?|넘겨\s*(줘|주세요)?|다\s*했어(요)?|끝났어(요)?|완료)\s*$/;
function extractNavCommand(txt) {
  const m = txt.match(NAV_CMD_RE);
  if (m) return { isNav: true, answer: txt.slice(0, m.index).trim() };
  return { isNav: false, answer: txt };
}

// 실제 발화가 있을 때만 답변 버퍼에 누적하고 현재 문항 key에 저장한다.
function appendTestAnswer(t) {
  if (!t) return;
  App.answerBuffer = (App.answerBuffer ? App.answerBuffer + ' ' : '') + t;
  const key = getCurrentSTTKey();
  App.responses[key] = App.answerBuffer;
  App.multiAnswers[key] = App.answerBuffer;
  enableSubmit();
}

function updateTranscriptDisplay(interim) {
  const el = document.getElementById('transcriptText');
  if (!el) return;
  const shown = ((App.answerBuffer || '') + ' ' + (interim || '')).trim();
  if (shown) { el.textContent = shown; el.classList.add('has-text'); }
}

// 최종 인식 조각 처리: "다음" 류면 진행, 아니면 답변으로 누적(침묵으론 아무 것도 안 함).
function handleFinalChunk(txt) {
  if (!txt || App.stepAdvancing) return;
  const { isNav, answer } = extractNavCommand(txt);
  if (answer) appendTestAnswer(answer);
  if (isNav) submitItem();   // submitItem이 버퍼 저장 + 다음 단계/제출을 처리
}

function initSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { console.warn('SpeechRecognition 미지원'); return null; }
  const r = new SR();
  r.lang = 'ko-KR';
  r.continuous = true;
  r.interimResults = true;

  r.onresult = (e) => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i];
      const txt = (res[0] && res[0].transcript) ? res[0].transcript : '';
      if (res.isFinal) handleFinalChunk(txt.trim());
      else interim += txt;
    }
    updateTranscriptDisplay(interim);
    scheduleAnswerIdle();   // 발화가 있으면 무입력 안내 타이머 리셋
  };
  r.onend = () => {
    // 침묵으로 인식 세션이 끊겨도 사용자가 멈춘 게 아니면 계속 듣는다(고민 시간 보장).
    if (App.recording && !App.micStopRequested) {
      setTimeout(() => {
        try {
          App.recognition?.start();
        } catch (e) {
          console.warn('STT 재시작 오류:', e.message);
        }
      }, 150);
      return;
    }

    App.recording = false;
    App.micStopRequested = false;
    const btn = document.getElementById('micBtn');
    if (btn) { btn.classList.remove('recording'); }
    const status = document.getElementById('micStatus');
    if (status) status.textContent = '완료';
  };
  r.onerror = (e) => { console.warn('STT 오류:', e.error); };
  return r;
}

// ── 마이크 시작/정지 (자동·수동 공용) ──
function startTestMic() {
  if (['drawing', 'clapping'].includes(App.itemType)) return;
  if (!App.recognition) App.recognition = initSpeech();
  if (!App.recognition || App.recording) return;
  // 마이크 회수: 펭트가 쓰고 있으면 멈추고 검사 STT가 마이크를 가져간다.
  if (window.PengteuAssistant && typeof window.PengteuAssistant.stop === 'function') {
    window.PengteuAssistant.stop();
  }
  MicBus.owner = 'test';
  App.micStopRequested = false;
  try {
    App.recognition.start();
  } catch (e) {
    console.warn('STT 시작 오류:', e.message);
    return;
  }
  App.recording = true;
  const btn = document.getElementById('micBtn');
  if (btn) btn.classList.add('recording');
  const status = document.getElementById('micStatus');
  if (status) status.textContent = '듣는 중...';
  scheduleAnswerIdle();
}

function stopTestMic() {
  clearTimeout(App.answerIdleTimer);
  if (!App.recognition || !App.recording) { App.recording = false; return; }
  App.micStopRequested = true;
  App.recording = false;
  try { App.recognition.stop(); } catch (e) {}
  const btn = document.getElementById('micBtn');
  if (btn) btn.classList.remove('recording');
  const status = document.getElementById('micStatus');
  if (status) status.textContent = '완료';
}

function toggleMic() {
  if (App.recording) stopTestMic(); else startTestMic();
}

// 문항/단계 음성이 재생 중이면 그 소리를 주워듣지 않도록, 오디오가 끝난 뒤에 마이크를 켠다.
function autoStartMicSafe(tries) {
  if (['drawing', 'clapping'].includes(App.itemType)) return;
  const busy = (window.MicBus && MicBus.audioBusy) ||
               (App.activeAudio && !App.activeAudio.paused && !App.activeAudio.ended);
  if (busy) {
    if ((tries || 0) < 30) setTimeout(() => autoStartMicSafe((tries || 0) + 1), 300);
    return;
  }
  if (!App.recording) startTestMic();
}

// ── 무입력 안내: 오래 조용하면 펭트가 "다음이라 말하세요"를 제안(세션 전체 예산) ──
const ANSWER_IDLE_MS = 7000;
const MAX_ADVANCE_NUDGE = 2;   // 세션 통틀어 최대 횟수(잔소리 방지)
function scheduleAnswerIdle() {
  if (!App.recording) return;
  if (['drawing', 'clapping'].includes(App.itemType)) return;
  clearTimeout(App.answerIdleTimer);
  App.answerIdleTimer = setTimeout(fireAdvanceNudge, ANSWER_IDLE_MS);
}
function fireAdvanceNudge() {
  if (!App.recording || App.stepAdvancing) { scheduleAnswerIdle(); return; }
  const used = parseInt(sessionStorage.getItem('pt_advNudge') || '0', 10);
  if (used >= MAX_ADVANCE_NUDGE) return;
  if (!(window.PengteuProactive && window.PengteuProactive.say)) return;
  // 안내 발화를 마이크가 주워듣지 않게 STT를 잠깐 멈췄다가, 말이 끝나면 재개한다.
  stopTestMic();
  const said = window.PengteuProactive.say("다 말씀하셨으면 '다음'이라고 말씀해 주세요. 더 생각하셔도 괜찮아요.");
  if (said) {
    sessionStorage.setItem('pt_advNudge', String(used + 1));
    const resume = () => {
      window.removeEventListener('pengteu-speaking-end', resume);
      startTestMic();
    };
    window.addEventListener('pengteu-speaking-end', resume);
  } else {
    startTestMic();
  }
}

function showAnswerCaptured(text, onDone) {
  App.stepAdvancing = true;

  const transcript = document.getElementById('transcriptText');
  const transcriptBox = document.getElementById('transcriptBox');
  const status = document.getElementById('micStatus');

  if (transcript) {
    transcript.textContent = `입력됨: ${text}`;
    transcript.classList.add('has-text');
  }
  if (transcriptBox) transcriptBox.classList.add('captured');
  if (status) status.textContent = '입력 완료';

  setTimeout(() => {
    if (transcriptBox) transcriptBox.classList.remove('captured');
    if (transcript) {
      transcript.textContent = '다음 답변을 말씀해 주세요';
      transcript.classList.remove('has-text');
    }
    if (status) status.textContent = App.recording ? '듣는 중...' : '준비';
    App.stepAdvancing = false;
    onDone();
  }, 1200);
}

function submitAfterCaptured() {
  enableSubmit();
  setTimeout(() => submitItem(), 250);
}

// 현재 단계에 맞는 응답 key
function getCurrentSTTKey() {
  const type = App.itemType;
  if (type === 'voice')  return 'stt';
  if (type === 'naming') return ['animal1_stt','animal2_stt','animal3_stt'][App.multiStep] || 'stt';
  if (type === 'voice_multi') return ['stt1','stt2'][App.multiStep] || 'stt1';
  if (type === 'orientation') {
    return ['year','month','day','weekday','place','sigungu'][App.multiStep] || 'year';
  }
  if (type === 'memory') return App.multiStep === 0 ? 'trial1_stt' : 'trial2_stt';
  return 'stt';
}

// ────────────────────────────────────────────
// 어휘력 (naming - 동물 3마리 순차)
// ────────────────────────────────────────────
function initNaming() {
  const container = document.getElementById('namingContainer');
  if (!container) return;
  App.namingAnimals = JSON.parse(container.dataset.animals || '[]');
  App.multiStep = 0;
  showAnimal(0);
}

function showAnimal(idx) {
  if (idx >= App.namingAnimals.length) { enableSubmit(); return; }
  const a   = App.namingAnimals[idx];
  const imgEl = document.getElementById('animalImg');
  const idx_el = document.getElementById('animalIndex');
  if (imgEl) imgEl.src = `/static/images/${a.key}.png`;
  if (idx_el) idx_el.textContent = idx + 1;
}

function advanceMultiStep(text) {
  stopTimer();
  
  if (App.itemType === 'naming') {
    const keys = ['animal1_stt', 'animal2_stt', 'animal3_stt'];
    App.multiAnswers[keys[App.multiStep]] = text || "";
    App.multiStep++;
    App.timerStarted = false;
    if (App.multiStep < (App.namingAnimals || []).length) {
      showAnimal(App.multiStep);
      startTimer(App.duration);
    } else {
      submitAfterCaptured();
    }
  } 
  else if (App.itemType === 'orientation') {
    const keys = ['year', 'month', 'day', 'weekday', 'place', 'sigungu'];
    App.multiAnswers[keys[App.multiStep]] = text || "";
    App.multiStep++;
    App.timerStarted = false; // 타이머 기동 락 해제
    const q = App.orientQuestions?.[App.multiStep];
    if (q) {
      const qEl  = document.getElementById('orientQuestion');
      const idxEl = document.getElementById('orientIndex');
      if (qEl)  qEl.textContent  = q.label;
      if (idxEl) idxEl.textContent = App.multiStep + 1;

      // 마이크 상태 디스플레이 갱신
      const mst = document.getElementById('micStatus');
      if (mst) mst.textContent = App.recording ? '듣는 중...' : '준비';

      if (q.audio) {
        // 문제 다시 듣기용 URL 배열 업데이트
        App.ttsUrls = [q.audio];
        App.ttsIndex = 0;

        // 배너 파형 연출 및 다시 듣기 버튼 숨김
        const waves = document.getElementById('ttsWaves');
        const txt = document.getElementById('ttsText');
        const replayBtn = document.getElementById('replayBtn');
        if (waves) waves.style.display = 'flex';
        if (txt) txt.style.display = 'inline';
        if (replayBtn) replayBtn.style.display = 'none';

        if (App.activeAudio) {
          try { App.activeAudio.pause(); } catch(e){}
        }
        const au = new Audio(q.audio);
        App.activeAudio = au;
        au.onended = () => {
          App.activeAudio = null;
          onTTSComplete();
        };
        au.onerror = () => {
          App.activeAudio = null;
          onTTSComplete();
        };
        au.play().catch(() => {
          App.activeAudio = null;
          onTTSComplete();
        });
      } else {
        startTimer(App.duration);
      }
    } else {
      submitAfterCaptured();
    }
  }
}

function onStepComplete(text) {
  showAnswerCaptured(text, () => {
    if (App.itemType === 'naming' || App.itemType === 'orientation') {
      advanceMultiStep(text);
    } else if (App.itemType === 'voice_multi') {
      onVoiceMultiStep(text);
    } else if (App.itemType === 'memory') {
      onMemoryStep(text);
    }
  });
}

// ────────────────────────────────────────────
// 기억력 즉각회상 (2회)
// ────────────────────────────────────────────
function initMemory() {
  App.multiStep = 0;
  // TTS가 단어 다 읽으면 onTTSComplete → 타이머 + 녹음 활성화
  // 타이머 종료 또는 수동 제출 시 trial1 완료 → trial2 시작
}

function onMemoryStep(text) {
  if (App.multiStep === 0) {
    App.multiAnswers.trial1_stt = text;
    showAnswerCaptured(text, () => {
      App.multiStep = 1;

      // 2회차 안내 + 단어 재생
      const lbl = document.getElementById('trialLabel');
      if (lbl) lbl.textContent = '2회차';
      const words = document.getElementById('wordsDisplay');
      if (words) words.textContent = '잘 들으세요';

      const audio2 = JSON.parse(
        document.getElementById('memoryContainer')?.dataset.trial2Audio || '[]'
      );
      let idx = 0;
      function playNext2() {
        if (idx >= audio2.length) {
          App.activeAudio = null;
          if (window.MicBus) MicBus.setAudioBusy(false);
          startTimer(App.duration);
          return;
        }
        const au = new Audio(audio2[idx]);
        App.activeAudio = au;
        au.onended = () => { idx++; playNext2(); };
        au.onerror = () => { idx++; playNext2(); };
        au.play().catch(() => { idx++; playNext2(); });
      }
      stopTimer();
      if (window.MicBus) MicBus.setAudioBusy(true);
      playNext2();
    });
  } else {
    App.multiAnswers.trial2_stt = text;
    showAnswerCaptured(text, () => submitAfterCaptured());
  }
}

// ────────────────────────────────────────────
// 따라말하기 / 추상력 (2단계 음성)
// ────────────────────────────────────────────
function initVoiceMulti() {
  App.multiStep = 0;
  App.voiceMultiParts = [];
  const container = document.getElementById('voiceMultiContainer');
  if (container) {
    App.voiceMultiParts  = JSON.parse(container.dataset.parts || '[]');
    App.voiceMultiKeys   = JSON.parse(container.dataset.keys  || '[]');
    App.voiceMultiLabels = JSON.parse(container.dataset.labels || '[]');
  }
}

function onVoiceMultiStep(text) {
  const key = App.voiceMultiKeys[App.multiStep] || `stt${App.multiStep + 1}`;
  App.multiAnswers[key] = text;
  showAnswerCaptured(text, () => {
    App.multiStep++;
    App.timerStarted = false;

    if (App.multiStep < App.voiceMultiParts.length) {
      // 다음 파트 TTS 재생
      const lbl = document.getElementById('multiStepLabel');
      if (lbl && App.voiceMultiLabels[App.multiStep]) {
        lbl.textContent = App.voiceMultiLabels[App.multiStep];
      }
      const au = new Audio(App.voiceMultiParts[App.multiStep]);
      App.activeAudio = au;
      if (window.MicBus) MicBus.setAudioBusy(true);
      const nextTurn = () => {
        App.activeAudio = null;
        if (window.MicBus) MicBus.setAudioBusy(false);
        startTimer(App.duration);
      };
      au.onended = nextTurn;
      au.onerror = nextTurn;
      stopTimer();
      App.timerStarted = true;
      au.play().catch(nextTurn);

      const mst = document.getElementById('micStatus');
      if (mst) mst.textContent = App.recording ? '듣는 중...' : '준비';
    } else {
      submitAfterCaptured();
    }
  });
}

// ────────────────────────────────────────────
// 지남력 (6단계)
// ────────────────────────────────────────────
function initOrientation() {
  App.multiStep = 0;
  App.orientationLocationRequested = false;
  const container = document.getElementById('orientationContainer');
  if (container) {
    App.orientQuestions = JSON.parse(container.dataset.questions || '[]');
  }
}

function requestOrientationLocationOnce() {
  if (App.orientationLocationRequested) return;
  App.orientationLocationRequested = true;
  if (window.AndroidBridge && typeof window.AndroidBridge.requestOrientationLocation === 'function') {
    window.AndroidBridge.requestOrientationLocation();
  } else if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition((pos) => {
      fetch('/api/orientation/location', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
        }),
      }).catch(() => {});
    }, () => {}, {enableHighAccuracy: true, timeout: 8000, maximumAge: 300000});
  }
}

window.onOrientationLocationEvent = function(event) {
  const status = document.getElementById('micStatus');
  if (!status || !event) return;
  if (event.ok) {
    status.textContent = '위치 확인 완료';
  }
};

function onOrientationStep(text) {
  const keys = ['year','month','day','weekday','place','sigungu'];
  App.multiAnswers[keys[App.multiStep]] = text;
  showAnswerCaptured(text, () => {
    App.multiStep++;

    const q = App.orientQuestions?.[App.multiStep];
    if (q) {
      const qEl  = document.getElementById('orientQuestion');
      const idxEl = document.getElementById('orientIndex');
      if (qEl)  qEl.textContent  = q.label;
      if (idxEl) idxEl.textContent = App.multiStep + 1;
      if (q.key === 'place' || q.key === 'sigungu') {
        requestOrientationLocationOnce();
      }

      if (q.audio) {
        const au = new Audio(q.audio);
        au.onerror = () => {};
        au.play().catch(() => {});
      }

      const mst = document.getElementById('micStatus');
      if (mst) mst.textContent = App.recording ? '듣는 중...' : '준비';
    } else {
      submitAfterCaptured();
    }
  });
}

// ────────────────────────────────────────────
// 손뼉치기
// ────────────────────────────────────────────
function initClapping() {
  App.tappedIndices = [];
  App.clapCurrentIdx = -1;

  // TAP 이벤트: 화면 어디든 탭
  document.getElementById('itemScreen')?.addEventListener('click', onClap);
  document.getElementById('itemScreen')?.addEventListener('touchstart', onClap, { passive: true });
}

function onClap() {
  if (App.clapCurrentIdx < 0) return;
  App.tappedIndices.push(App.clapCurrentIdx);
  const el = document.getElementById('clapCount');
  if (el) el.textContent = App.tappedIndices.length;

  const letter = document.getElementById('clapLetter');
  if (letter) {
    letter.classList.add('target-flash');
    setTimeout(() => letter.classList.remove('target-flash'), 300);
  }
}

function startClappingSequence() {
  const cfg      = window.ITEM_CONFIG;
  const sequence = cfg.sequence || [];
  const seqAudio = cfg.seqAudio;

  // 오디오 재생
  if (seqAudio) {
    const au = new Audio(seqAudio);
    au.onerror = () => {};
    au.play().catch(() => {});
  }

  // 시각 시퀀스 (1.5초 간격)
  const letter = document.getElementById('clapLetter');
  sequence.forEach((ch, i) => {
    setTimeout(() => {
      App.clapCurrentIdx = i;
      if (letter) letter.textContent = ch;
    }, i * 1500);
  });

  // 시퀀스 종료 후 탭 종료
  setTimeout(() => {
    App.clapCurrentIdx = -1;
    enableSubmit();
  }, sequence.length * 1500 + 1000);
}

// ────────────────────────────────────────────
// 캔버스 (그리기)
// ────────────────────────────────────────────
let _ctx, _drawing = false, _strokes = [], _currentStroke = [];
App.strokePoints = [];

// 길만들기 노드 위치 (trail_making.py NODE_POSITIONS와 동일) — PDF 문제지 배치 기준
const TRAIL_NODES = {
  "마": [0.30, 0.10],
  "가": [0.62, 0.14],
  "5":  [0.07, 0.37],
  "나": [0.45, 0.42],
  "2":  [0.66, 0.30],
  "1":  [0.24, 0.56],
  "라": [0.10, 0.70],
  "4":  [0.47, 0.70],
  "3":  [0.68, 0.68],
  "다": [0.18, 0.87],
};

// 점선 화살표 보조 함수
function _drawDashedArrow(ctx, x1, y1, x2, y2, nodeR) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const sx = x1 + Math.cos(angle) * nodeR;
  const sy = y1 + Math.sin(angle) * nodeR;
  const ex = x2 - Math.cos(angle) * nodeR;
  const ey = y2 - Math.sin(angle) * nodeR;
  // 점선
  ctx.beginPath();
  ctx.setLineDash([6, 5]);
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
  // 화살촉
  const al = nodeR * 0.6;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - al * Math.cos(angle - 0.4), ey - al * Math.sin(angle - 0.4));
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - al * Math.cos(angle + 0.4), ey - al * Math.sin(angle + 0.4));
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawTrailNodes(ctx, w, h) {
  const r = Math.min(w, h) * 0.055;
  ctx.save();

  // 예시 연결선: 1→가→2 (PDF 문제지 힌트와 동일)
  const hint = [["1","가"], ["가","2"]];
  hint.forEach(([a, b]) => {
    const [ax, ay] = TRAIL_NODES[a];
    const [bx, by] = TRAIL_NODES[b];
    _drawDashedArrow(ctx, ax * w, ay * h, bx * w, by * h, r);
  });

  // 노드 그리기
  Object.entries(TRAIL_NODES).forEach(([label, [rx, ry]]) => {
    const x = rx * w, y = ry * h;
    // 원
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.strokeStyle = '#1A2B3C';
    ctx.lineWidth = 2;
    ctx.stroke();
    // 노드 라벨
    ctx.fillStyle = '#1A2B3C';
    ctx.font = `bold ${Math.round(r * 0.95)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y);
    // 시작/끝 보조 라벨
    if (label === '1' || label === '마') {
      const subText = label === '1' ? '시작' : '끝';
      ctx.font = `${Math.round(r * 0.7)}px sans-serif`;
      ctx.fillStyle = '#555';
      ctx.fillText(subText, x, y + r + Math.round(r * 0.75));
    }
  });
  ctx.restore();
}

function initCanvas() {
  const canvas = document.getElementById('drawCanvas');
  if (!canvas) return;

  // 부모 너비에 맞게 캔버스 크기 설정
  const w = canvas.parentElement.clientWidth;
  canvas.width  = w;
  canvas.height = Math.min(w, window.innerHeight * 0.45);

  _ctx = canvas.getContext('2d');
  _ctx.strokeStyle = '#1A2B3C';
  _ctx.lineWidth   = 3;
  _ctx.lineCap     = 'round';
  _ctx.lineJoin    = 'round';

  // 길만들기: 노드 초기 렌더링
  if (App.itemName === 'trail_making') {
    drawTrailNodes(_ctx, canvas.width, canvas.height);
  }

  // 터치 이벤트
  canvas.addEventListener('touchstart', e => { e.preventDefault(); startDraw(e.touches[0], canvas); }, { passive: false });
  canvas.addEventListener('touchmove',  e => { e.preventDefault(); moveDraw(e.touches[0], canvas); },  { passive: false });
  canvas.addEventListener('touchend',   e => { e.preventDefault(); endDraw(); });

  // 마우스 이벤트 (PC 테스트용)
  canvas.addEventListener('mousedown', e => startDraw(e, canvas));
  canvas.addEventListener('mousemove', e => { if (_drawing) moveDraw(e, canvas); });
  canvas.addEventListener('mouseup',   () => endDraw());

  enableSubmit();
}

function getPos(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  };
}

function startDraw(e, canvas) {
  _drawing = true;
  _currentStroke = [];
  const p = getPos(e, canvas);
  _ctx.beginPath();
  _ctx.moveTo(p.x, p.y);
  _currentStroke.push([p.x, p.y]);
}

function moveDraw(e, canvas) {
  if (!_drawing) return;
  const p = getPos(e, canvas);
  _ctx.lineTo(p.x, p.y);
  _ctx.stroke();
  _currentStroke.push([p.x, p.y]);
}

function endDraw() {
  if (!_drawing) return;
  _drawing = false;
  if (_currentStroke.length) {
    _strokes.push([..._currentStroke]);
    App.strokePoints = _strokes.flat();
  }
}

function undoStroke() {
  if (!_strokes.length) return;
  _strokes.pop();
  App.strokePoints = _strokes.flat();
  redrawCanvas();
}

function clearCanvas() {
  _strokes = [];
  App.strokePoints = [];
  const canvas = document.getElementById('drawCanvas');
  if (!_ctx) return;
  _ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (App.itemName === 'trail_making') drawTrailNodes(_ctx, canvas.width, canvas.height);
}

function redrawCanvas() {
  const canvas = document.getElementById('drawCanvas');
  _ctx.clearRect(0, 0, canvas.width, canvas.height);
  // 선 먼저
  _ctx.strokeStyle = '#1A2B3C';
  _ctx.lineWidth   = 3;
  _ctx.lineCap     = 'round';
  _ctx.lineJoin    = 'round';
  _strokes.forEach(stroke => {
    _ctx.beginPath();
    stroke.forEach(([x, y], i) => {
      if (i === 0) _ctx.moveTo(x, y);
      else { _ctx.lineTo(x, y); _ctx.stroke(); }
    });
  });
  // 노드를 선 위에 덮어씌워 항상 보이게
  if (App.itemName === 'trail_making') drawTrailNodes(_ctx, canvas.width, canvas.height);
}
(() => {
  const widget = document.getElementById('pengteuWidget');
  if (!widget) return;

  const toggle = document.getElementById('pengteuToggle');
  const closeBtn = document.getElementById('pengteuClose');
  const panel = document.getElementById('pengteuPanel');
  const form = document.getElementById('pengteuForm');
  const input = document.getElementById('pengteuInput');
  const micBtn = document.getElementById('pengteuMicBtn');
  const messages = document.getElementById('pengteuMessages');
  const textScale = document.getElementById('pengteuTextScale');
  const voiceRate = document.getElementById('pengteuVoiceRate');
  const volume = document.getElementById('pengteuVolume');
  const highContrast = document.getElementById('pengteuHighContrast');
  const reducedMotion = document.getElementById('pengteuReducedMotion');
  let profile = {
    voice_rate: 0.85,
    tts_volume: 0.85,
    text_scale: 1,
    high_contrast: 0,
    reduced_motion: 0,
  };
  let pengteuRecognition = null;
  let pengteuListening = false;
  let pengteuSpeaking = false;
  let pengteuAudio = null;   // 네이티브 브릿지가 없을 때 쓰는 폴백 TTS 오디오

  function openPanel() {
    panel.hidden = false;
    toggle.setAttribute('aria-expanded', 'true');
    setTimeout(() => input && input.focus(), 0);
  }

  function closePanel() {
    panel.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
  }

  function appendMessage(role, text) {
    const node = document.createElement('div');
    node.className = `pengteu-message pengteu-message-${role}`;
    node.textContent = text;
    messages.appendChild(node);
    messages.scrollTop = messages.scrollHeight;
  }

  function applyProfile(nextProfile) {
    profile = { ...profile, ...(nextProfile || {}) };
    const scale = Number(profile.text_scale || 1);
    document.documentElement.style.setProperty('--pengteu-text-scale', String(scale));
    document.body.classList.toggle('pengteu-high-contrast', Boolean(Number(profile.high_contrast)));
    document.body.classList.toggle('pengteu-reduced-motion', Boolean(Number(profile.reduced_motion)));
    if (textScale) textScale.value = String(scale);
    if (voiceRate) voiceRate.value = String(profile.voice_rate || 0.85);
    if (volume) volume.value = String(profile.tts_volume || 0.85);
    if (highContrast) highContrast.checked = Boolean(Number(profile.high_contrast));
    if (reducedMotion) reducedMotion.checked = Boolean(Number(profile.reduced_motion));
  }

  async function loadProfile() {
    try {
      const res = await fetch('/assistant/profile');
      const data = await res.json();
      if (data.ok) applyProfile(data.profile);
    } catch (err) {
      console.warn('[pengteu profile]', err);
    }
  }

  async function saveProfile() {
    const payload = {
      voice_rate: Number(voiceRate.value),
      tts_volume: Number(volume.value),
      text_scale: Number(textScale.value),
      high_contrast: highContrast.checked,
      reduced_motion: reducedMotion.checked,
    };
    applyProfile(payload);
    try {
      const res = await fetch('/assistant/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.ok) applyProfile(data.profile);
    } catch (err) {
      console.warn('[pengteu profile save]', err);
    }
  }

  async function persistProfile(nextProfile) {
    const payload = {
      voice_rate: Number(nextProfile.voice_rate ?? profile.voice_rate ?? 0.85),
      tts_volume: Number(nextProfile.tts_volume ?? profile.tts_volume ?? 0.85),
      text_scale: Number(nextProfile.text_scale ?? profile.text_scale ?? 1),
      high_contrast: Boolean(Number(nextProfile.high_contrast ?? profile.high_contrast)),
      reduced_motion: Boolean(Number(nextProfile.reduced_motion ?? profile.reduced_motion)),
    };
    applyProfile(payload);
    try {
      const res = await fetch('/assistant/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.ok) applyProfile(data.profile);
    } catch (err) {
      console.warn('[pengteu profile command save]', err);
    }
  }

  function getPengteuCommand(message) {
    const text = (message || '').replace(/\s+/g, '').toLowerCase();
    if (!text) return null;
    const next = { ...profile };

    // 글씨 키우기 — 잘 안 보임 / 눈이 침침 / 흐릿 등 자연어 포함
    if (text.includes('잘안보') || text.includes('안보여') || text.includes('안보임') || text.includes('글씨키')
        || text.includes('크게보') || text.includes('화면키') || text.includes('침침') || text.includes('흐려')
        || text.includes('흐릿') || text.includes('눈이안') || text.includes('글씨크')) {
      next.text_scale = Math.min(1.45, Math.max(Number(profile.text_scale || 1), 1.25) + 0.1);
      next.high_contrast = 1;
      return { profile: next, reply: '좋아요. 글씨를 더 키우고 대비도 높였어요. 이제 화면이 더 또렷하게 보일 거예요.' };
    }
    if (text.includes('글씨작') || text.includes('작게보') || text.includes('화면줄') || text.includes('글씨줄')) {
      next.text_scale = Math.max(1, Number(profile.text_scale || 1) - 0.1);
      return { profile: next, reply: '좋아요. 글씨 크기를 조금 줄였어요.' };
    }
    if (text.includes('천천히') || text.includes('느리게') || text.includes('말속도줄') || text.includes('천천')) {
      next.voice_rate = Math.max(0.65, Number(profile.voice_rate || 0.85) - 0.1);
      return { profile: next, reply: '네, 제가 더 천천히 말할게요.' };
    }
    if (text.includes('빨리말') || text.includes('빠르게말') || text.includes('말속도올') || text.includes('빨리해')) {
      next.voice_rate = Math.min(1.15, Number(profile.voice_rate || 0.85) + 0.1);
      return { profile: next, reply: '알겠어요. 말하는 속도를 조금 빠르게 바꿨어요.' };
    }
    // 볼륨 키우기 — 잘 안 들림 / 목소리 크게 등
    if (text.includes('소리키') || text.includes('볼륨키') || text.includes('크게말') || text.includes('안들려')
        || text.includes('잘안들') || text.includes('목소리키') || text.includes('목소리크') || text.includes('소리크')) {
      next.tts_volume = Math.min(1, Number(profile.tts_volume || 0.85) + 0.15);
      return { profile: next, reply: '좋아요. 제 목소리 볼륨을 더 크게 했어요.' };
    }
    if (text.includes('소리줄') || text.includes('볼륨줄') || text.includes('작게말') || text.includes('시끄')) {
      next.tts_volume = Math.max(0.15, Number(profile.tts_volume || 0.85) - 0.15);
      return { profile: next, reply: '네, 제 목소리 볼륨을 조금 낮췄어요.' };
    }
    if (text.includes('움직임줄') || text.includes('어지러') || text.includes('애니메이션줄') || text.includes('어지럽')) {
      next.reduced_motion = 1;
      return { profile: next, reply: '알겠어요. 화면 움직임을 줄여서 더 편하게 보이도록 했어요.' };
    }
    return null;
  }

  window.PengteuAssistantNative = {
    onTtsEnd: () => {
      pengteuSpeaking = false;
      window.dispatchEvent(new CustomEvent('pengteu-speaking-end'));
    },
    onSttStart: () => {
      pengteuListening = true;
      if (micBtn) {
        micBtn.classList.add('is-listening');
        micBtn.textContent = '듣는 중';
      }
    },
    onSttEnd: () => {
      pengteuListening = false;
      if (micBtn) {
        micBtn.classList.remove('is-listening');
        micBtn.textContent = '마이크';
      }
      maybeReleaseMic();
    },
    onSttResult: (text) => {
      const message = String(text || '').trim();
      if (!message) return;
      window.PengteuProactive && window.PengteuProactive.registerSttSuccess();
      input.value = message;
      askPengteu(message);
    },
    onSttError: (message) => {
      pengteuListening = false;
      if (micBtn) {
        micBtn.classList.remove('is-listening');
        micBtn.textContent = '마이크';
      }
      if (message) appendMessage('assistant', message);
      window.PengteuProactive && window.PengteuProactive.registerSttFailure();
    },
  };

  // 진행 중인 폴백 TTS 오디오를 멈춘다(브릿지 없는 환경에서 펭트 말 끊기용).
  function stopPengteuAudio() {
    if (pengteuAudio) {
      try {
        pengteuAudio.pause();
        pengteuAudio.onended = null;
        pengteuAudio.onerror = null;
      } catch (e) {}
      pengteuAudio = null;
    }
    try { if ('speechSynthesis' in window) window.speechSynthesis.cancel(); } catch (e) {}
  }

  // Google 번역 TTS는 요청당 길이 제한(~200자)이 있어 문장 단위로 쪼갠다.
  function splitTtsChunks(text, maxLen = 180) {
    const sentences = String(text).replace(/\s+/g, ' ').trim().split(/(?<=[.!?。！？])\s+/);
    const chunks = [];
    let buf = '';
    const push = (s) => { if (s) chunks.push(s); };
    for (let piece of sentences) {
      while (piece.length > maxLen) {
        push(buf); buf = '';
        push(piece.slice(0, maxLen));
        piece = piece.slice(maxLen);
      }
      if ((buf + ' ' + piece).trim().length > maxLen) { push(buf); buf = piece; }
      else { buf = (buf ? buf + ' ' : '') + piece; }
    }
    push(buf);
    return chunks.filter(Boolean);
  }

  // 네이티브 브릿지가 없을 때(폰 브라우저·데스크톱)의 폴백.
  // Web Speech API(speechSynthesis)는 안드로이드 WebView에서 무음이라,
  // 운동 TTS와 동일하게 Google 번역 TTS를 Audio()로 재생한다.
  function speakViaAudioFallback(text) {
    const volume = Math.min(1, Math.max(0, Number(profile.tts_volume || 0.85)));
    const chunks = splitTtsChunks(text);
    let idx = 0;
    let started = false;

    const finish = () => {
      pengteuAudio = null;
      pengteuSpeaking = false;
      window.dispatchEvent(new CustomEvent('pengteu-speaking-end'));
    };

    // 최후 폴백: Google 오디오도 실패하면 Web Speech(데스크톱에서만 실효)로.
    const speakSynthFallback = () => {
      if (!('speechSynthesis' in window)) { finish(); return; }
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'ko-KR';
      u.rate = Number(profile.voice_rate || 0.85);
      u.volume = volume;
      u.onend = finish;
      u.onerror = finish;
      window.speechSynthesis.speak(u);
    };

    const playNext = () => {
      if (idx >= chunks.length) { finish(); return; }
      const url = `https://translate.google.com/translate_tts?ie=UTF-8&tl=ko&client=tw-ob&q=${encodeURIComponent(chunks[idx])}`;
      const audio = new Audio();
      pengteuAudio = audio;
      audio.referrerPolicy = 'no-referrer';
      audio.volume = volume;
      audio.src = url;
      audio.onended = () => { idx += 1; playNext(); };
      audio.onerror = () => {
        // 첫 청크부터 실패하면 Web Speech로, 중간 실패면 다음 청크로 진행.
        if (!started) speakSynthFallback(); else { idx += 1; playNext(); }
      };
      audio.play().then(() => { started = true; }).catch(() => {
        if (!started) speakSynthFallback(); else { idx += 1; playNext(); }
      });
    };

    pengteuSpeaking = true;
    window.dispatchEvent(new CustomEvent('pengteu-speaking-start'));
    playNext();
  }

  function speak(text) {
    if (!text || Number(profile.tts_volume) <= 0) return;
    if (window.AndroidBridge && typeof window.AndroidBridge.speakPengteu === 'function') {
      pengteuSpeaking = true;
      window.dispatchEvent(new CustomEvent('pengteu-speaking-start'));
      window.AndroidBridge.speakPengteu(
        text,
        Number(profile.voice_rate || 0.85),
        Number(profile.tts_volume || 0.85)
      );
      return;
    }
    // 브릿지가 없으면(폰 브라우저·데스크톱) 운동과 동일한 Audio 방식으로 재생.
    stopPengteuAudio();
    speakViaAudioFallback(text);
  }

  function initPengteuRecognition() {
    if (window.AndroidBridge && typeof window.AndroidBridge.startPengteuStt === 'function') {
      return null;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR || !micBtn) {
      if (micBtn) micBtn.disabled = true;
      return null;
    }
    const recognition = new SR();
    recognition.lang = 'ko-KR';
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onstart = () => {
      pengteuListening = true;
      micBtn.classList.add('is-listening');
      micBtn.textContent = '듣는 중';
    };
    recognition.onend = () => {
      pengteuListening = false;
      micBtn.classList.remove('is-listening');
      micBtn.textContent = '마이크';
      maybeReleaseMic();
    };
    recognition.onerror = () => {
      pengteuListening = false;
      micBtn.classList.remove('is-listening');
      micBtn.textContent = '마이크';
      window.PengteuProactive && window.PengteuProactive.registerSttFailure();
      maybeReleaseMic();
    };
    recognition.onresult = (event) => {
      const text = Array.from(event.results || [])
        .map((result) => result[0] && result[0].transcript)
        .filter(Boolean)
        .join(' ')
        .trim();
      if (text) {
        window.PengteuProactive && window.PengteuProactive.registerSttSuccess();
        input.value = text;
        askPengteu(text);
      }
    };
    return recognition;
  }

  // ── "펭트야" 웨이크워드 (무핸즈 진입점) ─────────────
  // MicBus 반이중 상태머신은 이미 완성 → 여기서 진입점만 연결한다.
  //  · 브라우저(Web Speech 지원): 저부하 연속 인식으로 "펭트야" 감지.
  //  · 네이티브 앱(WebView): SpeechRecognition이 없어 웹 리스너는 꺼지고,
  //    네이티브 웨이크워드 엔진이 window.PengteuWake.activate()(=onWakeWord)를 호출.
  let wakeRecognition = null;
  let wakeWantsRun = false;
  let wakeActivating = false;

  function isWakeWord(text) {
    const t = String(text || '').replace(/\s+/g, '');
    if (!t) return false;
    if (/(펭|팽|펜)트(야|아|님|씨|이)/.test(t)) return true;   // 펭트야/펜트야/펭트님...
    if (t.length <= 3 && /(펭|팽|펜)트/.test(t)) return true;     // 짧게 "펭트"만 불러도
    return false;
  }

  function stopWake() {
    if (wakeRecognition) { try { wakeRecognition.stop(); } catch (e) {} }
  }

  // 마이크가 다른 용도(검사 STT·문항 TTS·펭트 대화)일 땐 웨이크 인식을 멈춘다.
  function wakeShouldPause() {
    return (
      document.hidden ||
      pengteuListening || pengteuSpeaking ||
      (window.MicBus && (MicBus.audioBusy || MicBus.owner === 'test')) ||
      (window.App && App.recording)
    );
  }

  function pumpWake() {
    if (!wakeRecognition || !wakeWantsRun || wakeShouldPause()) return;
    try { wakeRecognition.start(); } catch (e) { /* 이미 실행 중이면 무시 */ }
  }

  // 웨이크워드 감지 시: 패널 열고 짧게 응대 후 곧바로 명령 청취로 전환.
  function activateByWake() {
    if (wakeActivating || pengteuListening) return;
    wakeActivating = true;
    stopWake();
    openPanel();
    const ack = '네, 말씀하세요.';
    appendMessage('assistant', ack);
    const startListen = () => {
      window.removeEventListener('pengteu-speaking-end', startListen);
      wakeActivating = false;
      startPengteuListening();
    };
    window.addEventListener('pengteu-speaking-end', startListen);
    speak(ack);
    // TTS가 꺼져 있거나(볼륨 0) 즉시 끝난 경우 대비
    if (!pengteuSpeaking) {
      window.removeEventListener('pengteu-speaking-end', startListen);
      wakeActivating = false;
      startPengteuListening();
    }
  }

  function initWakeListener() {
    // 네이티브 앱(WebView): Web Speech가 없으므로 네이티브 always-on 엔진을 구동한다.
    // 마이크 경합(문항TTS·검사STT·펭트 대화)엔 웹의 wakeShouldPause 로직이 큰 스위치를 껐다 켠다.
    if (window.AndroidBridge && typeof window.AndroidBridge.startWakeWord === 'function') {
      let nativeWakeOn = false;
      const syncNativeWake = () => {
        const shouldRun = !wakeShouldPause();
        if (shouldRun && !nativeWakeOn) {
          nativeWakeOn = true;
          try { window.AndroidBridge.startWakeWord(); } catch (e) {}
        } else if (!shouldRun && nativeWakeOn) {
          nativeWakeOn = false;
          try { window.AndroidBridge.stopWakeWord(); } catch (e) {}
        }
      };
      window.addEventListener('pengteu-speaking-end', () => setTimeout(syncNativeWake, 300));
      document.addEventListener('visibilitychange', syncNativeWake);
      setInterval(syncNativeWake, 3000);
      syncNativeWake();
      return;
    }
    // 네이티브 STT만 있고 웨이크 엔진이 없는 구버전 브릿지: 웹 연속인식도 끈다.
    if (window.AndroidBridge && typeof window.AndroidBridge.startPengteuStt === 'function') return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    wakeRecognition = new SR();
    wakeRecognition.lang = 'ko-KR';
    wakeRecognition.continuous = true;
    wakeRecognition.interimResults = true;
    wakeRecognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const alt = event.results[i][0];
        if (alt && isWakeWord(alt.transcript)) { stopWake(); activateByWake(); return; }
      }
    };
    wakeRecognition.onend = () => { if (wakeWantsRun && !wakeShouldPause()) pumpWake(); };
    wakeRecognition.onerror = () => {};   // onend/pump에서 재개
    wakeWantsRun = true;
    pumpWake();
    // 마이크가 풀리면 웨이크 인식 재개(안전망 포함)
    window.addEventListener('pengteu-speaking-end', () => setTimeout(pumpWake, 300));
    document.addEventListener('visibilitychange', pumpWake);
    setInterval(pumpWake, 4000);
  }

  // 펭트 STT 시작(공통 진입점) — 마이크 양도 후 네이티브/웹 STT 시작.
  function startPengteuListening() {
    stopWake();  // 웨이크 리스너가 돌고 있으면 마이크 양보
    if (window.MicBus) MicBus.grantToPengteu();
    if (window.AndroidBridge && typeof window.AndroidBridge.startPengteuStt === 'function') {
      if (!pengteuListening) window.AndroidBridge.startPengteuStt();
      return;
    }
    if (!pengteuRecognition) pengteuRecognition = initPengteuRecognition();
    if (!pengteuRecognition || pengteuListening) return;
    try { pengteuRecognition.start(); } catch (err) { console.warn('[pengteu stt]', err); }
  }

  function togglePengteuMic() {
    if (window.AndroidBridge && typeof window.AndroidBridge.startPengteuStt === 'function') {
      if (pengteuSpeaking && typeof window.AndroidBridge.stopPengteuTts === 'function') {
        window.AndroidBridge.stopPengteuTts();
        pengteuSpeaking = false;
      }
      startPengteuListening();
      return;
    }
    if (!pengteuRecognition) pengteuRecognition = initPengteuRecognition();
    if (!pengteuRecognition || pengteuSpeaking) {
      stopPengteuAudio();
      pengteuSpeaking = false;
    }
    if (!pengteuRecognition) return;
    if (pengteuListening) {
      pengteuRecognition.stop();
      return;
    }
    startPengteuListening();
  }

  // 운동 페이지에서 "어려워/모르겠어/다시 설명" 류를 도움 요청으로 인식.
  function isExerciseHelpIntent(message) {
    const path = window.location.pathname || '';
    if (!path.startsWith('/exercise')) return false;
    const t = (message || '').replace(/\s+/g, '');
    return ['어려', '모르겠', '못하겠', '못따라', '어떻게해', '다시설명', '다시알려',
            '헷갈', '이해가안', '따라하기힘', '천천히설명'].some(k => t.includes(k));
  }

  // 로컬 폴백: 현재 안내 문장을 짧게 끊어 "차근차근" 다시(항상 동작, GPT 불필요).
  function buildSimpleExerciseSteps(info) {
    const name = (info && info.name) ? `지금은 ${info.name} 시간이에요. ` : '';
    const cue = (info && info.cue || '').trim();
    const prefixes = ['먼저', '그다음', '이어서', '마지막으로'];
    let body;
    if (cue) {
      const parts = cue.split(/[.。!?！？]\s*/).map(s => s.trim()).filter(Boolean).slice(0, 4);
      body = parts.length
        ? parts.map((s, i) => `${prefixes[i] || ''} ${s}`).join('. ')
        : cue;
    } else {
      body = '화면의 그림을 보고 저를 천천히 따라 해 주세요';
    }
    return `괜찮아요, 천천히 같이 해봐요. ${name}${body}. 급하지 않아요, 편하게 하시면 돼요. 제가 옆에서 기다릴게요.`;
  }

  // "운동 어려워" → 현재 동작을 GPT로 '다른 표현'으로 쉽게(키 있을 때), 아니면 로컬 폴백.
  async function reexplainExercise() {
    const info = (window.ExercisePengteuInfo && window.ExercisePengteuInfo.current)
      ? (window.ExercisePengteuInfo.current() || {}) : {};
    const cue = (info.cue || info.name || '').trim();
    if (cue) {
      try {
        const res = await fetch('/assistant/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message:
            `어르신이 지금 하는 운동 동작을 어려워하세요. 아래 안내를 더 짧고 쉬운 다른 표현으로, ` +
            `한 동작씩 차근차근 2~4문장으로 다시 설명해 주세요. 새로운 동작을 지어내지 말고 ` +
            `같은 동작만 쉽게 풀어 주세요: "${cue}"` }),
        });
        const data = await res.json();
        if (data && data.ok && data.reply_source === 'openai_fallback' && data.reply) {
          appendMessage('assistant', data.reply);
          speak(data.reply);
          return;
        }
      } catch (e) { /* 로컬 폴백으로 진행 */ }
    }
    const local = buildSimpleExerciseSteps(info);
    appendMessage('assistant', local);
    speak(local);
  }

  async function askPengteu(message) {
    openPanel();
    appendMessage('user', message);
    // 운동 중 "어려워/다시 설명" → 현재 동작을 쉽게 다시 설명(로컬 명령·GPT보다 우선).
    if (isExerciseHelpIntent(message)) {
      await reexplainExercise();
      return;
    }
    const command = getPengteuCommand(message);
    if (command) {
      await persistProfile(command.profile);
      appendMessage('assistant', command.reply);
      speak(command.reply);
      return;
    }
    appendMessage('assistant', '잠깐만요. 기록을 확인하고 있어요.');
    const waitingNode = messages.lastElementChild;
    try {
      const res = await fetch('/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      const reply = data.ok ? data.reply : '지금은 답변을 만들지 못했어요. 잠시 뒤 다시 물어봐 주세요.';
      waitingNode.textContent = reply;
      speak(reply);
    } catch (err) {
      waitingNode.textContent = '서버 연결이 잠시 불안정해요. 다시 시도해 주세요.';
      console.warn('[pengteu chat]', err);
    }
  }

  toggle.addEventListener('click', () => {
    if (panel.hidden) openPanel();
    else closePanel();
  });
  closeBtn.addEventListener('click', closePanel);
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    askPengteu(message);
  });
  if (micBtn) {
    micBtn.addEventListener('click', togglePengteuMic);
    initPengteuRecognition();
  }

  [textScale, voiceRate, volume, highContrast, reducedMotion].forEach((control) => {
    control.addEventListener('change', saveProfile);
    control.addEventListener('input', () => {
      if (control === textScale) applyProfile({ text_scale: Number(textScale.value) });
    });
  });

  function maybeReleaseMic() {
    if (pengteuSpeaking || pengteuListening) return;
    if (window.MicBus) MicBus.releaseFromPengteu();
  }

  // 검사 STT가 마이크를 회수할 때 펭트를 완전히 멈춘다 (말하기·듣기 모두).
  function stopPengteu() {
    stopPengteuAudio();
    if (window.AndroidBridge && typeof window.AndroidBridge.stopPengteuTts === 'function') {
      try { window.AndroidBridge.stopPengteuTts(); } catch (e) {}
    }
    if (pengteuRecognition && pengteuListening) {
      try { pengteuRecognition.stop(); } catch (e) {}
    }
    pengteuSpeaking = false;
    pengteuListening = false;
    if (window.MicBus) MicBus.setAudioBusy(false);
  }

  // 펭트가 말하기 시작하면 마이크를 가져오고(문항 TTS 일시정지) audioBusy로 에코 차단,
  // 말이 끝나면 audioBusy 해제 + (듣지도 않으면) 마이크 반납.
  window.addEventListener('pengteu-speaking-start', () => {
    if (window.MicBus) { MicBus.grantToPengteu(); MicBus.setAudioBusy(true); }
  });
  window.addEventListener('pengteu-speaking-end', () => {
    if (window.MicBus) MicBus.setAudioBusy(false);
    maybeReleaseMic();
  });

  window.PengteuAssistant = {
    ask: askPengteu,
    open: openPanel,
    stop: stopPengteu,
  };

  document.querySelectorAll('[data-pengteu-prompt]').forEach((button) => {
    button.addEventListener('click', () => {
      const prompt = button.getAttribute('data-pengteu-prompt') || '';
      if (prompt) askPengteu(prompt);
    });
  });

  // ── 펭트 능동 안내 (상황 인식) ────────────────────
  // 검사 페이지에서 일정 시간 아무 입력/행동이 없으면(막혀 있으면)
  // 펭트가 먼저 "많이 어려우신가요?"라고 도움을 제안한다. 잔소리 방지를 위해
  // 페이지당 최대 횟수를 제한한다. 운동 페이지는 사용자가 몸을 움직이는 중이라
  // DOM 무입력으로 오발동하므로 제외한다(센서 기반은 별도).
  const pengteuPath = window.location.pathname || '';
  const isTestPage = pengteuPath.startsWith('/item');
  const IDLE_MS = 10000;
  const MAX_IDLE_PROMPTS = 2;   // 세션 전체 상한(페이지마다 리셋되지 않음)
  let idleTimer = null;
  let sttFailStreak = 0;

  function proactiveSay(text) {
    if (pengteuSpeaking || pengteuListening) return false;
    if (window.MicBus && MicBus.audioBusy) return false;   // 문항 TTS 재생 중엔 끼어들지 않음
    if (window.App && App.recording) return false;          // 검사 답변 녹음 중엔 끼어들지 않음(녹음 보호)
    appendMessage('assistant', text);
    speak(text);
    return true;
  }

  function fireIdlePrompt() {
    const used = parseInt(sessionStorage.getItem('pt_idlePrompt') || '0', 10);
    if (used >= MAX_IDLE_PROMPTS) return;                       // 세션 전체 예산 소진
    if (pengteuSpeaking || pengteuListening) { scheduleIdle(); return; }
    if (window.App && App.recording) { scheduleIdle(); return; } // 녹음 중이면 advance-nudge가 담당
    const said = proactiveSay('많이 어려우신가요? 원하시면 "글씨 키워줘"라고 말씀해 주세요. 제가 글씨를 키우거나 천천히 안내해 드릴게요.');
    if (said) sessionStorage.setItem('pt_idlePrompt', String(used + 1));
    scheduleIdle();
  }

  function scheduleIdle() {
    if (!isTestPage) return;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(fireIdlePrompt, IDLE_MS);
  }

  // STT가 연속으로 인식 실패하면(잘 안 들림) 볼륨/속도 조정을 먼저 제안
  function registerSttFailure() {
    sttFailStreak += 1;
    if (sttFailStreak >= 2) {
      sttFailStreak = 0;
      proactiveSay('잘 안 들리시나 봐요. "소리 키워줘" 또는 "천천히 말해줘"라고 하시면 제가 맞춰 드릴게요.');
    }
  }
  function registerSttSuccess() { sttFailStreak = 0; }
  // say: 검사 화면의 무입력 안내(advance-nudge)가 펭트 목소리로 말할 수 있게 노출.
  window.PengteuProactive = { registerSttFailure, registerSttSuccess, say: proactiveSay };

  if (isTestPage) {
    ['pointerdown', 'keydown', 'touchstart', 'input', 'wheel'].forEach((ev) => {
      window.addEventListener(ev, scheduleIdle, { passive: true });
    });
    window.addEventListener('pengteu-speaking-end', scheduleIdle);
    scheduleIdle();
  }

  // 웨이크워드 진입점 연결: 웹은 연속인식, 네이티브는 엔진이 onWakeWord 호출.
  initWakeListener();
  if (window.PengteuAssistantNative) window.PengteuAssistantNative.onWakeWord = activateByWake;
  window.PengteuWake = { activate: activateByWake, isWakeWord };

  // ── 페이지 진입 능동 안내(가벼운 오리엔테이션) + 결과 위로 ──
  // proactiveSay는 문항TTS(audioBusy)·녹음·펭트 대화 중이면 스스로 생략하므로
  // /item 처럼 자체 안내가 있는 화면과는 겹치지 않는다.
  function proactiveOnLoad() {
    const meta = document.getElementById('pengteuPageMeta');
    // (b) 결과 화면에서 저하(MCI 의심)면 먼저 따뜻하게 위로한다.
    if (meta && meta.getAttribute('data-impaired') === '1') {
      openPanel();
      setTimeout(() => proactiveSay('오늘 검사 하시느라 정말 수고 많으셨어요. 결과에 너무 걱정하지 마세요. 이건 건강을 미리 살피기 위한 과정이에요. 궁금한 점이 있으면 저에게 편하게 물어봐 주세요.'), 900);
      return;
    }
    // (a) 지정 페이지에 한해 짧은 안내를 한 번 말한다(겹치면 자동 생략).
    let hint = (meta && meta.getAttribute('data-page-hint')) || '';
    if (!hint && (window.location.pathname || '').startsWith('/gait')) {
      hint = '보행 검사 화면이에요. 준비되시면 안내에 따라 편하게 걸어 주세요. 도움이 필요하면 저를 불러 주세요.';
    }
    if (hint) setTimeout(() => proactiveSay(hint), 1200);
  }
  proactiveOnLoad();

  loadProfile();
})();
