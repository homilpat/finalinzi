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

  // 다중 응답 (naming, sentence_repeat 등)
  multiStep:     0,
  multiAnswers:  {},
};

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
    onTTSComplete();
    return;
  }

  const url   = App.ttsUrls[App.ttsIndex];
  const audio = new Audio(url);
  App.activeAudio = audio;
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
    au.play().catch((e) => console.error('Audio play error:', e));
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

  // naming 이나 orientation 진행 도중이고 마지막 스텝이 아닌 경우, 다음 스텝으로 진행
  if (App.itemType === 'naming') {
    if (App.multiStep < (App.namingAnimals || []).length - 1) {
      advanceMultiStep("");
      return;
    }
  }
  if (App.itemType === 'orientation') {
    if (App.multiStep < (App.orientQuestions || []).length - 1) {
      advanceMultiStep("");
      return;
    }
  }
  if (App.itemType === 'voice_multi') {
    if (App.multiStep < (App.voiceMultiParts || []).length - 1) {
      onVoiceMultiStep("");
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
function initSpeech(onResult) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { console.warn('SpeechRecognition 미지원'); return null; }
  const r = new SR();
  r.lang = 'ko-KR';
  r.continuous = true;
  r.interimResults = true;

  r.onresult = (e) => {
    const text = Array.from(e.results).map(x => x[0].transcript).join('');
    const el = document.getElementById('transcriptText');
    if (el) { el.textContent = text; el.classList.add('has-text'); }
    if (e.results[e.results.length - 1].isFinal && onResult && !App.stepAdvancing) {
      const finalText = text.trim();
      if (finalText) onResult(finalText);
    }
  };
  r.onend = () => {
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

function toggleMic() {
  if (!App.recognition) {
    App.recognition = initSpeech((text) => {
      const key = getCurrentSTTKey();
      App.responses[key] = text;
      App.multiAnswers[key] = text;
      enableSubmit();

      // 다중 단계: 자동 다음 단계
      if (['naming', 'memory', 'voice_multi', 'orientation'].includes(App.itemType)) {
        onStepComplete(text);
      }
    });
  }

  if (!App.recognition) return;

  if (App.recording) {
    App.micStopRequested = true;
    App.recording = false;
    App.recognition.stop();
    const btn = document.getElementById('micBtn');
    if (btn) btn.classList.remove('recording');
    const status = document.getElementById('micStatus');
    if (status) status.textContent = '완료';
  } else {
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
    document.getElementById('micStatus').textContent = '듣는 중...';
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
          startTimer(App.duration);
          return;
        }
        const au = new Audio(audio2[idx]);
        au.onended = () => { idx++; playNext2(); };
        au.onerror = () => { idx++; playNext2(); };
        au.play().catch(() => { idx++; playNext2(); });
      }
      stopTimer();
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
      au.onended = () => {
        startTimer(App.duration);
      };
      au.onerror = () => {
        startTimer(App.duration);
      };
      stopTimer();
      App.timerStarted = true;
      au.play().catch(() => {
        startTimer(App.duration);
      });

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
  const container = document.getElementById('orientationContainer');
  if (container) {
    App.orientQuestions = JSON.parse(container.dataset.questions || '[]');
  }
}

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

  function speak(text) {
    if (!('speechSynthesis' in window) || !text || Number(profile.tts_volume) <= 0) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    utterance.rate = Number(profile.voice_rate || 0.85);
    utterance.volume = Number(profile.tts_volume || 0.85);
    const finishSpeaking = () => {
      pengteuSpeaking = false;
      window.dispatchEvent(new CustomEvent('pengteu-speaking-end'));
    };
    utterance.onstart = () => {
      pengteuSpeaking = true;
      window.dispatchEvent(new CustomEvent('pengteu-speaking-start'));
    };
    utterance.onend = finishSpeaking;
    utterance.onerror = finishSpeaking;
    window.speechSynthesis.speak(utterance);
  }

  function initPengteuRecognition() {
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
    };
    recognition.onerror = () => {
      pengteuListening = false;
      micBtn.classList.remove('is-listening');
      micBtn.textContent = '마이크';
    };
    recognition.onresult = (event) => {
      const text = Array.from(event.results || [])
        .map((result) => result[0] && result[0].transcript)
        .filter(Boolean)
        .join(' ')
        .trim();
      if (text) {
        input.value = text;
        askPengteu(text);
      }
    };
    return recognition;
  }

  function togglePengteuMic() {
    if (!pengteuRecognition) pengteuRecognition = initPengteuRecognition();
    if (!pengteuRecognition || pengteuSpeaking) {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      pengteuSpeaking = false;
    }
    if (!pengteuRecognition) return;
    if (pengteuListening) {
      pengteuRecognition.stop();
      return;
    }
    try {
      pengteuRecognition.start();
    } catch (err) {
      console.warn('[pengteu stt]', err);
    }
  }

  async function askPengteu(message) {
    openPanel();
    appendMessage('user', message);
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

  window.PengteuAssistant = {
    ask: askPengteu,
    open: openPanel,
  };

  document.querySelectorAll('[data-pengteu-prompt]').forEach((button) => {
    button.addEventListener('click', () => {
      const prompt = button.getAttribute('data-pengteu-prompt') || '';
      if (prompt) askPengteu(prompt);
    });
  });

  loadProfile();
})();
