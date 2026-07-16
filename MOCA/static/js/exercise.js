(function () {
  'use strict';

  // 1. 상태 변수
  let currentType = 'A'; // A, B, C, D (유형)
  let currentPhase = 1;  // 1, 2, 3, 4 (단계)
  let phaseElapsed = 0;  // 현재 단계에서 경과한 시간 (초)
  let isPlaying = false; // 재생 상태
  let timerInterval = null;
  let simulatedDuration = 180; // 현재 단계의 전체 초 (1단계: 180초, 2단계: 300초, 3단계: 540초, 4단계: 180초)
  let activeTtsAudio = null;

  // DOM Elements
  const activeScreen = document.querySelector('[data-exercise-active]');
  if (!activeScreen) return;

  const timerDisplay = document.getElementById('exerciseTimer');
  const progressBar = document.getElementById('exerciseProgress');
  const phaseNameLabel = document.getElementById('phaseNameLabel');
  const phaseDescLabel = document.getElementById('phaseDescLabel');
  const bgmPlayingText = document.getElementById('bgmPlayingText');
  const timeSeekSlider = document.getElementById('timeSeekSlider');
  const seekTimeDisplay = document.getElementById('seekTimeDisplay');
  
  // Interactive Feedbacks
  const sensorBadge = document.getElementById('sensorBadge');
  const sensorCueBox = document.getElementById('sensorCueBox');
  const motionPulse = document.getElementById('activeMotionPulse');
  
  const bgmAudio = document.getElementById('bgmAudio');
  const effectAudio = document.getElementById('effectAudio');

  // 2. 운동 프로토콜 가이드 텍스트 & 오디오 정보
  const PhaseDurations = { 1: 180, 2: 300, 3: 540, 4: 180 };
  
  const ExerciseProtocols = {
    A: {
      bgm1: 'A_1단계워밍업.mp3',
      bgm2: 'AC_2단계음악BGM.mp3',
      bgm3: null, // BGM 무음
      bgm4: 'A_4단계쿨다운.mp3',
      vol2: 0.15, vol3: 0
    },
    B: {
      bgm1: 'A_1단계워밍업.mp3',
      bgm2: 'BD_2단계음악BGM.mp3',
      bgm3: null, // BGM 무음
      bgm4: 'A_4단계쿨다운.mp3',
      vol2: 0.15, vol3: 0
    },
    C: {
      bgm1: 'D_1단계워밍업.mp3',
      bgm2: 'AC_2단계음악BGM.mp3',
      bgm3: null, // BGM 무음
      bgm4: 'D_4단계쿨다운.mp3',
      vol2: 0.15, vol3: 0
    },
    D: {
      bgm1: 'D_1단계워밍업.mp3',
      bgm2: 'BD_2단계음악BGM.mp3',
      bgm3: 'BD_2단계음악BGM.mp3',
      bgm4: 'D_4단계쿨다운.mp3',
      vol2: 0.40, vol3: 0.15 // BGM 크게 유지
    }
  };

  const AudioManager = {
    playBgm: function(file, volume = 0.5) {
      if (!bgmAudio) return;
      if (!file) {
        bgmAudio.pause();
        bgmPlayingText.textContent = "재생 중인 배경음: 없음 (음소거)";
        return;
      }
      
      const fullPath = `/static/audio/${file}`;
      if (bgmAudio.src.indexOf(encodeURIComponent(file)) === -1) {
        bgmAudio.src = fullPath;
      }
      bgmAudio.volume = volume;
      bgmAudio.originalVolume = volume;
      bgmAudio.isDucked = false;
      bgmPlayingText.textContent = `재생 중인 배경음: ${file}`;
      
      if (isPlaying) {
        bgmAudio.play().catch(e => console.warn("BGM 재생 차단됨:", e));
      }
    },
    playEffect: function(file) {
      if (!effectAudio) return;
      effectAudio.src = `/static/audio/${file}`;
      effectAudio.play().catch(e => console.warn("효과음 재생 차단됨:", e));
      
      if (file === 'applause.mp3') {
        if (window.applauseTimeout) clearTimeout(window.applauseTimeout);
        window.applauseTimeout = setTimeout(() => {
          try {
            effectAudio.pause();
            effectAudio.currentTime = 0;
          } catch(e){}
        }, 3000);
      }
    },
    speak: function(text, onEnd) {
      if (!isPlaying) return;
      console.log("Speak:", text);
      
      // 이전 재생 중이던 TTS 오디오 인스턴스가 있다면 일시정지 및 초기화
      if (activeTtsAudio) {
        try {
          activeTtsAudio.pause();
          activeTtsAudio.onended = null;
        } catch (e) {}
        activeTtsAudio = null;
      }
      
      // Web Speech API를 호출 중이었다면 캔슬
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }

      const googleTtsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&tl=ko&client=tw-ob&q=${encodeURIComponent(text)}`;
      
      if (bgmAudio && bgmAudio.originalVolume === undefined) {
        bgmAudio.originalVolume = bgmAudio.volume;
      }
      const originalVolume = bgmAudio ? bgmAudio.originalVolume : 0.5;
      
      const tempAudio = new Audio();
      activeTtsAudio = tempAudio;
      tempAudio.referrerPolicy = "no-referrer";
      tempAudio.src = googleTtsUrl;
      
      if (bgmAudio) {
        bgmAudio.isDucked = true;
        bgmAudio.duckedVolume = Math.min(0.02, originalVolume * 0.10); // Duck BGM to at most 2% volume during TTS
        bgmAudio.volume = bgmAudio.duckedVolume;
      }
      
      let fallbackTriggered = false;
      const runFallback = () => {
        if (fallbackTriggered) return;
        fallbackTriggered = true;
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = 'ko-KR';
          utterance.rate = 1.1;
          utterance.onend = () => {
            if (bgmAudio) {
              bgmAudio.isDucked = false;
              bgmAudio.volume = originalVolume;
            }
            if (onEnd) onEnd();
          };
          utterance.onerror = () => {
            if (bgmAudio) {
              bgmAudio.isDucked = false;
              bgmAudio.volume = originalVolume;
            }
            if (onEnd) onEnd();
          };
          window.speechSynthesis.speak(utterance);
        } else {
          if (bgmAudio) {
            bgmAudio.isDucked = false;
            bgmAudio.volume = originalVolume;
          }
          if (onEnd) onEnd();
        }
      };

      tempAudio.onerror = runFallback;
      tempAudio.play().then(() => {
        tempAudio.onended = () => {
          if (activeTtsAudio === tempAudio) {
            activeTtsAudio = null;
          }
          if (bgmAudio) {
            bgmAudio.isDucked = false;
            bgmAudio.volume = originalVolume;
          }
          if (onEnd) onEnd();
        };
      }).catch(runFallback);
    }
  };

  // 3. 네이티브 안드로이드 브릿지 정의
  const SensorBridge = {
    isAndroid: typeof window.AndroidBridge !== 'undefined',
    startCalibration: function() {
      if (this.isAndroid) window.AndroidBridge.startCalibration();
    },
    startMeasurement: function(stage) {
      if (this.isAndroid) window.AndroidBridge.startMeasurement(stage);
    },
    stopMeasurement: function() {
      if (this.isAndroid) window.AndroidBridge.stopMeasurement();
    },
    onSensorEvent: function(eventJson) {
      try {
        const event = typeof eventJson === 'string' ? JSON.parse(eventJson) : eventJson;
        handleSensorData(event);
      } catch(e) {
        console.error("SensorBridge parse error:", e);
      }
    }
  };
  window.SensorBridge = SensorBridge;

  // 4. 타이머 및 화면 갱신 핵심 로직
  function startTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      if (!isPlaying) return;

      phaseElapsed += 1;
      
      // A/B/C유형 1단계: 30초 대기 중 및 시작 시점 제어 (안내 멘트 완료와 연동)
      if (currentPhase === 1 && (currentType === 'A' || currentType === 'B' || currentType === 'C')) {
        if (phaseGameData.welcomeSpeaking) {
          // 안내 TTS가 끝날 때까지 29초 지점에서 홀딩
          if (phaseElapsed >= 29) {
            phaseElapsed = 29;
          }
          if (bgmAudio && !bgmAudio.paused) {
            bgmAudio.pause();
          }
          sensorCueBox.innerHTML = `안내 TTS 진행 중... (안내가 끝나면 노래가 시작됩니다)`;
        } else {
          // 안내 완료 후: 30초 이상 구간 BGM 재생 상태 유지
          if (bgmAudio && bgmAudio.paused && isPlaying) {
            bgmAudio.play().catch(e => console.warn(e));
          }
        }
      }

      // D유형 1단계: 60초 대기 중 및 시작 시점 제어 (안내 멘트 완료와 연동)
      if (currentPhase === 1 && currentType === 'D') {
        if (phaseGameData.welcomeSpeaking) {
          // 안내 TTS가 끝날 때까지 59초 지점에서 홀딩
          if (phaseElapsed >= 59) {
            phaseElapsed = 59;
          }
          if (bgmAudio && !bgmAudio.paused) {
            bgmAudio.pause();
          }
          sensorCueBox.innerHTML = `안내 TTS 진행 중... (안내가 끝나면 노래가 시작됩니다)`;
        } else {
          // 안내 완료 후: 60초 이상 구간 BGM 재생 상태 유지
          if (bgmAudio && bgmAudio.paused && isPlaying) {
            bgmAudio.play().catch(e => console.warn(e));
          }
        }
      }

      // C유형 2단계: 안내 멘트 완료 전까지 19초 지점에서 홀딩
      if (currentPhase === 2 && currentType === 'C') {
        if (phaseGameData.welcomeSpeaking) {
          if (phaseElapsed >= 19) {
            phaseElapsed = 19;
          }
        }
      }

      // D유형 2단계: 안내 멘트 완료 전까지 19초 지점에서 홀딩
      if (currentPhase === 2 && currentType === 'D') {
        if (phaseGameData.welcomeSpeaking) {
          if (phaseElapsed >= 19) {
            phaseElapsed = 19;
          }
        }
      }

      // D유형 3단계: 안내 멘트(introSpeaking) 진행 중일 때 타겟 초에서 타이머 홀딩
      if (currentPhase === 3 && currentType === 'D') {
        if (phaseGameData.introSpeaking && phaseGameData.holdTime !== undefined) {
          if (phaseElapsed >= phaseGameData.holdTime) {
            phaseElapsed = phaseGameData.holdTime;
          }
        }
      }

      // A유형 2단계: 5분간 연속 시나리오 TTS 출력 제어
      if (currentPhase === 2 && currentType === 'A') {
        const script = {
          20: "양발을 어깨너비로 벌려 주세요. 시선은 정면을 바라보고 허리는 곧게 펴겠습니다. 투명 의자에 앉듯 엉덩이를 뒤로 보내며 천천히 내려가겠습니다.",
          31: "하나...",
          34: "둘...",
          37: "셋...",
          40: "그대로 자세를 유지합니다.",
          44: "하나...",
          47: "둘...",
          51: "이제 천천히 일어나겠습니다. 아주 좋습니다.",
          58: "천천히 내려갑니다.",
          63: "하나...",
          66: "둘...",
          69: "셋...",
          73: "천천히 일어납니다.",
          79: "다시 한 번 내려갑니다.",
          84: "하나...",
          87: "둘...",
          90: "셋...",
          94: "천천히 일어납니다.",
          100: "마지막입니다.",
          104: "하나...",
          107: "둘...",
          110: "셋...",
          114: "천천히 일어나겠습니다. 아주 잘하셨습니다.",
          122: "잠시 몸 상태를 확인해 보겠습니다. 호흡은 편안하신가요? 힘들지 않으시면 지금처럼 천천히 계속하겠습니다.",
          133: "두 다리에 체중을 고르게 실어 주세요. 오른쪽 다리를 옆으로 천천히 들어 보겠습니다.",
          141: "오른쪽 다리 올립니다.",
          145: "하나... 둘... 셋...",
          149: "천천히 내려옵니다.",
          154: "다시 한 번 올립니다.",
          158: "하나... 둘... 셋...",
          162: "천천히 내려옵니다.",
          167: "마지막입니다. 올립니다.",
          171: "하나... 둘... 셋...",
          175: "천천히 내려옵니다. 아주 좋습니다.",
          181: "이번에는 왼쪽 다리입니다. 옆으로 올립니다.",
          186: "하나... 둘... 셋...",
          190: "천천히 내려옵니다.",
          195: "다시 한 번 올립니다.",
          199: "하나... 둘... 셋...",
          203: "천천히 내려옵니다.",
          208: "마지막입니다. 올립니다.",
          212: "하나... 둘... 셋...",
          216: "천천히 내려옵니다. 몸의 중심을 아주 잘 유지하고 계십니다.",
          224: "이번에는 다리를 뒤로 보내는 운동입니다. 허리는 곧게 세우고 시선은 앞을 바라봐 주세요. 오른쪽 다리를 뒤로 보냅니다. 하나... 둘... 셋... 천천히 돌아옵니다.",
          238: "이번에는 왼쪽 다리입니다. 뒤로 보냅니다. 하나... 둘... 셋... 천천히 돌아옵니다.",
          250: "두 발을 편안하게 벌리고 서 주세요. 이제 발뒤꿈치를 천천히 들어 보겠습니다. 하나... 둘... 셋... 그대로 유지합니다. 하나... 둘... 천천히 내려옵니다.",
          265: "다시 한 번 들어 올립니다. 하나... 둘... 셋... 천천히 내려옵니다.",
          273: "마지막입니다. 들어 올립니다. 하나... 둘... 셋... 천천히 내려옵니다. 아주 잘하고 계십니다.",
          285: "두 번째 운동을 모두 마쳤습니다. 자세를 바르게 유지하며 천천히 움직이는 것이 다리의 힘과 균형을 기르는 데 도움이 됩니다. 몸 상태를 한 번 확인해 보시고, 필요하면 물을 한 모금 드셔도 좋습니다. 준비가 되셨다면 다음 운동으로 함께 넘어가겠습니다."
        };
        if (phaseGameData.spokenKeys === undefined) {
          phaseGameData.spokenKeys = {};
        }
        Object.keys(script).forEach(kStr => {
          const k = parseInt(kStr);
          if (k <= phaseElapsed && !phaseGameData.spokenKeys[k]) {
            phaseGameData.spokenKeys[k] = true;
            sensorCueBox.textContent = script[k];
            AudioManager.speak(script[k]);
          }
        });
      }

      // B유형 2단계: 5분간 연속 시나리오 TTS 출력 제어 (이동 시간 확보를 위한 여유로운 멘트 분산 배치)
      if (currentPhase === 2 && currentType === 'B') {
        const scriptB = {
          20: "다리를 어깨너비보다 조금 넓게 벌려 주세요. 무릎은 살짝 굽힌 상태를 유지합니다. 이제 오른쪽으로 천천히 한 걸음 이동하겠습니다.",
          32: "하나...",
          36: "둘...",
          40: "다시 오른쪽으로 한 걸음 더 이동합니다.",
          46: "하나...",
          50: "둘...",
          55: "아주 좋습니다. 어깨와 허리는 정면을 향한 채로 움직여 보겠습니다.",
          65: "이번에는 왼쪽으로 천천히 돌아가겠습니다.",
          71: "하나...",
          75: "둘...",
          80: "천천히 이어가겠습니다. 왼쪽으로 한 걸음 더 이동합니다.",
          86: "하나...",
          90: "둘...",
          95: "아주 좋습니다. 발을 끌지 말고 가볍게 옮겨 보세요.",
          105: "잠시 몸 상태를 살펴보겠습니다. 다리에 힘이 많이 들어가지는 않으신가요? 괜찮으시면 그대로 이어가겠습니다. 만약 어지럽거나 발이 불안하게 느껴진다면 잠시 멈춰 쉬셔도 괜찮습니다.",
          125: "이번에는 걸음을 조금 작게 해보겠습니다. 발을 가까이 옮기며 천천히 이동합니다. 오른쪽입니다.",
          135: "하나...",
          139: "둘...",
          143: "셋...",
          148: "아주 좋습니다. 이제 왼쪽입니다.",
          155: "하나...",
          159: "둘...",
          163: "셋...",
          168: "지금처럼 몸의 중심을 잘 유지해 주세요.",
          178: "이번에는 걸음을 조금만 넓게 해보겠습니다. 무리하지 않는 범위에서 움직이면 됩니다. 오른쪽으로 이동합니다.",
          188: "하나...",
          192: "둘...",
          198: "천천히 이어갑니다. 오른쪽으로 한 걸음 더.",
          204: "하나...",
          208: "둘...",
          214: "이번에는 왼쪽입니다. 왼쪽으로 한 걸음 이동합니다.",
          222: "하나...",
          226: "둘...",
          232: "천천히 이어갑니다. 왼쪽으로 한 걸음 더.",
          238: "하나...",
          242: "둘...",
          248: "아주 안정적으로 잘 따라오고 계십니다. 호흡은 편안하신가요? 조금 천천히 하셔도 괜찮습니다. 몸이 편안한 범위에서 움직이는 것이 가장 중요합니다. 괜찮으시면 마지막으로 한 번 더 이어가겠습니다.",
          268: "마지막입니다. 오른쪽으로 천천히 이동합니다.",
          274: "하나...",
          278: "둘...",
          284: "이제 왼쪽으로 돌아오겠습니다.",
          290: "하나...",
          294: "둘...",
          298: "두 번째 운동을 모두 마쳤습니다. 옆으로 걷는 동작은 다리의 힘과 균형을 기르는 데 도움이 됩니다. 잠시 숨을 고르고 몸을 편안하게 쉬어 주세요. 준비가 되셨다면 다음 운동으로 함께 넘어가겠습니다."
        };
        if (phaseGameData.spokenKeys === undefined) {
          phaseGameData.spokenKeys = {};
        }
        Object.keys(scriptB).forEach(kStr => {
          const k = parseInt(kStr);
          if (k <= phaseElapsed && !phaseGameData.spokenKeys[k]) {
            phaseGameData.spokenKeys[k] = true;
            sensorCueBox.textContent = scriptB[k];
            AudioManager.speak(scriptB[k]);
          }
        });
      }
      
      // C유형 2단계: 5분간 연속 시나리오 TTS 출력 제어
      if (currentPhase === 2 && currentType === 'C') {
        const scriptC = {
          20: "먼저 오른쪽 다리를 앞으로 천천히 펴겠습니다.",
          25: "하나.",
          28: "둘.",
          31: "셋.",
          34: "그대로 잠시 유지합니다.",
          38: "하나.",
          41: "둘.",
          44: "천천히 내려놓겠습니다. 아주 좋습니다.",
          50: "같은 동작을 한 번 더 하겠습니다.",
          55: "하나.",
          58: "둘.",
          61: "셋.",
          64: "천천히 내려놓습니다.",
          69: "이번에는 왼쪽 다리입니다.",
          74: "하나.",
          77: "둘.",
          80: "셋.",
          83: "그대로 유지합니다.",
          87: "하나.",
          90: "둘.",
          93: "천천히 내려놓습니다.",
          98: "한 번 더 반복하겠습니다.",
          103: "하나.",
          106: "둘.",
          109: "셋.",
          112: "천천히 내려놓습니다. 허벅지에 힘이 들어가는 것을 느껴 보세요.",
          120: "잠시 몸 상태를 확인하겠습니다. 허리나 무릎에 불편함은 없으신가요? 힘이 많이 들어간다면 잠시 쉬셔도 괜찮습니다. 괜찮으시면 다음 동작을 이어가겠습니다.",
          135: "이번에는 오른쪽 무릎을 천천히 들어 올리겠습니다.",
          141: "하나.",
          144: "둘.",
          147: "셋.",
          150: "천천히 내려놓습니다.",
          155: "한 번 더 하겠습니다.",
          160: "하나.",
          163: "둘.",
          166: "셋.",
          169: "내려놓습니다.",
          174: "이번에는 왼쪽입니다.",
          179: "하나.",
          182: "둘.",
          185: "셋.",
          188: "천천히 내려놓습니다.",
          193: "한 번 더 반복하겠습니다.",
          198: "하나.",
          201: "둘.",
          204: "셋.",
          207: "내려놓습니다. 아주 잘하고 계십니다. 몸은 곧게 세운 자세를 유지해 주세요.",
          215: "이번에는 발목을 움직여 보겠습니다. 발끝을 몸쪽으로 천천히 당깁니다.",
          221: "하나.",
          224: "둘.",
          227: "셋.",
          230: "이번에는 발끝을 앞으로 밀어 보겠습니다.",
          235: "하나.",
          238: "둘.",
          241: "셋. 아주 좋습니다.",
          246: "한 번 더 반복하겠습니다. 몸쪽으로 당깁니다.",
          252: "하나.",
          255: "둘.",
          258: "셋.",
          261: "앞으로 밀어냅니다.",
          265: "하나.",
          268: "둘.",
          271: "셋. 종아리와 발목이 부드럽게 움직이고 있습니다. 숨은 편안하게 쉬고 계신가요? 어깨와 목에 힘이 들어가지 않도록 편안하게 유지해 주세요.",
          282: "두 발을 바닥에 편안하게 놓아 주세요. 발뒤꿈치를 천천히 들어 올립니다.",
          288: "하나.",
          291: "둘.",
          294: "셋.",
          297: "천천히 내려놓습니다.",
          298: "두 번째 운동을 모두 마쳤습니다. 앉아서 하는 운동도 다리의 힘을 기르고 걷는 능력을 유지하는 데 도움이 됩니다. 바로 일어나지 마시고, 잠시 편안하게 호흡을 고르겠습니다. 몸 상태가 괜찮으시면 다음 운동을 시작하겠습니다."
        };
        if (phaseGameData.spokenKeys === undefined) {
          phaseGameData.spokenKeys = {};
        }
        Object.keys(scriptC).forEach(kStr => {
          const k = parseInt(kStr);
          if (k <= phaseElapsed && !phaseGameData.spokenKeys[k]) {
            phaseGameData.spokenKeys[k] = true;
            sensorCueBox.textContent = scriptC[k];
            AudioManager.speak(scriptC[k]);
          }
        });
      }

      // D유형 2단계: 5분간 연속 시나리오 TTS 출력 제어
      if (currentPhase === 2 && currentType === 'D') {
        const scriptD = {
          20: "오른쪽",
          24: "왼쪽",
          28: "오른쪽",
          32: "왼쪽",
          36: "와, 아주 좋습니다!",
          41: "이번에는 조금 더 리듬을 타볼까요?",
          47: "오른쪽",
          51: "왼쪽",
          55: "하나 더!",
          59: "오른쪽",
          63: "왼쪽",
          67: "정말 잘하고 계십니다!",
          78: "이번에는 무릎을 번갈아 들어 보겠습니다!",
          84: "오른쪽 무릎, 쑥!",
          88: "내립니다.",
          92: "왼쪽 무릎, 쑥!",
          96: "내립니다.",
          100: "좋아요!",
          104: "음악에 맞춰 한 번 더 갑니다!",
          109: "오른쪽!",
          113: "왼쪽!",
          117: "오른쪽!",
          121: "왼쪽!",
          125: "이야, 박자를 정말 잘 맞추고 계십니다!",
          135: "잠깐 쉬어 갈게요! 힘들지는 않으신가요? 숨을 천천히 쉬어 보세요. 어지럽거나 몸이 불편하면 잠시 쉬셔도 괜찮습니다. 보호자분도 어르신이 편안한지 한 번 살펴봐 주세요.",
          160: "이번에는 발끝을 까딱까딱 움직여 볼까요?",
          166: "발끝 위로!",
          170: "내립니다!",
          174: "다시 위로!",
          178: "내립니다!",
          182: "아주 좋아요!",
          186: "이번에는 음악에 맞춰 더 신나게!",
          191: "까딱!",
          195: "까딱!",
          199: "까딱!",
          203: "까딱!",
          207: "종아리도 함께 운동하고 있습니다!",
          220: "마지막 운동입니다!",
          225: "발을 바닥에 가볍게 톡톡 두드려 볼까요?",
          231: "톡!",
          235: "톡!",
          239: "톡!",
          243: "톡!",
          247: "아주 좋아요!",
          251: "이번에는 조금 더 신나게 갑니다!",
          256: "톡톡!",
          260: "톡톡!",
          264: "톡톡!",
          268: "끝까지 잘하고 계십니다! 최고입니다!",
          278: "정말 잘하셨습니다! 몸을 즐겁게 움직이는 것만으로도 다리에 힘이 생기고 건강에 큰 도움이 됩니다. 지금은 바로 일어나지 마시고 의자에 편안히 앉아 잠시 쉬어 주세요. 보호자분께서는 어르신의 몸 상태를 한 번 확인해 주세요. 준비가 되셨다면 다음 운동도 해 보겠습니다!"
        };
        if (phaseGameData.spokenKeys === undefined) {
          phaseGameData.spokenKeys = {};
        }
        Object.keys(scriptD).forEach(kStr => {
          const k = parseInt(kStr);
          if (k <= phaseElapsed && !phaseGameData.spokenKeys[k]) {
            phaseGameData.spokenKeys[k] = true;
            sensorCueBox.textContent = scriptD[k];
            AudioManager.speak(scriptD[k]);
          }
        });
      }

      // 단계 시간 초과 시 다음 단계 자동 이행
      if (phaseElapsed >= simulatedDuration) {
        if (currentPhase < 4) {
          jumpToPhase(currentPhase + 1);
        } else {
          // 쿨다운 완료
          isPlaying = false;
          clearInterval(timerInterval);
          AudioManager.speak("모든 맞춤형 운동 프로그램이 완료되었습니다. 운동 완료하기를 눌러주세요.");
        }
        return;
      }

      updateUI();
      runPhaseGameTick();
    }, 1000);
  }

  function updateUI() {
    // 1. 남은 시간 타이머 업데이트
    const remaining = Math.max(0, simulatedDuration - phaseElapsed);
    const min = Math.floor(remaining / 60);
    const sec = String(remaining % 60).padStart(2, '0');
    timerDisplay.textContent = `${min}:${sec}`;

    // 2. 프로그레스 바 너비 설정
    const pct = (phaseElapsed / simulatedDuration) * 100;
    progressBar.style.width = `${pct}%`;

    // 3. 오디오 슬라이더 시간 동기화
    if (bgmAudio && bgmAudio.duration && !(currentPhase === 1 && (currentType === 'A' || currentType === 'B'))) {
      // 실제 오디오 타임 기준 동기화
      timeSeekSlider.max = Math.floor(bgmAudio.duration);
      timeSeekSlider.value = Math.floor(bgmAudio.currentTime);
      
      const curM = Math.floor(bgmAudio.currentTime / 60);
      const curS = String(Math.floor(bgmAudio.currentTime % 60)).padStart(2, '0');
      const durM = Math.floor(bgmAudio.duration / 60);
      const durS = String(Math.floor(bgmAudio.duration % 60)).padStart(2, '0');
      seekTimeDisplay.textContent = `${curM}:${curS} / ${durM}:${durS}`;
    } else {
      // BGM이 없는 3단계 등 또는 30초 대기가 있는 A유형 1단계에서는 가상 진행 타임코드를 슬라이더에 표시
      timeSeekSlider.max = simulatedDuration;
      timeSeekSlider.value = phaseElapsed;
      
      const curM = Math.floor(phaseElapsed / 60);
      const curS = String(phaseElapsed % 60).padStart(2, '0');
      const durM = Math.floor(simulatedDuration / 60);
      const durS = String(simulatedDuration % 60).padStart(2, '0');
      seekTimeDisplay.textContent = `${curM}:${curS} / ${durM}:${durS}`;
    }
  }

  // 5. 단계 및 유형 선택 제어기
  window.setExerciseType = function(type) {
    currentType = type;
    
    // 버튼 스타일 업데이트
    ['A', 'B', 'C', 'D'].forEach(t => {
      const btn = document.getElementById(`btnType${t}`);
      if (btn) btn.classList.toggle('active', t === type);
    });

    console.log(`유형 변경: ${type}유형`);
    
    // 강제 동기화 호출
    jumpToPhase(currentPhase, true);
  };

  // 단계 점프
  let phaseGameData = {};
  let waitingSensorAction = null;
  let calibrationPercent = 0;

  window.jumpToPhase = function(phase, force = false) {
    if (currentPhase !== phase || force) {
      currentPhase = phase;
      phaseElapsed = 0;
      simulatedDuration = PhaseDurations[phase];
      
      // 단계 버튼 활성화 스타일
      for(let i=1; i<=4; i++) {
        const btn = document.getElementById(`btnPhase${i}`);
        if (btn) btn.classList.toggle('active', i === phase);
      }

      // 단계 텍스트 변경
      const phaseNames = {
        1: "1단계: 워밍업 (3분)",
        2: "2단계: 신체강화 (5분)",
        3: "3단계: 본운동 (9분)",
        4: "4단계: 쿨다운 (3분)"
      };
      const phaseDescs = {
        1: "가벼운 제자리 걷기 및 관절 돌리기 (음악 리듬에 맞춰 진행)",
        2: "인지 부하 없이 정확한 자세로 FAME/오타고 근력·균형 운동 수행",
        3: "신체 운동 + 센서 기반 인지 과제 결합 [핵심 구간]",
        4: "심호흡 및 정적 스트레칭 진행 (안정화 밸런스 측정)"
      };
      
      phaseNameLabel.textContent = phaseNames[phase];
      phaseDescLabel.textContent = phaseDescs[phase];

      // BGM 및 TTS 초기화
      const proto = ExerciseProtocols[currentType];
      if (phase === 1) {
        SensorBridge.startCalibration();
        if (currentType === 'A' || currentType === 'B' || currentType === 'C' || currentType === 'D') {
          // A/B/C/D유형 1단계는 안내 동안 오디오를 일시정지하고 대기
          if (bgmAudio) {
            bgmAudio.pause();
            bgmAudio.removeAttribute('src'); // 오디오 소스 제거로 동시 재생 원천 방지
            bgmAudio.currentTime = 0;
          }
          const waitSec = currentType === 'D' ? 60 : 30;
          bgmPlayingText.textContent = `대기 중: ${waitSec}초 안내 후 재생 시작 (${proto.bgm1})`;
        } else {
          AudioManager.playBgm(proto.bgm1, 0.4);
        }
        initPhase1State();
      } else if (phase === 2) {
        SensorBridge.startMeasurement("strength");
        AudioManager.playBgm(proto.bgm2, proto.vol2);
        initPhase2State();
      } else if (phase === 3) {
        SensorBridge.startMeasurement("dual");
        AudioManager.playBgm(proto.bgm3, proto.vol3);
        initPhase3State();
      } else if (phase === 4) {
        SensorBridge.stopMeasurement();
        AudioManager.playBgm(proto.bgm4, 0.4);
        initPhase4State();
      }

      updateUI();
    }
  };

  // 슬라이더 탐색 처리
  window.onSeekBarScrub = function(val) {
    const scrubVal = parseFloat(val);
    if (currentPhase === 1 && (currentType === 'A' || currentType === 'B' || currentType === 'D')) {
      phaseElapsed = Math.floor(scrubVal);
      const limit = currentType === 'D' ? 60 : 30;
      if (phaseElapsed < limit) {
        if (bgmAudio) {
          bgmAudio.pause();
          bgmAudio.currentTime = 0;
        }
      } else {
        // 1. 안내 TTS가 재생 중이었다면 일시정지 및 취소
        if (activeTtsAudio) {
          try { activeTtsAudio.pause(); } catch(e){}
          activeTtsAudio = null;
        }
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
        }
        phaseGameData.welcomeSpeaking = false;

        // 2. BGM의 소스가 비어있거나 소실된 경우 소스 재할당
        const proto = ExerciseProtocols[currentType];
        if (bgmAudio) {
          const expectedSrc = `/static/audio/${proto.bgm1}`;
          if (!bgmAudio.src || bgmAudio.src.indexOf(encodeURIComponent(proto.bgm1)) === -1) {
            bgmAudio.src = expectedSrc;
            bgmAudio.volume = 0.4;
            bgmPlayingText.textContent = `재생 중인 배경음: ${proto.bgm1}`;
          }
          bgmAudio.currentTime = phaseElapsed - limit;
          if (isPlaying) {
            bgmAudio.play().catch(e => console.warn(e));
          }
        }
      }
    } else {
      if (bgmAudio && bgmAudio.duration) {
        bgmAudio.currentTime = scrubVal;
        phaseElapsed = Math.floor(scrubVal);
      } else {
        phaseElapsed = Math.floor(scrubVal);
      }
    }
    updateUI();
  };

  // 6. 단계별 가이드 시나리오 설정
  function initPhase1State() {
    sensorBadge.textContent = "워밍업";
    calibrationPercent = 0;
    
    let tts = "";
    if (currentType === 'A' || currentType === 'B') {
      tts = "안녕하세요. 오늘도 함께 즐겁게 운동을 시작해 보겠습니다. 첫 번째 운동은 노래를 들으며 움직이는 운동입니다. 노래에서 멈추세요라고 하면 멈추고, 오른쪽, 왼쪽이라는 말이 나오면 그 방향으로 움직이면 됩니다. 즐겁게 시작해 보겠습니다!";
      sensorCueBox.innerHTML = "안내 TTS가 진행 중입니다. 안내가 끝나면 BGM이 재생됩니다.";
      waitingSensorAction = "p1_bgm_lyrics_match";
      phaseGameData.welcomeSpeaking = true;
      
      AudioManager.speak(tts, () => {
        if (currentPhase === 1 && (currentType === 'A' || currentType === 'B') && isPlaying) {
          phaseGameData.welcomeSpeaking = false;
          phaseElapsed = 30;
          const proto = ExerciseProtocols[currentType];
          if (bgmAudio) {
            bgmAudio.src = `/static/audio/${proto.bgm1}`;
            bgmAudio.volume = 0.4;
            bgmAudio.currentTime = 0;
            bgmPlayingText.textContent = `재생 중인 배경음: ${proto.bgm1}`;
            bgmAudio.play().catch(e => console.warn(e));
          }
        }
      });
    }
    else if (currentType === 'D') {
      tts = "안녕하세요. 오늘도 함께 운동을 시작하겠습니다. 먼저, 보호자분께서는 어르신 곁에 함께 있어 주세요. 운동하는 동안 어르신이 안전하게 움직일 수 있도록 가까이에서 살펴봐 주시기 바랍니다. 어르신께서는 등을 기대지 않고 의자에 깊숙이 편안하게 앉아 주세요. 두 발은 바닥에 편안하게 놓고, 몸이 흔들리지 않는지 확인하겠습니다. 잠시 후 노래가 시작됩니다. 노래를 들으면서 천천히 따라 움직여 보겠습니다. 노래에서 오른쪽이라고 들리면 몸을 오른쪽으로 천천히 움직여 주세요. 왼쪽이라고 들리면 몸을 왼쪽으로 천천히 움직여 주세요. 혹시 몸이 불편하거나 어지러운 느낌이 들면 바로 움직임을 멈춰 주세요. 보호자분께서도 함께 쉬도록 도와주시기 바랍니다. 준비가 되셨다면 노래를 들으며 함께 움직여 보겠습니다.";
      sensorCueBox.innerHTML = "안내 TTS가 진행 중입니다. 안내가 끝나면 BGM이 재생됩니다.";
      waitingSensorAction = "p1_bgm_lyrics_match";
      phaseGameData.welcomeSpeaking = true;
      
      AudioManager.speak(tts, () => {
        if (currentPhase === 1 && currentType === 'D' && isPlaying) {
          phaseGameData.welcomeSpeaking = false;
          phaseElapsed = 60;
          const proto = ExerciseProtocols[currentType];
          if (bgmAudio) {
            bgmAudio.src = `/static/audio/${proto.bgm1}`;
            bgmAudio.volume = 0.4;
            bgmAudio.currentTime = 0;
            bgmPlayingText.textContent = `재생 중인 배경음: ${proto.bgm1}`;
            bgmAudio.play().catch(e => console.warn(e));
          }
        }
      });
    }
    else if (currentType === 'C') {
      tts = "안녕하세요. 오늘도 함께 운동을 시작하겠습니다. 이번 운동은 의자에 앉아 노래를 들으며 몸을 천천히 움직이는 시간입니다. 먼저 의자에 앉아주세요. 노래에서 오른쪽이라고 들리면 몸을 오른쪽으로 움직여 주세요. 왼쪽이라고 들리면 몸을 왼쪽으로 움직여 주세요. 편안한 범위에서 천천히 따라오시면 됩니다. 시작합니다.";
      sensorCueBox.innerHTML = "안내 TTS가 진행 중입니다. 안내가 끝나면 BGM이 재생됩니다.";
      waitingSensorAction = "p1_bgm_lyrics_match";
      phaseGameData.welcomeSpeaking = true;
      
      AudioManager.speak(tts, () => {
        if (currentPhase === 1 && currentType === 'C' && isPlaying) {
          phaseGameData.welcomeSpeaking = false;
          phaseElapsed = 30;
          const proto = ExerciseProtocols[currentType];
          if (bgmAudio) {
            bgmAudio.src = `/static/audio/${proto.bgm1}`;
            bgmAudio.volume = 0.4;
            bgmAudio.currentTime = 0;
            bgmPlayingText.textContent = `재생 중인 배경음: ${proto.bgm1}`;
            bgmAudio.play().catch(e => console.warn(e));
          }
        }
      });
    }
  }

  function initPhase2State() {
    sensorBadge.textContent = "자세 집중";
    waitingSensorAction = null;
    phaseGameData = { welcomeSpeaking: false, spokenKeys: {} };
    let tts = "";
    
    if (currentType === 'A') {
      sensorCueBox.textContent = "두 번째 운동을 시작하겠습니다. 이번에는 자세에 집중하며 다리의 힘과 균형을 기르는 시간입니다. 제가 안내하는 동작을 하나씩 천천히 따라 해 주세요. 속도는 중요하지 않습니다. 정확하고 편안하게 움직이는 것이 가장 중요합니다. 무리하지 않는 범위에서 함께 운동해 보겠습니다.";
      if (isPlaying) {
        AudioManager.speak("두 번째 운동을 시작하겠습니다. 이번에는 자세에 집중하며 다리의 힘과 균형을 기르는 시간입니다. 제가 안내하는 동작을 하나씩 천천히 따라 해 주세요. 속도는 중요하지 않습니다. 정확하고 편안하게 움직이는 것이 가장 중요합니다. 무리하지 않는 범위에서 함께 운동해 보겠습니다.");
      }
      return;
    }
    
    if (currentType === 'B') {
      sensorCueBox.textContent = "두 번째 운동을 시작하겠습니다. 이번에는 옆으로 천천히 걸으며 다리의 힘과 균형을 기르는 운동입니다. 발을 너무 크게 내딛기보다는 안정적인 자세를 유지하는 것이 중요합니다. 제가 안내하는 속도에 맞춰 편안하게 따라와 주세요. 준비되셨다면 함께 시작하겠습니다.";
      if (isPlaying) {
        AudioManager.speak("두 번째 운동을 시작하겠습니다. 이번에는 옆으로 천천히 걸으며 다리의 힘과 균형을 기르는 운동입니다. 발을 너무 크게 내딛기보다는 안정적인 자세를 유지하는 것이 중요합니다. 제가 안내하는 속도에 맞춰 편안하게 따라와 주세요. 준비되셨다면 함께 시작하겠습니다.");
      }
      return;
    }
    
    if (currentType === 'C') {
      sensorCueBox.textContent = "두 번째 운동을 시작하겠습니다. 이번 운동은 의자에 앉은 상태에서 다리의 힘을 기르는 운동입니다. 의자는 흔들리지 않는 안정된 의자를 사용해 주세요. 엉덩이를 의자 안쪽까지 깊숙이 앉고 등을 곧게 펴겠습니다. 준비가 되셨다면 천천히 시작하겠습니다.";
      phaseGameData.welcomeSpeaking = true;
      if (isPlaying) {
        AudioManager.speak("두 번째 운동을 시작하겠습니다. 이번 운동은 의자에 앉은 상태에서 다리의 힘을 기르는 운동입니다. 의자는 흔들리지 않는 안정된 의자를 사용해 주세요. 엉덩이를 의자 안쪽까지 깊숙이 앉고 등을 곧게 펴겠습니다. 준비가 되셨다면 천천히 시작하겠습니다.", () => {
          phaseGameData.welcomeSpeaking = false;
        });
      } else {
        phaseGameData.welcomeSpeaking = false;
      }
      return;
    } else if (currentType === 'D') {
      sensorCueBox.textContent = "자, 어르신! 음악 좋지요? 두 번째 운동을 시작하겠습니다. 의자에 편안하게 앉아주세요. 신나는 음악에 맞춰 엉덩이를 들썩들썩 해볼까요? 크게 움직이지 않아도 괜찮습니다. 준비되셨나요? 그럼 시작해 보겠습니다!";
      phaseGameData.welcomeSpeaking = true;
      if (isPlaying) {
        AudioManager.speak("자, 어르신! 음악 좋지요? 두 번째 운동을 시작하겠습니다. 의자에 편안하게 앉아주세요. 신나는 음악에 맞춰 엉덩이를 들썩들썩 해볼까요? 크게 움직이지 않아도 괜찮습니다. 준비되셨나요? 그럼 시작해 보겠습니다!", () => {
          phaseGameData.welcomeSpeaking = false;
        });
      } else {
        phaseGameData.welcomeSpeaking = false;
      }
      return;
    }
  }

  function initPhase3State() {
    sensorBadge.textContent = "본운동 (듀얼과제)";
    waitingSensorAction = "p3_game";
    
    if (currentType === 'A') {
      phaseGameData = {
        subIndex: -1,
        subTick: 0
      };
      // runPhaseGameTick()에서 첫 프레임부터 단계 설정 및 발성
    } 
    else if (currentType === 'B') {
      phaseGameData = {
        subIndex: -1,
        subTick: 0
      };
      // runPhaseGameTick()에서 첫 프레임부터 단계 설정 및 발성
    } 
    else if (currentType === 'C') {
      phaseGameData = {
        subIndex: -1,
        subTick: 0
      };
      // runPhaseGameTick()에서 첫 프레임부터 단계 설정 및 발성
    } 
    else if (currentType === 'D') {
      phaseGameData = {
        subIndex: -1,
        subTick: 0
      };
      // runPhaseGameTick()에서 첫 프레임부터 단계 설정 및 발성
    }
  }

  function initPhase4State() {
    sensorBadge.textContent = "정서 안정";
    waitingSensorAction = null;
    let tts = "";
    if (currentType === 'A') {
      // A유형 4단계는 BGM 가사에 멘트가 포함되어 있어 TTS 발성을 제거함
      tts = "";
      sensorCueBox.textContent = "코로 들이마시고 입으로 내쉬며 심호흡 (BGM 가사에 맞춰 진행)";
    } else if (currentType === 'B') {
      // B유형 4단계는 BGM 가사에 멘트가 포함되어 있어 TTS 발성을 제거함
      tts = "";
      sensorCueBox.textContent = "가벼운 피드백 및 기분 좋은 기지개 펴기 (BGM 가사에 맞춰 진행)";
    } else if (currentType === 'C') {
      // C유형 4단계는 BGM 가사(음악)에 대사가 들어있어 TTS 발성을 차단함
      tts = "";
      sensorCueBox.textContent = "앉아서 다리 뻗고 상체 숙이기 스트레칭 (BGM 가사에 맞춰 진행)";
    } else if (currentType === 'D') {
      // D유형 4단계는 BGM 가사에 멘트가 포함되어 있어 TTS 발성을 제거함
      tts = "";
      sensorCueBox.textContent = "오늘의 수련 100점 만점! 축하드립니다 (BGM 가사에 맞춰 진행).";
    }

    if (tts !== "") {
      AudioManager.speak(tts);
    }
  }

  // 3단계 듀얼과제 시간의 가상 시나리오 전환용
  function runPhaseGameTick() {
    if (currentPhase !== 3 || !isPlaying) return;

    if (currentType === 'A') {
      // 540초 동안 단일 이중과제 운동 진행
      const elapsed = phaseElapsed;
      
      if (phaseGameData.welcomeDone === undefined) {
        phaseGameData = {
          welcomeDone: false,
          round1IntroDone: false,
          round2IntroDone: false,
          breakDone: false,
          round3IntroDone: false,
          round4IntroDone: false,
          endingDone: false,
          currentWord: null,
          wordType: null,
          targetAction: null,
          lastWordTime: 0,
          praiseTriggered: false,
          introSpeaking: false
        };
      }
      
      // 1. 0s: 도입 안내 멘트
      if (elapsed >= 0 && elapsed < 30 && !phaseGameData.welcomeDone) {
        phaseGameData.welcomeDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "세 번째 운동을 시작하겠습니다. 제자리에서 무릎을 높이 들며 걷기를 계속해 주세요...";
        AudioManager.speak(
          "세 번째 운동을 시작하겠습니다. 이번 운동은 몸과 두뇌를 함께 사용하는 인지 운동입니다. 제자리에서 무릎을 높이 들며 걷기를 계속해 주세요. 제가 말하는 단어를 잘 듣고, 규칙에 맞게 움직이면 됩니다. 규칙은 중간에 바뀔 수도 있으니 제 안내를 잘 들어 주세요. 틀려도 괜찮습니다. 다음 동작부터 다시 따라오시면 됩니다. 준비되셨다면 함께 시작하겠습니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
      // 2. 30s: 1라운드 규칙 안내
      else if (elapsed >= 30 && elapsed < 45 && !phaseGameData.round1IntroDone) {
        phaseGameData.round1IntroDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "[규칙 1] 과일 ➔ 멈춤(정지) | 동물 ➔ 계속 걷기";
        AudioManager.speak(
          "무릎을 조금 더 높이 들어 보겠습니다. 첫 번째 규칙입니다. 과일이면 멈춥니다. 동물이면 계속 걷습니다. 시작하겠습니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
      // 3. 45s ~ 165s: 1라운드 게임 진행 (과일 = 멈춤)
      else if (elapsed >= 45 && elapsed < 165) {
        if (phaseGameData.introSpeaking) return;
        const interval = 10;
        if (elapsed - phaseGameData.lastWordTime >= interval || phaseGameData.currentWord === null) {
          phaseGameData.lastWordTime = elapsed;
          phaseGameData.praiseTriggered = false;
          triggerRandomWord("stop_fruit"); // Fruit = Stop, Animal = Walk
        }
      }
      // 4. 165s: 2라운드 규칙 안내 (동물 = 멈춤)
      else if (elapsed >= 165 && elapsed < 185 && !phaseGameData.round2IntroDone) {
        phaseGameData.round2IntroDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "[규칙 2] 과일 ➔ 계속 걷기 | 동물 ➔ 멈춤(정지)";
        AudioManager.speak(
          "이제 규칙이 바뀝니다. 이제는 과일이면 계속 걷고, 동물이면 멈춥니다. 천천히 다시 시작하겠습니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
      // 5. 185s ~ 305s: 2라운드 게임 진행
      else if (elapsed >= 185 && elapsed < 305) {
        if (phaseGameData.introSpeaking) return;
        const interval = 10;
        if (elapsed - phaseGameData.lastWordTime >= interval || phaseGameData.currentWord === null) {
          phaseGameData.lastWordTime = elapsed;
          phaseGameData.praiseTriggered = false;
          triggerRandomWord("stop_animal"); // Fruit = Walk, Animal = Stop
        }
      }
      // 6. 305s: 중간 호흡 조절 및 몸상태 확인
      else if (elapsed >= 305 && elapsed < 325 && !phaseGameData.breakDone) {
        phaseGameData.breakDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "잠시 호흡을 가다듬으며 몸 상태를 확인해 보겠습니다.";
        AudioManager.speak(
          "잘하고 계십니다. 호흡은 편안하게 이어가 주세요. 잠시 몸 상태를 확인해 보겠습니다. 힘들지는 않으신가요? 괜찮으시면 지금처럼 천천히 계속해 보겠습니다. 무리하지 않는 것이 가장 중요합니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
      // 7. 325s: 3라운드 규칙 안내 (과일 = 멈춤)
      else if (elapsed >= 325 && elapsed < 340 && !phaseGameData.round3IntroDone) {
        phaseGameData.round3IntroDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "[규칙 3] 과일 ➔ 멈춤(정지) | 동물 ➔ 계속 걷기";
        AudioManager.speak(
          "이번에는 조금 더 집중해 보겠습니다. 규칙은 다시 바뀝니다. 과일이면 멈추고, 동물이면 계속 걷습니다. 시작합니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
      // 8. 340s ~ 460s: 3라운드 게임 진행
      else if (elapsed >= 340 && elapsed < 460) {
        if (phaseGameData.introSpeaking) return;
        const interval = 10;
        if (elapsed - phaseGameData.lastWordTime >= interval || phaseGameData.currentWord === null) {
          phaseGameData.lastWordTime = elapsed;
          phaseGameData.praiseTriggered = false;
          triggerRandomWord("stop_fruit");
        }
      }
      // 9. 460s: 4라운드 규칙 안내 (동물 = 멈춤 - 고속)
      else if (elapsed >= 460 && elapsed < 475 && !phaseGameData.round4IntroDone) {
        phaseGameData.round4IntroDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "[마지막 라운드 - 고속] 과일 ➔ 계속 걷기 | 동물 ➔ 멈춤(정지)";
        AudioManager.speak(
          "마지막 라운드입니다. 이번에는 조금 더 빠르게 진행하겠습니다. 과일이면 계속 걷습니다. 동물이면 멈춥니다. 시작합니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
      // 10. 475s ~ 520s: 4라운드 게임 진행
      else if (elapsed >= 475 && elapsed < 520) {
        if (phaseGameData.introSpeaking) return;
        const interval = 7; // 고속 진행
        if (elapsed - phaseGameData.lastWordTime >= interval || phaseGameData.currentWord === null) {
          phaseGameData.lastWordTime = elapsed;
          phaseGameData.praiseTriggered = false;
          triggerRandomWord("stop_animal");
        }
      }
      // 11. 520s: 엔딩 안내 멘트
      else if (elapsed >= 520 && !phaseGameData.endingDone) {
        phaseGameData.endingDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "오늘의 이중과제 운동을 모두 마쳤습니다. 끝까지 집중해 주셔서 감사합니다.";
        AudioManager.speak(
          "오늘의 이중과제 운동을 모두 마쳤습니다. 몸을 움직이면서 규칙을 기억하는 연습을 아주 잘해 주셨습니다. 끝까지 집중해 주셔서 감사합니다. 천천히 걸음을 멈추고 편안하게 호흡해 주세요. 잠시 숨을 고른 뒤 다음 운동으로 이어가겠습니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
    }
    else if (currentType === 'B') {
      // 540초 동안 단일 방향 기억 사이드 스텝 진행
      const elapsed = phaseElapsed;
      
      if (phaseGameData.welcomeDone === undefined) {
        phaseGameData = {
          welcomeDone: false,
          round: 0,
          currentSequence: [],
          currentIndex: 0,
          waitingSensor: false,
          introSpeaking: false,
          splitMode: false,
          splitPart: 1,
          splitSequence1: [],
          splitSequence2: [],
          praiseTriggered: false,
          lastActionTime: 0,
          nextTriggerTime: 0,
          isBreakActive: false,
          endingDone: false
        };
      }

      // 대기 후 다음 문제 자동 출제 타이머 체크
      if (phaseGameData.nextTriggerTime > 0 && elapsed >= phaseGameData.nextTriggerTime) {
        phaseGameData.nextTriggerTime = 0;
        triggerNextSequence();
      }
      // 미동작 상태 타임아웃 감지 (동작당 3초 + 버퍼 3초: 2동작은 9초, 3동작은 12초 등)
      if (phaseGameData.waitingSensor && !phaseGameData.introSpeaking && phaseGameData.nextTriggerTime === 0) {
        const timeoutLimit = phaseGameData.currentSequence.length * 3 + 3;
        if (elapsed - phaseGameData.lastActionTime > timeoutLimit) {
          handleFailure();
        }
      }
      function speakSequence(seq) {
        phaseGameData.introSpeaking = true;
        phaseGameData.currentIndex = 0;
        phaseGameData.waitingSensor = true;
        phaseGameData.praiseTriggered = false;
        
        const speakText = seq.join(", ");
        sensorCueBox.innerHTML = `순서 기억: <strong style="font-size: 20px; color: #005EA8;">${seq.join(" ➔ ")}</strong><br>(입력 대기 중... 순서대로 옆으로 걸어보세요)`;
        AudioManager.speak(speakText + ".", () => {
          phaseGameData.introSpeaking = false;
          phaseGameData.lastActionTime = phaseElapsed; // 발음 종료 시점부터 타임아웃 카운트 시작
        });
      }

      function generateSequence(len) {
        const dirs = ["오른쪽", "왼쪽"];
        const seq = [];
        for (let i = 0; i < len; i++) {
          seq.push(dirs[Math.floor(Math.random() * dirs.length)]);
        }
        return seq;
      }

      function triggerNextSequence() {
        if (phaseGameData.round < 1 || phaseGameData.round > 4) return;
        const len = phaseGameData.round + 1; // Round 1: 2dirs, Round 2: 3dirs, etc.
        phaseGameData.splitMode = false;
        const seq = generateSequence(len);
        phaseGameData.currentSequence = seq;
        speakSequence(seq);
      }
      phaseGameData.triggerNextSequence = triggerNextSequence;

      // 1. 0s: 도입 안내 멘트
      if (elapsed >= 0 && elapsed < 30 && !phaseGameData.welcomeDone) {
        phaseGameData.welcomeDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "세 번째 운동을 시작하겠습니다. 방향 기억 스텝 안내 중...";
        AudioManager.speak(
          "세 번째 운동을 시작하겠습니다. 이번에는 몸과 두뇌를 함께 사용하는 운동입니다. 방금 연습한 옆으로 걷기 기억하시죠? 제가 방향을 한 번에 말씀드리면, 잘 기억하셨다가 같은 순서대로 옆으로 걸어 보겠습니다. 처음에는 두 가지 방향부터 시작하고, 점점 조금씩 길어집니다. 시작!",
          () => {
            phaseGameData.introSpeaking = false;
            phaseGameData.round = 1;
            phaseGameData.lastActionTime = phaseElapsed;
            triggerNextSequence();
          }
        );
      }
      // 2. 30s ~ 150s: 1라운드 진행 (2개 방향)
      else if (elapsed >= 30 && elapsed < 150) {
        if (phaseGameData.round !== 1 && !phaseGameData.introSpeaking && phaseGameData.nextTriggerTime === 0 && !phaseGameData.waitingSensor) {
          phaseGameData.round = 1;
          triggerNextSequence();
        }
      }
      // 3. 150s ~ 270s: 2라운드 진행 (3개 방향)
      else if (elapsed >= 150 && elapsed < 270) {
        if (phaseGameData.round !== 2) {
          phaseGameData.round = 2;
          phaseGameData.waitingSensor = false;
          phaseGameData.splitMode = false;
          phaseGameData.introSpeaking = true;
          sensorCueBox.textContent = "이번에는 세 가지 방향입니다.";
          AudioManager.speak("이번에는 세 가지 방향입니다.", () => {
            phaseGameData.introSpeaking = false;
            phaseGameData.lastActionTime = phaseElapsed;
            triggerNextSequence();
          });
        }
      }
      // 4. 270s ~ 300s: 안전을 위한 브레이크 타임
      else if (elapsed >= 270 && elapsed < 300) {
        if (!phaseGameData.isBreakActive) {
          phaseGameData.isBreakActive = true;
          phaseGameData.round = 0;
          phaseGameData.waitingSensor = false;
          phaseGameData.introSpeaking = true;
          sensorCueBox.textContent = "잠시 다리 피로도를 살피며 몸 상태를 확인하겠습니다.";
          AudioManager.speak(
            "잠시 몸 상태를 살펴보겠습니다. 다리는 괜찮으신가요? 조금 천천히 움직여도 괜찮습니다. 몸이 편안한 범위에서 계속해 보겠습니다.",
            () => { phaseGameData.introSpeaking = false; }
          );
        }
      }
      // 5. 300s ~ 420s: 3라운드 진행 (4개 방향)
      else if (elapsed >= 300 && elapsed < 420) {
        if (phaseGameData.round !== 3) {
          phaseGameData.round = 3;
          phaseGameData.waitingSensor = false;
          phaseGameData.splitMode = false;
          phaseGameData.introSpeaking = true;
          sensorCueBox.textContent = "이번에는 네 가지 방향입니다.";
          AudioManager.speak("이번에는 네 가지 방향입니다.", () => {
            phaseGameData.introSpeaking = false;
            phaseGameData.lastActionTime = phaseElapsed;
            triggerNextSequence();
          });
        }
      }
      // 6. 420s ~ 510s: 4라운드 진행 (5개 방향)
      else if (elapsed >= 420 && elapsed < 510) {
        if (phaseGameData.round !== 4) {
          phaseGameData.round = 4;
          phaseGameData.waitingSensor = false;
          phaseGameData.splitMode = false;
          phaseGameData.introSpeaking = true;
          sensorCueBox.textContent = "마지막입니다. 다섯 가지 방향입니다.";
          AudioManager.speak("마지막입니다. 이번에는 다섯 가지 방향입니다.", () => {
            phaseGameData.introSpeaking = false;
            phaseGameData.lastActionTime = phaseElapsed;
            triggerNextSequence();
          });
        }
      }
      // 7. 510s ~ 540s: 엔딩 안내 멘트
      else if (elapsed >= 510 && !phaseGameData.endingDone) {
        phaseGameData.endingDone = true;
        phaseGameData.round = 5;
        phaseGameData.waitingSensor = false;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "세 번째 이중과제 운동 완료.";
        AudioManager.speak(
          "세 번째 운동을 모두 마쳤습니다. 방향을 기억하면서 차분하게 잘 따라와 주셨습니다. 혹시 기억이 잠시 헷갈렸더라도 괜찮습니다. 이렇게 반복하는 것이 기억력과 집중력을 기르는 데 도움이 됩니다. 그럼 다음 마무리 운동으로 가겠습니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
    }
    else if (currentType === 'C') {
      // 540초 동안 단어 글자수 무릎 펴기 진행
      const elapsed = phaseElapsed;
      
      if (phaseGameData.welcomeDone === undefined) {
        phaseGameData = {
          welcomeDone: false,
          round: 0,
          currentWord: "",
          currentIndex: 0,
          target: 0,
          waitingSensor: false,
          introSpeaking: false,
          lastActionTime: 0,
          nextTriggerTime: 0,
          isBreakActive: false,
          endingDone: false,
          triggerNextSequence: null
        };
      }

      // 대기 후 다음 단어 자동 출제 타이머 체크
      if (phaseGameData.nextTriggerTime > 0 && elapsed >= phaseGameData.nextTriggerTime) {
        phaseGameData.nextTriggerTime = 0;
        triggerNextWord();
      }

      // 미동작 상태 타임아웃 감지 (글자당 3초 + 버퍼 3초)
      if (phaseGameData.waitingSensor && !phaseGameData.introSpeaking && phaseGameData.nextTriggerTime === 0) {
        const timeoutLimit = phaseGameData.target * 3 + 3;
        if (elapsed - phaseGameData.lastActionTime > timeoutLimit) {
          handleFailureC();
        }
      }

      function generateWord() {
        const list2 = ["사과", "토끼", "의자", "연필", "기차", "신발", "바다", "포도", "나비", "구두"];
        const list3 = ["비행기", "자전거", "강아지", "코끼리", "손수건", "냉장고", "토마토", "원숭이", "종이컵"];
        if (phaseGameData.round === 1) {
          return list2[Math.floor(Math.random() * list2.length)];
        } else if (phaseGameData.round === 2) {
          return list3[Math.floor(Math.random() * list3.length)];
        } else {
          // Round 3: mixed
          const combined = list2.concat(list3);
          return combined[Math.floor(Math.random() * combined.length)];
        }
      }

      function triggerNextWord() {
        if (phaseGameData.round < 1 || phaseGameData.round > 3) return;
        const word = generateWord();
        phaseGameData.currentWord = word;
        phaseGameData.target = word.length;
        phaseGameData.currentIndex = 0;
        phaseGameData.waitingSensor = true;
        phaseGameData.introSpeaking = true;

        sensorCueBox.innerHTML = `단어 제시: <strong style="font-size: 24px; color: #005EA8;">${word}</strong> (${word.length}글자)<br>(글자 수만큼 무릎을 펴주세요)`;
        AudioManager.speak(word, () => {
          phaseGameData.introSpeaking = false;
          phaseGameData.lastActionTime = phaseElapsed;
        });
      }
      phaseGameData.triggerNextSequence = triggerNextWord;

      // 1. 0s: 도입 안내 멘트
      if (elapsed >= 0 && elapsed < 30 && !phaseGameData.welcomeDone) {
        phaseGameData.welcomeDone = true;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "세 번째 운동을 시작하겠습니다. 단어 글자수 무릎 펴기 안내 중...";
        AudioManager.speak(
          "세 번째 운동입니다. 이번에는 다리 운동과 두뇌 운동을 함께 해보겠습니다. 먼저 의자에 깊숙이 앉아 주세요. 등을 곧게 펴고 두 발은 바닥에 편안하게 놓습니다. 엉덩이가 의자에서 떨어지지 않도록 유지해 주세요. 제가 단어를 말씀드리면, 단어의 글자 수를 생각한 뒤 그 숫자만큼 양쪽 무릎을 앞으로 쭉 펴고 다시 내려놓겠습니다. 시작입니다.",
          () => {
            phaseGameData.introSpeaking = false;
            phaseGameData.round = 1;
            phaseGameData.lastActionTime = phaseElapsed;
            triggerNextWord();
          }
        );
      }
      // 2. 30s ~ 180s: 1라운드 진행 (2글자 단어)
      else if (elapsed >= 30 && elapsed < 180) {
        if (phaseGameData.round !== 1 && !phaseGameData.introSpeaking && phaseGameData.nextTriggerTime === 0 && !phaseGameData.waitingSensor) {
          phaseGameData.round = 1;
          triggerNextWord();
        }
      }
      // 3. 180s ~ 330s: 2라운드 진행 (3글자 단어)
      else if (elapsed >= 180 && elapsed < 330) {
        if (phaseGameData.round !== 2) {
          phaseGameData.round = 2;
          phaseGameData.waitingSensor = false;
          phaseGameData.introSpeaking = true;
          sensorCueBox.textContent = "이번에는 세 글자 단어입니다.";
          AudioManager.speak("이번에는 세 글자 단어입니다.", () => {
            phaseGameData.introSpeaking = false;
            phaseGameData.lastActionTime = phaseElapsed;
            triggerNextWord();
          });
        }
      }
      // 4. 330s ~ 360s: 안전을 위한 브레이크 타임
      else if (elapsed >= 330 && elapsed < 360) {
        if (!phaseGameData.isBreakActive) {
          phaseGameData.isBreakActive = true;
          phaseGameData.round = 0;
          phaseGameData.waitingSensor = false;
          phaseGameData.introSpeaking = true;
          sensorCueBox.textContent = "잠시 다리 피로도를 살피며 몸 상태를 확인하겠습니다.";
          AudioManager.speak(
            "잠시 몸 상태를 살펴보겠습니다. 허리나 무릎에 불편함은 없으신가요? 힘이 드시면 잠시 쉬셔도 괜찮습니다. 괜찮으시면 다음 문제를 이어가겠습니다.",
            () => { phaseGameData.introSpeaking = false; }
          );
        }
      }
      // 5. 360s ~ 500s: 3라운드 진행 (2 & 3글자 혼합 단어)
      else if (elapsed >= 360 && elapsed < 500) {
        if (phaseGameData.round !== 3) {
          phaseGameData.round = 3;
          phaseGameData.waitingSensor = false;
          phaseGameData.introSpeaking = true;
          sensorCueBox.textContent = "마지막은 두 글자 단어와 세 글자 단어를 섞어 말하겠습니다.";
          AudioManager.speak("마지막은 두 글자 단어와 세 글자 단어를 섞어 말하겠습니다.", () => {
            phaseGameData.introSpeaking = false;
            phaseGameData.lastActionTime = phaseElapsed;
            triggerNextWord();
          });
        }
      }
      // 6. 500s ~ 540s: 엔딩 안내 멘트
      else if (elapsed >= 500 && !phaseGameData.endingDone) {
        phaseGameData.endingDone = true;
        phaseGameData.round = 4;
        phaseGameData.waitingSensor = false;
        phaseGameData.introSpeaking = true;
        sensorCueBox.textContent = "세 번째 이중과제 운동 완료.";
        AudioManager.speak(
          "세 번째 운동을 모두 마쳤습니다. 단어의 글자 수를 생각하면서 무릎을 움직이는 운동은 다리의 힘을 기르면서 집중력도 함께 사용하는 좋은 운동입니다. 바로 일어나지 마시고, 잠시 앉아서 호흡을 고르겠습니다. 이제 다음 운동으로 이어가겠습니다.",
          () => { phaseGameData.introSpeaking = false; }
        );
      }
    }
    else if (currentType === 'D') {
      const elapsed = phaseElapsed;
      
      const scriptD3 = {
        0: {
          text: "자, 어르신! 이제 오늘의 가장 재미있는 운동을 시작하겠습니다! 먼저 의자에 깊숙이 편안하게 앉아 주세요. 엉덩이는 의자에 붙인 채로 운동하겠습니다. 보호자분께서는 어르신 곁에서 함께 응원해 주세요.",
          wait: false
        },
        20: {
          text: "이번에는 상상 놀이를 해보겠습니다. 제가 '커진다!' 하면 다리를 넓게 벌려 주세요. 제가 '작아진다!' 하면 다리를 모아 주세요. 조금만 움직여도 아주 잘하시는 겁니다. 음악을 즐기면서 함께 움직여 보겠습니다!",
          wait: false
        },
        45: {
          text: "자~ 시작합니다!",
          wait: false
        },
        50: {
          text: "코끼리처럼 크다!",
          wait: true
        },
        68: {
          text: "이번에는 개미처럼 작다!",
          wait: true
        },
        86: {
          text: "코끼리처럼 커진다!",
          wait: true
        },
        104: {
          text: "다시 한번! 개미처럼 작다!",
          wait: true
        },
        122: {
          text: "코끼리처럼 커진다!",
          wait: true
        },
        140: {
          text: "다시 한번! 개미처럼 작다!",
          wait: true
        },
        170: {
          text: "이번에는 조금 더 신나게 가볼까요?",
          wait: false
        },
        180: {
          text: "풍선처럼 커진다!",
          wait: true
        },
        198: {
          text: "다시 갑니다!",
          wait: false
        },
        204: {
          text: "콩알처럼 작아진다!",
          wait: true
        },
        222: {
          text: "한번 더!",
          wait: false
        },
        228: {
          text: "산처럼 커진다!",
          wait: true
        },
        246: {
          text: "콩알처럼 작아진다!",
          wait: true
        },
        264: {
          text: "풍선처럼 커진다!",
          wait: true
        },
        282: {
          text: "콩알처럼 작아진다!",
          wait: true
        },
        310: {
          text: "잠깐 쉬어 갈게요! 숨을 천천히 쉬어 보세요. 힘들면 잠시 쉬어도 괜찮습니다. 보호자분께서는 어르신이 편안하게 앉아 계신지 한 번 확인해 주세요. 괜찮으시다면 계속 이어가겠습니다!",
          wait: false
        },
        360: {
          text: "고래처럼 커진다!",
          wait: true
        },
        378: {
          text: "새우처럼 작아진다!",
          wait: true
        },
        396: {
          text: "고래처럼 커진다!",
          wait: true
        },
        414: {
          text: "새우처럼 작아진다!",
          wait: true
        },
        432: {
          text: "마지막 세번 갑니다!",
          wait: false
        },
        440: {
          text: "고래처럼 커진다!",
          wait: true
        },
        458: {
          text: "새우처럼 작아진다!",
          wait: true
        },
        495: {
          text: "와아! 오늘 운동도 훌륭하게 마치셨습니다! 몸을 조금씩 움직인 것만으로도 아주 좋은 운동이 되었습니다. 이제는 바로 일어나지 마시고 의자에 편안하게 앉아 잠시 쉬어 주세요. 보호자분께서는 어르신의 몸 상태를 한 번 확인해 주세요. 다음은 음악에 맞춰 숨을 고르는 단계로 넘어가겠습니다.",
          wait: false
        }
      };

      if (phaseGameData.spokenKeys === undefined) {
        phaseGameData.spokenKeys = {};
      }

      Object.keys(scriptD3).forEach(kStr => {
        const k = parseInt(kStr);
        if (k <= elapsed && !phaseGameData.spokenKeys[k]) {
          phaseGameData.spokenKeys[k] = true;
          phaseGameData.holdTime = k;
          sensorCueBox.textContent = scriptD3[k].text;
          
          phaseGameData.introSpeaking = true;
          if (scriptD3[k].wait) {
            AudioManager.speak(scriptD3[k].text, () => {
              phaseGameData.introSpeaking = false;
              phaseGameData.waitingSensor = true;
            });
          } else {
            phaseGameData.introSpeaking = true;
            AudioManager.speak(scriptD3[k].text, () => {
              phaseGameData.introSpeaking = false;
              phaseGameData.waitingSensor = false;
            });
          }
        }
      });
    }
  }

  // 7. 가상 센서 클릭 및 실제 센서 이벤트 처리
  window.triggerVirtualSensor = function() {
    console.log("가상 센서 신호 유도됨.");
    
    let action = null;
    const t = bgmAudio ? bgmAudio.currentTime : 0;
    
    if (currentPhase === 1) {
      if (currentType === 'A' || currentType === 'B') {
        if (t >= 53 && t <= 56) action = 'stop';
        else if (t >= 67 && t <= 73) action = 'weight_right';
        else if (t >= 78 && t <= 81) action = 'weight_left';
        else if (t >= 116 && t <= 122) action = 'step_forward_right';
        else if (t >= 124 && t <= 125) action = 'step_backward';
      }
      else if (currentType === 'C') {
        if (t >= 55 && t <= 61) action = 'weight_right';
        else if (t >= 69 && t <= 78) action = 'weight_left';
      }
      else if (currentType === 'D') {
        if ((t >= 55 && t <= 61) || (t >= 69 && t <= 78)) action = 'any_reaction';
      }
      
      if (!action) {
        sensorCueBox.innerHTML = `<span style='color: #ef4444; font-weight: 800;'>인식 실패 ❌</span><br>가사의 동작 지시 구간이 아닙니다. (현재 BGM 시간: ${Math.floor(t)}초)`;
        return;
      }
    }
    
    let mockEvt = {};
    if (currentPhase === 1) {
      mockEvt = { type: 'motion', action: action };
    } 
    else if (currentPhase === 3) {
      if (currentType === 'A') {
        mockEvt = { type: 'motion', action: phaseGameData.targetAction || 'stop' };
      } else if (currentType === 'B') {
        const expected = phaseGameData.currentSequence && phaseGameData.currentSequence[phaseGameData.currentIndex];
        mockEvt = { type: 'motion', action: expected === '왼쪽' ? 'step_left' : 'step_right' };
      } else if (currentType === 'C') {
        mockEvt = { type: 'motion', action: 'knee_extension' };
      } else if (currentType === 'D') {
        mockEvt = { type: 'motion', action: 'any_reaction' };
      }
    } else {
      mockEvt = { type: 'motion', action: 'repetition' };
    }

    SensorBridge.onSensorEvent(JSON.stringify(mockEvt));
  };

  // A/B/C유형 1단계 실시간 BGM 시간대별 지시문 디스플레이 함수
  function updateA1Cue(t) {
    if ((currentType === 'A' || currentType === 'B') && currentPhase === 1) {
      if (t >= 53 && t <= 56) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #ef4444; font-weight: 800;'>[대기] 멈춤 (정지)</span><br>BGM 가사: '멈춰!'";
      } else if (t >= 67 && t <= 73) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #005EA8; font-weight: 800;'>[대기] 오른쪽 체중 이동</span><br>BGM 가사: '오른쪽 엉덩이로 체중을 눌러요'";
      } else if (t >= 78 && t <= 81) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #10b981; font-weight: 800;'>[대기] 왼쪽 체중 이동</span><br>BGM 가사: '왼쪽으로 누르세요'";
      } else if (t >= 116 && t <= 122) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #f59e0b; font-weight: 800;'>[대기] 오른발 앞으로 한발</span><br>BGM 가사: '오른발 앞으로 한발'";
      } else if (t >= 124 && t <= 125) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #8b5cf6; font-weight: 800;'>[대기] 뒤로 한발</span><br>BGM 가사: '이번엔 뒤로 한발'";
      } else {
        sensorBadge.textContent = "워밍업";
        sensorCueBox.innerHTML = "BGM의 가사 지시에 맞춰 움직여주세요.<br>(가사 멘트가 나오는 구간에만 센서가 작동합니다)";
      }
    }
    else if (currentType === 'C' && currentPhase === 1) {
      if (t >= 55 && t <= 61) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #005EA8; font-weight: 800;'>[대기] 오른쪽 체중 이동</span><br>BGM 가사: '오른쪽 엉덩이로 체중 누르세요'";
      } else if (t >= 69 && t <= 78) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #10b981; font-weight: 800;'>[대기] 왼쪽 체중 이동</span><br>BGM 가사: '이번엔 왼쪽 엉덩이를 눌러주세요'";
      } else {
        sensorBadge.textContent = "워밍업";
        sensorCueBox.innerHTML = "BGM의 가사 지시에 맞춰 움직여주세요.<br>(가사 멘트가 나오는 구간에만 센서가 작동합니다)";
      }
    }
    else if (currentType === 'D' && currentPhase === 1) {
      if (t >= 55 && t <= 61) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #005EA8; font-weight: 800;'>[대기] 오른쪽 체중 이동</span><br>BGM 가사: '오른쪽 엉덩이로 체중 누르세요'";
      } else if (t >= 69 && t <= 78) {
        sensorBadge.textContent = "동작 지시";
        sensorCueBox.innerHTML = "<span style='color: #10b981; font-weight: 800;'>[대기] 왼쪽 체중 이동</span><br>BGM 가사: '이번엔 왼쪽 엉덩이로 눌러주세요'";
      } else {
        sensorBadge.textContent = "워밍업";
        sensorCueBox.innerHTML = "BGM의 가사 지시에 맞춰 움직여주세요.<br>(가사 멘트가 나오는 구간에만 센서가 작동합니다)";
      }
    }
  }

  function handleSensorData(data) {
    if (!isPlaying) return;

    if (currentPhase === 1) {
      if (currentType === 'A' || currentType === 'B') {
        if (!bgmAudio) return;
        const t = bgmAudio.currentTime;
        if (t >= 53 && t <= 56) {
          if (data.action === 'stop') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 정지 (얼음)";
          }
        } else if (t >= 67 && t <= 73) {
          if (data.action === 'weight_right') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 오른쪽 체중 이동";
          }
        } else if (t >= 78 && t <= 81) {
          if (data.action === 'weight_left') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 왼쪽 체중 이동";
          }
        } else if (t >= 116 && t <= 122) {
          if (data.action === 'step_forward_right') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 오른발 앞으로 한발";
          }
        } else if (t >= 124 && t <= 125) {
          if (data.action === 'step_backward') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 뒤로 한발";
          }
        }
      }
      else if (currentType === 'C') {
        if (!bgmAudio) return;
        const t = bgmAudio.currentTime;
        if (t >= 55 && t <= 61) {
          if (data.action === 'weight_right') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 오른쪽 체중 이동";
          }
        } else if (t >= 69 && t <= 78) {
          if (data.action === 'weight_left') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 왼쪽 체중 이동";
          }
        }
      }
      else if (currentType === 'D') {
        if (!bgmAudio) return;
        const t = bgmAudio.currentTime;
        if ((t >= 55 && t <= 61) || (t >= 69 && t <= 78)) {
          if (data.action === 'any_reaction') {
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: 반응 성공";
          }
        }
      }
    } 
    else if (currentPhase === 3) {
      if (currentType === 'A') {
        // A유형 3단계는 단일 이중과제 운동 진행 (멈춤 감지 시 ding_bright.mp3 및 칭찬 송출)
        if (phaseGameData.targetAction === 'stop' && data.action === 'stop') {
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>정답 인식 성공! 🔔</span><br>단어: ${phaseGameData.currentWord} (동작: 멈춤)`;
          
          // 칭찬 멘트 무작위 송출 (30% 확률)
          if (!phaseGameData.praiseTriggered && Math.random() < 0.3) {
            phaseGameData.praiseTriggered = true;
            const praises = ["아주 좋습니다.", "잘하셨습니다.", "멋집니다!"];
            const p = praises[Math.floor(Math.random() * praises.length)];
            setTimeout(() => {
              if (currentPhase === 3 && isPlaying) {
                AudioManager.speak(p);
              }
            }, 1200);
          }
        }
      }
      else if (currentType === 'B') {
        // B유형 3단계: 방향 기억 연속 사이드 스텝
        if (!phaseGameData.waitingSensor || phaseGameData.introSpeaking) return;
        
        const expected = phaseGameData.currentSequence[phaseGameData.currentIndex];
        let actionMatched = false;
        
        if (expected === "오른쪽" && data.action === "step_right") {
          actionMatched = true;
        } else if (expected === "왼쪽" && data.action === "step_left") {
          actionMatched = true;
        }
        
        if (actionMatched) {
          phaseGameData.currentIndex++;
          sensorCueBox.innerHTML = `순서 일치 진행 중: <span style="color: #059669; font-weight: 800;">${phaseGameData.currentIndex} / ${phaseGameData.currentSequence.length} 성공</span>`;
          
          if (phaseGameData.currentIndex === phaseGameData.currentSequence.length) {
            // 전체 시퀀스 매칭 성공!
            AudioManager.playEffect('ding_bright.mp3');
            phaseGameData.waitingSensor = false;
            
            if (phaseGameData.splitMode && phaseGameData.splitPart === 1) {
              // 분할 모드 1부 성공 시 2부로 바로 연결
              sensorCueBox.textContent = "아주 잘하셨습니다! 이어서 다음 두 방향입니다.";
              setTimeout(() => {
                if (currentPhase === 3 && isPlaying) {
                  phaseGameData.splitPart = 2;
                  phaseGameData.currentSequence = phaseGameData.splitSequence2;
                  phaseGameData.currentIndex = 0;
                  phaseGameData.waitingSensor = true;
                  sensorCueBox.innerHTML = `이어서 다음 두 방향: <strong style="font-size: 20px; color: #005EA8;">${phaseGameData.splitSequence2.join(" ➔ ")}</strong>`;
                  AudioManager.speak("이어서 다음 두 방향입니다. " + phaseGameData.splitSequence2.join(", ") + ".", () => {
                    phaseGameData.lastActionTime = phaseElapsed;
                  });
                }
              }, 2000);
            } else {
              // 일반 성공 혹은 분할 모드 2부 최종 성공
              phaseGameData.splitMode = false;
              sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>전체 순서 기억 성공! 🔔</span>";
              
              const praises = ["네, 잘하셨습니다.", "잘 기억하고 계십니다.", "아주 훌륭합니다!", "동작이 정확하십니다."];
              const p = praises[Math.floor(Math.random() * praises.length)];
              
              AudioManager.speak(p, () => {
                if (currentPhase === 3 && isPlaying) {
                  setTimeout(() => {
                    if (currentPhase === 3 && isPlaying && phaseGameData.triggerNextSequence) {
                      phaseGameData.triggerNextSequence();
                    }
                  }, 1500);
                }
              });
            }
          }
        } else {
          handleFailure();
        }
      }
      else if (currentType === 'C') {
        // C유형 3단계: 단어 글자수 무릎 앞으로 펴기
        if (!phaseGameData.waitingSensor || phaseGameData.introSpeaking) return;

        if (data.action === "knee_extension") {
          phaseGameData.currentIndex++;
          sensorCueBox.innerHTML = `무릎 펴기 진행 중: <span style="color: #059669; font-weight: 800;">${phaseGameData.currentIndex} / ${phaseGameData.target}회</span>`;

          if (phaseGameData.currentIndex === phaseGameData.target) {
            // 정답 완료!
            phaseGameData.waitingSensor = false;
            AudioManager.playEffect('ding_bright.mp3');
            sensorCueBox.innerHTML = `<span style="color: #059669; font-weight: 800;">성공! 🔔</span><br>단어: ${phaseGameData.currentWord} (${phaseGameData.target}글자)`;
            
            // 50% 확률로 칭찬 멘트 송출
            if (Math.random() < 0.5) {
              const praises = ["잘하셨습니다.", "맞아요.", "정확합니다.", "아주 좋습니다."];
              const p = praises[Math.floor(Math.random() * praises.length)];
              AudioManager.speak(p, () => {
                if (currentPhase === 3 && isPlaying) {
                  setTimeout(() => {
                    if (currentPhase === 3 && isPlaying && phaseGameData.triggerNextSequence) {
                      phaseGameData.triggerNextSequence();
                    }
                  }, 1500);
                }
              });
            } else {
              // 칭찬이 없을 경우에도 2초 뒤 자동 다음 단어 출제
              setTimeout(() => {
                if (currentPhase === 3 && isPlaying && phaseGameData.triggerNextSequence) {
                  phaseGameData.triggerNextSequence();
                }
              }, 2000);
            }
          }
        }
      }
      else if (currentType === 'D') {
        if (phaseGameData.waitingSensor && !phaseGameData.introSpeaking) {
          phaseGameData.waitingSensor = false;
          AudioManager.playEffect('applause.mp3');
          
          const praises = [
            "와! 정말 최고예요!",
            "아주 잘하셨습니다!",
            "아주 잘 움직이고 계십니다!",
            "대단하십니다! 멋진 움직임이에요!",
            "최고예요! 아주 훌륭한 동작입니다.",
            "박자를 맞춰서 참 잘하십니다!"
          ];
          const chosenPraise = praises[Math.floor(Math.random() * praises.length)];
          sensorCueBox.innerHTML = `<span style="color: #059669; font-weight: 800;">성공! 👏</span><br>${chosenPraise}`;
          AudioManager.speak(chosenPraise);
        }
      }
    }
  }

  // A유형 3단계 랜덤 단어 출제 및 속성 판정 함수
  function triggerRandomWord(mode) {
    const fruits = ["포도", "사과", "귤", "복숭아", "참외"];
    const animals = ["사자", "토끼", "코끼리", "기린", "호랑이", "고양이", "강아지"];
    const wordsPool = fruits.concat(animals);
    const w = wordsPool[Math.floor(Math.random() * wordsPool.length)];
    
    phaseGameData.currentWord = w;
    const isFruit = fruits.includes(w);
    phaseGameData.wordType = isFruit ? "fruit" : "animal";
    
    if (mode === "stop_fruit") {
      phaseGameData.targetAction = isFruit ? "stop" : "walk";
    } else {
      phaseGameData.targetAction = isFruit ? "walk" : "stop";
    }
    
    const ruleDesc = mode === "stop_fruit" ? "과일 ➔ 멈춤 | 동물 ➔ 걷기" : "과일 ➔ 걷기 | 동물 ➔ 멈춤";
    sensorCueBox.innerHTML = `단어: <strong style="font-size: 20px; color: #005EA8;">${w}</strong><br>규칙: ${ruleDesc} (정답 동작: ${phaseGameData.targetAction === 'stop' ? '멈춤 🛑' : '걷기 🏃'})`;
    AudioManager.speak(w);
  }

  // B유형 3단계 전용 실패/타임아웃 통합 피드백 함수
  function handleFailure() {
    phaseGameData.waitingSensor = false;
    
    if (currentPhase === 3 && phaseGameData.round === 3 && !phaseGameData.splitMode) {
      // Round 3에서 실패한 경우 처음 두 방향 분할 모드 작동
      phaseGameData.splitMode = true;
      phaseGameData.splitPart = 1;
      const seq = phaseGameData.currentSequence;
      phaseGameData.splitSequence1 = [seq[0], seq[1]];
      phaseGameData.splitSequence2 = [seq[2], seq[3]];
      phaseGameData.currentSequence = phaseGameData.splitSequence1;
      phaseGameData.currentIndex = 0;
      
      sensorCueBox.textContent = "틀렸습니다. 두 개씩 나누어 연습하겠습니다.";
      AudioManager.speak("괜찮습니다. 이번에는 처음 두 방향부터 함께 해보겠습니다. " + phaseGameData.splitSequence1.join(", ") + ".", () => {
        phaseGameData.waitingSensor = true;
        phaseGameData.lastActionTime = phaseElapsed;
      });
    } else {
      // 일반 오류 격려 멘트 다양화 (어이쿠 연발 방지)
      sensorCueBox.innerHTML = "<span style='color: #ef4444; font-weight: 800;'>동작 오류/미감지 ❌</span><br>다음 문제로 넘어갑니다.";
      
      const errors = [
        "어이쿠, 방향이 살짝 헷갈리셨죠? 괜찮습니다! 저랑 같이 다시 해볼까요?",
        "괜찮습니다! 다음 방향으로 갈까요?",
        "괜찮습니다. 차분하게 다음 동작을 따라 해 보세요.",
        "방향이 헷갈리셔도 괜찮습니다. 천천히 다시 시작해 볼게요."
      ];
      const errText = errors[Math.floor(Math.random() * errors.length)];
      AudioManager.speak(errText, () => {
        if (currentPhase === 3 && isPlaying) {
          setTimeout(() => {
            if (currentPhase === 3 && isPlaying && phaseGameData.triggerNextSequence) {
              phaseGameData.triggerNextSequence();
            }
          }, 1500);
        }
      });
    }
  }

  // C유형 3단계 전용 실패/타임아웃 통합 피드백 함수
  function handleFailureC() {
    phaseGameData.waitingSensor = false;
    
    const w = phaseGameData.currentWord;
    const len = phaseGameData.target;
    let lenStr = len === 2 ? "두" : "세";
    
    const errors = [
      {
        type: "retry",
        text: `괜찮습니다. 다시 천천히 해보겠습니다. ${w}. ${lenStr} 글자입니다. 무릎을 ${lenStr} 번 천천히 펴겠습니다.`
      },
      {
        type: "retry",
        text: `괜찮습니다. 다시 해볼까요? ${w}. ${lenStr} 번 천천히 펴겠습니다.`
      },
      {
        type: "retry",
        text: `다시 해보겠습니다. ${w}.`
      },
      {
        type: "skip",
        text: "괜찮아요. 다음 단어로 넘어가겠습니다."
      },
      {
        type: "skip",
        text: "다음 단어를 말하겠습니다."
      }
    ];
    
    const chosenError = errors[Math.floor(Math.random() * errors.length)];
    sensorCueBox.innerHTML = `<span style='color: #ef4444; font-weight: 800;'>동작 오류/미감지 ❌</span><br>${chosenError.type === 'skip' ? '다음 단어로 이동' : '재시도 대기 중'}`;
    
    AudioManager.speak(chosenError.text, () => {
      if (currentPhase === 3 && isPlaying) {
        if (chosenError.type === "retry") {
          phaseGameData.currentIndex = 0;
          phaseGameData.waitingSensor = true;
          phaseGameData.lastActionTime = phaseElapsed;
        } else {
          setTimeout(() => {
            if (currentPhase === 3 && isPlaying && phaseGameData.triggerNextSequence) {
              phaseGameData.triggerNextSequence();
            }
          }, 1500);
        }
      }
    });
  }

  // 8. 초기 구동 및 제스처 스타터
  function init() {
    
    // BGM 오디오 재생 시간 업데이트 리스너 바인딩
    if (bgmAudio) {
      bgmAudio.addEventListener('play', () => {
        if (bgmAudio.isDucked && bgmAudio.duckedVolume !== undefined) {
          bgmAudio.volume = bgmAudio.duckedVolume;
        }
      });
      bgmAudio.addEventListener('playing', () => {
        if (bgmAudio.isDucked && bgmAudio.duckedVolume !== undefined) {
          bgmAudio.volume = bgmAudio.duckedVolume;
        }
      });
      bgmAudio.addEventListener('loadedmetadata', () => {
        if (bgmAudio.isDucked && bgmAudio.duckedVolume !== undefined) {
          bgmAudio.volume = bgmAudio.duckedVolume;
        }
      });
      bgmAudio.addEventListener('volumechange', () => {
        if (bgmAudio.isDucked && bgmAudio.duckedVolume !== undefined && bgmAudio.volume !== bgmAudio.duckedVolume) {
          bgmAudio.volume = bgmAudio.duckedVolume;
        }
      });

      bgmAudio.ontimeupdate = () => {
        if (isPlaying) {
          const proto = ExerciseProtocols[currentType];
          const hasBgmInCurrentPhase = (currentPhase === 1 && proto.bgm1) ||
                                       (currentPhase === 2 && proto.bgm2) ||
                                       (currentPhase === 3 && proto.bgm3) ||
                                       (currentPhase === 4 && proto.bgm4);
          
          // D유형 3단계는 BGM을 게임 로직(얼음땡)에 의해 일시정지/재생하므로 시간을 동기화하지 않음
          // BGM이 실제로 재생 중(paused가 아님)일 때만 시간을 동기화해 레이스 컨디션을 방지
          if (hasBgmInCurrentPhase && bgmAudio.duration && !bgmAudio.paused && !(currentType === 'D' && currentPhase === 3)) {
            if (currentPhase === 1) {
              if (currentType === 'A' || currentType === 'B' || currentType === 'C') {
                phaseElapsed = Math.floor(bgmAudio.currentTime) + 30;
              } else if (currentType === 'D') {
                phaseElapsed = Math.floor(bgmAudio.currentTime) + 60;
              }
            }
          }
          if (currentPhase === 1 && (currentType === 'A' || currentType === 'B' || currentType === 'C' || currentType === 'D')) {
            updateA1Cue(bgmAudio.currentTime);
          }
          updateUI();
        }
      };
    }

    // Flask 데이터 기반 유형 설정
    const flaskType = window.EXERCISE_DATA.type || 'C유형 맞춤';
    let type = 'C';
    if (flaskType.includes('A')) type = 'A';
    else if (flaskType.includes('B')) type = 'B';
    else if (flaskType.includes('C')) type = 'C';
    else if (flaskType.includes('D')) type = 'D';

    setExerciseType(type);
    jumpToPhase(1, true);

    // 사용자 제스처 스타터 (자동 재생 제한 우회용)
    const gestureStarter = () => {
      if (!isPlaying) {
        isPlaying = true;
        motionPulse.classList.add('moving');
        jumpToPhase(currentPhase, true);
        startTimer();
        console.log("제스처에 의한 운동 루프 시작");
      }
      window.removeEventListener('click', gestureStarter);
      window.removeEventListener('touchstart', gestureStarter);
    };
    window.addEventListener('click', gestureStarter);
    window.addEventListener('touchstart', gestureStarter);
  }

  window.addEventListener('DOMContentLoaded', init);

})();
