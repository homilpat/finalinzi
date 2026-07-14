(function () {
  'use strict';

  // 1. 상태 변수
  let currentType = 'A'; // A, B, C, D (유형)
  let currentPhase = 1;  // 1, 2, 3, 4 (단계)
  let phaseElapsed = 0;  // 현재 단계에서 경과한 시간 (초)
  let isPlaying = false; // 재생 상태
  let timerInterval = null;
  let simulatedDuration = 180; // 현재 단계의 전체 초 (1단계: 180초, 2단계: 300초, 3단계: 540초, 4단계: 180초)

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
      vol2: 0.60, vol3: 0.40 // BGM 크게 유지
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
      bgmPlayingText.textContent = `재생 중인 배경음: ${file}`;
      
      if (isPlaying) {
        bgmAudio.play().catch(e => console.warn("BGM 재생 차단됨:", e));
      }
    },
    playEffect: function(file) {
      if (!effectAudio) return;
      effectAudio.src = `/static/audio/${file}`;
      effectAudio.play().catch(e => console.warn("효과음 재생 차단됨:", e));
    },
    speak: function(text, onEnd) {
      if (!isPlaying) return;
      console.log("Speak:", text);
      const googleTtsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&tl=ko&client=tw-ob&q=${encodeURIComponent(text)}`;
      const originalVolume = bgmAudio ? bgmAudio.volume : 0.5;
      
      const tempAudio = new Audio();
      tempAudio.referrerPolicy = "no-referrer";
      tempAudio.src = googleTtsUrl;
      
      if (bgmAudio) bgmAudio.volume = originalVolume * 0.25; // Duck BGM
      
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
            if (bgmAudio) bgmAudio.volume = originalVolume;
            if (onEnd) onEnd();
          };
          utterance.onerror = () => {
            if (bgmAudio) bgmAudio.volume = originalVolume;
            if (onEnd) onEnd();
          };
          window.speechSynthesis.speak(utterance);
        } else {
          if (bgmAudio) bgmAudio.volume = originalVolume;
          if (onEnd) onEnd();
        }
      };

      tempAudio.onerror = runFallback;
      tempAudio.play().then(() => {
        tempAudio.onended = () => {
          if (bgmAudio) bgmAudio.volume = originalVolume;
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
    if (bgmAudio && bgmAudio.duration) {
      // 실제 오디오 타임 기준 동기화
      timeSeekSlider.max = Math.floor(bgmAudio.duration);
      timeSeekSlider.value = Math.floor(bgmAudio.currentTime);
      
      const curM = Math.floor(bgmAudio.currentTime / 60);
      const curS = String(Math.floor(bgmAudio.currentTime % 60)).padStart(2, '0');
      const durM = Math.floor(bgmAudio.duration / 60);
      const durS = String(Math.floor(bgmAudio.duration % 60)).padStart(2, '0');
      seekTimeDisplay.textContent = `${curM}:${curS} / ${durM}:${durS}`;
    } else {
      // BGM이 없는 3단계 등에서는 가상 진행 타임코드를 슬라이더에 표시
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
        AudioManager.playBgm(proto.bgm1, 0.4);
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
    if (bgmAudio && bgmAudio.duration) {
      bgmAudio.currentTime = scrubVal;
      phaseElapsed = Math.floor(scrubVal);
    } else {
      phaseElapsed = Math.floor(scrubVal);
    }
    updateUI();
  };

  // 6. 단계별 가이드 시나리오 설정
  function initPhase1State() {
    sensorBadge.textContent = "워밍업";
    calibrationPercent = 0;
    
    let tts = "";
    if (currentType === 'A' || currentType === 'B' || currentType === 'C') {
      // A/B/C유형 1단계는 BGM 가사에 지시가 포함되어 있으므로 TTS를 발성하지 않음
      tts = "";
      sensorCueBox.innerHTML = "BGM의 가사 지시에 맞춰 움직여주세요.<br>(동작 가사 구간에 맞춰 가상 버튼을 눌러보세요)";
      waitingSensorAction = "p1_bgm_lyrics_match";
    } else if (currentType === 'D') {
      tts = "어르신 안녕 안녕 반가워요! 의자에 안전하게 앉아 신나게 손뼉을 치며 박수를 쳐볼게요! 짝짝짝!";
      sensorCueBox.textContent = "의자에 앉아 음악에 맞춰 박수 치기";
      waitingSensorAction = null;
    }

    if (tts !== "") {
      AudioManager.speak(tts);
    }
  }

  function initPhase2State() {
    sensorBadge.textContent = "자세 집중";
    waitingSensorAction = null;
    let tts = "";
    
    if (currentType === 'A') {
      sensorCueBox.textContent = "OTAGO 프로그램: 의자 잡고 스쿼트 운동 수행";
      AudioManager.speak("자세에 집중하는 시간입니다. 양발을 어깨너비로 벌리고, 투명 의자에 앉듯 엉덩이를 뒤로 빼며 천천히 앉아보세요.", () => {
        setTimeout(() => {
          if (currentPhase === 2 && currentType === 'A' && isPlaying) {
            AudioManager.speak("네, 허벅지에 힘을 주고 그대로 일어납니다. 아주 정확한 자세예요!");
          }
        }, 3000);
      });
      return;
    }
    
    if (currentType === 'B') {
      sensorCueBox.textContent = "사이드 스텝 운동 수행: 오른쪽 ➔ 왼쪽";
      AudioManager.speak("어르신, 이번엔 게걸음으로 옆으로 걸어볼게요. 다리를 조금만 벌려서 오른쪽으로 한 발짝, 두 발짝.", () => {
        setTimeout(() => {
          if (currentPhase === 2 && currentType === 'B' && isPlaying) {
            AudioManager.speak("네, 급할 것 하나 없습니다. 삐끗하지 않게 천천히 움직여주세요.");
          }
        }, 3000);
      });
      return;
    }
    
    if (currentType === 'C') {
      sensorCueBox.textContent = "OTAGO 프로그램: 의자 잡고 뒤꿈치 들어올리기 (종아리 강화)";
      AudioManager.speak("안전을 위해 꼭 의자 등받이를 양손으로 잡아주세요. 자, 발뒤꿈치를 천천히 위로 들어 올립니다.", () => {
        setTimeout(() => {
          if (currentPhase === 2 && currentType === 'C' && isPlaying) {
            AudioManager.speak("종아리가 당기는 느낌이 드시죠? 천천히 내립니다. 하체에 힘이 생기는 과정입니다.");
          }
        }, 3000);
      });
      return;
    } else if (currentType === 'D') {
      tts = "어르신! 음악 좋지요? 엉덩이를 들썩들썩 해볼까요? 오른쪽! 왼쪽! 억지로 힘주지 마시고 편안하게 음악을 즐기며 움직여보세요.";
      sensorCueBox.textContent = "음악에 맞춰 엉덩이 리듬 타기 (체중 좌우 이동)";
    }

    if (tts !== "") {
      AudioManager.speak(tts);
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
        mode: 'elephant'
      };
      const intro = "자, 이번엔 재미있는 상상 놀이를 할 거예요! 제가 코끼리처럼 커다랗다! 하면 다리를 넓게 쩍 벌려주세요!";
      sensorCueBox.textContent = "코끼리처럼 커다랗다! ➔ 다리 쩍 벌리기";
      AudioManager.speak(intro);
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
      tts = "다시 의자에 편안하게 앉아서 다리를 앞으로 쭉 뻗고 숙여봅니다. 어려운 퀴즈도 거뜬히 풀어내서 최고입니다. 깊게 숨을 마시고 쉬세요.";
      sensorCueBox.textContent = "앉아서 다리 뻗고 상체 숙이기 스트레칭";
    } else if (currentType === 'D') {
      tts = "짝짝짝! 정말 잘하셨어요! 힘든데도 마지막 최고 단계까지 신나게 함께해주셔서 만점입니다! 내일도 신나게 놀아요!";
      sensorCueBox.textContent = "오늘의 수련 100점 만점! 축하드립니다.";
    }

    if (tts !== "") {
      AudioManager.speak(tts);
    }
  }

  // 3단계 듀얼과제 시간의 가상 시나리오 전환용
  function runPhaseGameTick() {
    if (currentPhase !== 3 || !isPlaying) return;

    if (currentType === 'A') {
      // 9분 (540초) 동안 6가지 하위 운동을 각 90초씩 진행
      const subIndex = Math.min(5, Math.floor(phaseElapsed / 90));
      
      if (phaseGameData.subIndex !== subIndex) {
        phaseGameData.subIndex = subIndex;
        phaseGameData.subTick = 0;
        
        let introText = "";
        if (subIndex === 0) {
          introText = "첫 번째 본운동은 의자 스쿼트와 정지 신호등 버티기입니다. 제가 노란불! 이라고 외치면 엉덩이가 의자에 닿기 직전 상태로 멈춰서 삼초간 버티세요. 초록불! 이면 완전히 앉았다가 바로 일어납니다. 스쿼트 시작하세요.";
          sensorCueBox.textContent = "의자 스쿼트 진행 중 (신호등 대기)...";
          phaseGameData.light = "초록불";
        } else if (subIndex === 1) {
          introText = "두 번째 본운동은 일자 걷기와 타겟 무릎 올리기입니다. 앞꿈치와 뒤꿈치를 붙여 앞으로 세 걸음, 다시 뒤로 세 걸음 걸어보세요. 체리! 소리가 나면 한쪽 무릎을 높이 올리세요.";
          sensorCueBox.textContent = "일자 걷기 진행 중...";
        } else if (subIndex === 2) {
          introText = "세 번째 본운동은 청개구리 걷기와 메모리 스텝입니다. 앞뒤좌우 스텝을 밟으시면서, 제가 불러드리는 방향 순서대로 스텝을 밟아보세요.";
          sensorCueBox.textContent = "기억 스텝 대기 중...";
          phaseGameData.seq = ["앞", "왼쪽"];
          phaseGameData.index = 0;
        } else if (subIndex === 3) {
          introText = "네 번째 본운동은 제자리 무릎 들어올리기와 조건 변경입니다. 무릎을 높이 들며 빠르게 제자리 걷기를 시작해 주세요. 첫 번째 규칙! 제가 과일 이름을 부르면 멈추고, 동물 이름이면 계속 뛰세요! 사자!";
          sensorCueBox.textContent = "[규칙] 과일 ➔ 정지(얼음) | 동물 ➔ 계속 뛰기";
          phaseGameData.rule = "fruit_stop";
          phaseGameData.words = ["사자", "호랑이", "포도", "수박", "토끼"];
          phaseGameData.wordIndex = 0;
        } else if (subIndex === 4) {
          introText = "다섯 번째 본운동은 뒤꿈치 들기와 앤백 리듬입니다. 의자를 튼튼하게 잡고 뒤꿈치를 들었다가 내리며, 띵 소리가 나면 앉았다가 일어서세요.";
          sensorCueBox.textContent = "뒤꿈치 들기 진행 중...";
        } else if (subIndex === 5) {
          introText = "마지막 본운동은 수리 연산 스쿼트입니다. 제가 퀴즈를 내면 그 정답 횟수만큼 스쿼트를 하세요. 일 더하기 이은 무엇일까요?";
          sensorCueBox.textContent = "1 + 2 = ? 정답만큼 스쿼트";
          phaseGameData.answer = 3;
          phaseGameData.count = 0;
        }
        
        AudioManager.speak(introText);
      } else {
        phaseGameData.subTick += 1;
        
        // 하위 운동별 주기적 음성 가이드 및 지시
        if (subIndex === 0) {
          if (phaseGameData.subTick % 20 === 0) {
            const chosen = Math.random() > 0.5 ? "노란불" : "초록불";
            phaseGameData.light = chosen;
            if (chosen === "노란불") {
              sensorCueBox.innerHTML = "<span style='color: #f59e0b; font-weight: 800;'>노란불! 3초간 정지 후 버티세요!</span>";
              AudioManager.speak("노란불! 멈추세요!");
            } else {
              sensorCueBox.innerHTML = "<span style='color: #10b981; font-weight: 800;'>초록불! 완전히 앉았다 일어나세요!</span>";
              AudioManager.speak("초록불!");
            }
          }
        } 
        else if (subIndex === 1) {
          if (phaseGameData.subTick % 15 === 0) {
            const words = ["사과", "체리", "바나나"];
            const w = words[Math.floor(Math.random() * words.length)];
            sensorCueBox.innerHTML = `단어: <strong>${w}</strong> (체리일 때 무릎 올리기!)`;
            AudioManager.speak(w);
          }
        }
        else if (subIndex === 2) {
          if (phaseGameData.subTick % 25 === 0) {
            const dirs = [
              ["앞", "오른쪽"], 
              ["뒤", "왼쪽"], 
              ["왼쪽", "앞"],
              ["오른쪽", "뒤"],
              ["앞", "뒤", "왼쪽"],
              ["오른쪽", "앞", "왼쪽"]
            ];
            phaseGameData.seq = dirs[Math.floor(Math.random() * dirs.length)];
            phaseGameData.index = 0;
            sensorCueBox.textContent = `기억 스텝: ${phaseGameData.seq.join(" ➔ ")}`;
            AudioManager.speak(`${phaseGameData.seq.join(", ")}! 순서대로 밟아주세요.`);
          }
        }
        else if (subIndex === 3) {
          if (phaseGameData.subTick === 45) {
            phaseGameData.rule = "animal_stop";
            sensorCueBox.textContent = "[규칙 변경] 과일 ➔ 계속 뛰기 | 동물 ➔ 정지(얼음)";
            AudioManager.speak("삐빅! 규칙이 반대로 바뀝니다! 이제 과일에는 뛰고 동물 이름에 멈추세요! 수박!");
          } else if (phaseGameData.subTick % 12 === 0 && phaseGameData.subTick !== 45) {
            phaseGameData.wordIndex = (phaseGameData.wordIndex + 1) % phaseGameData.words.length;
            const w = phaseGameData.words[phaseGameData.wordIndex];
            const isFruit = (w === '포도' || w === '수박');
            const isStop = (phaseGameData.rule === 'fruit_stop' && isFruit) || (phaseGameData.rule === 'animal_stop' && !isFruit);
            sensorCueBox.innerHTML = `단어: <strong>${w}</strong> ${isStop ? "➔ 멈추세요! (얼음!)" : "(계속 걷기)"}`;
            AudioManager.speak(w);
          }
        }
        else if (subIndex === 4) {
          if (phaseGameData.subTick % 15 === 0) {
            sensorCueBox.innerHTML = "<span style='color: #ef4444; font-weight: 800;'>📢 띵! (앉았다가 일어나세요!)</span>";
            AudioManager.playEffect('ding_bright.mp3');
          }
        }
        else if (subIndex === 5) {
          if (phaseGameData.subTick % 25 === 0) {
            const num1 = Math.floor(Math.random() * 2) + 1; // 1~2
            const num2 = Math.floor(Math.random() * 2) + 1; // 1~2
            phaseGameData.answer = num1 + num2;
            phaseGameData.count = 0;
            sensorCueBox.textContent = `${num1} + ${num2} = ? 정답 횟수만큼 스쿼트`;
            AudioManager.speak(`${num1} 더하기 ${num2}는 무엇일까요? 정답 수만큼 앉았다 일어나세요.`);
          }
        }
      }
    }
    else if (currentType === 'B') {
      // 9분 (540초) 동안 10가지 하위 운동을 각 54초씩 진행
      const subIndex = Math.min(9, Math.floor(phaseElapsed / 54));
      
      if (phaseGameData.subIndex !== subIndex) {
        phaseGameData.subIndex = subIndex;
        phaseGameData.subTick = 0;
        
        let introText = "";
        if (subIndex === 0) {
          introText = "첫 번째 본운동은 제자리 걷기와 신호등 얼음땡입니다. 어르신, 편하게 제자리걸음을 해볼까요? 하나 둘, 하나 둘. 걷다가 징 소리가 나면 그 자리에 딱! 멈춰서 얼음! 해주세요.";
          sensorCueBox.textContent = "제자리 걷기 진행 중 (신호등 대기)...";
        } else if (subIndex === 1) {
          introText = "두 번째 본운동은 좌우 체중 이동과 쿵짝 박자 맞추기입니다. 다리를 어깨만큼 벌리고 서주세요. 음악이 쿵 짝짝 할 텐데, 쿵 소리가 날 때마다 오른쪽, 짝 소리엔 왼쪽으로 체중을 꾹꾹 눌러볼게요.";
          sensorCueBox.textContent = "좌우 체중 이동 진행 중...";
          phaseGameData.dir = "오른쪽";
        } else if (subIndex === 2) {
          introText = "세 번째 본운동은 무릎들어올리기와 횟수 기억하기입니다. 한쪽 무릎씩 들어올리기를 할 거예요. 제가 두 번! 이라고 말씀드리면, 속으로 하나, 둘 세면서 딱 두 번씩 무릎을 들어올려주세요. 오른쪽 두 번, 왼쪽 두 번.";
          sensorCueBox.textContent = "무릎 들어올리기 진행 중...";
        } else if (subIndex === 3) {
          introText = "네 번째 본운동은 사이드 스텝과 동물 과일 분류기입니다. 게걸음으로 옆으로 걸을 거예요. 제가 사과 같은 과일을 말하면 오른쪽으로 걷고, 강아지 같은 동물을 말하면 왼쪽으로 걸어볼까요? 자, 포도!";
          sensorCueBox.textContent = "[규칙] 과일 ➔ 오른쪽 스텝 | 동물 ➔ 왼쪽 스텝";
          phaseGameData.words = ["강아지", "사과", "고양이", "포도", "원숭이"];
          phaseGameData.wordIndex = 0;
        } else if (subIndex === 4) {
          introText = "다섯 번째 본운동은 발뒤꿈치 들기와 특정 색깔 잡기입니다. 의자를 튼튼하게 잡으시고요. 제가 여러 색깔을 섞어서 부를 텐데, 빨강! 이라고 할 때만 뒤꿈치를 번쩍 들어주세요. 노랑... 파랑... 빨강!";
          sensorCueBox.textContent = "[규칙] 빨강 ➔ 뒤꿈치 들기";
        } else if (subIndex === 5) {
          introText = "여섯 번째 본운동은 사이드 걷기와 순서 기억입니다. 제가 왼쪽 오른쪽 오른쪽 왼쪽 이렇게 알려주면 순서대로 사이드 걷기를 실행해보세요. 먼저, 왼쪽, 오른쪽, 시작!";
          sensorCueBox.textContent = "기억 스텝 대기 중...";
          phaseGameData.seq = ["왼쪽", "오른쪽"];
          phaseGameData.index = 0;
        } else if (subIndex === 6) {
          introText = "일곱 번째 본운동은 의자 스쿼트와 더하기 빼기 누적입니다. 의자에서 딱 두 번만 일어났다 앉아보세요.";
          sensorCueBox.textContent = "의자 스쿼트 2회 실시";
          phaseGameData.step = 0; // 0: 2회, 1: 더하기 1, 2: 빼기 2
          phaseGameData.target = 2;
          phaseGameData.count = 0;
        } else if (subIndex === 7) {
          introText = "여덟 번째 본운동은 제자리 걷기와 조건 변경입니다. 지금부터 낮 하면 걷고, 밤 하면 멈추세요. 낮!";
          sensorCueBox.textContent = "[규칙] 낮 ➔ 걷기 | 밤 ➔ 정지";
          phaseGameData.rule = "day_run";
        } else if (subIndex === 8) {
          introText = "아홉 번째 본운동은 좌우 체중 이동과 크기 비교기입니다. 제가 물건 이름을 말할 텐데, 머릿속으로 상상해 보세요. 그 물건이 수박보다 크면 오른쪽으로 체중을 꾹 누르고, 작으면 왼쪽으로 눌러보세요. 자동차!";
          sensorCueBox.textContent = "[규칙] 수박보다 크면 ➔ 오른쪽 | 작으면 ➔ 왼쪽";
        } else if (subIndex === 9) {
          introText = "마지막 본운동은 의자 스쿼트와 숫자 홀짝 분별입니다. 숫자를 하나 부를게요. 그 숫자가 오보다 크면 한 번 일어났다 앉으시고요. 오보다 작으면 제자리 걸음 해주세요. 자, 시작합니다. 삼!";
          sensorCueBox.textContent = "[규칙] 5보다 크면 ➔ 스쿼트 1회 | 5보다 작으면 ➔ 제자리 걸음";
        }
        
        AudioManager.speak(introText);
      } else {
        phaseGameData.subTick += 1;
        
        // 각 하위 운동별 주기적 TTS 및 지시
        if (subIndex === 0) {
          if (phaseGameData.subTick % 15 === 0) {
            sensorCueBox.innerHTML = "<span style='color: #ef4444; font-weight: 800;'>얼음! (그 자리에 딱 멈추세요!)</span>";
            AudioManager.playEffect('ding_bright.mp3');
          }
        }
        else if (subIndex === 1) {
          if (phaseGameData.subTick % 12 === 0) {
            phaseGameData.dir = Math.random() > 0.5 ? "오른쪽" : "왼쪽";
            const soundWord = phaseGameData.dir === "오른쪽" ? "쿵 (오른쪽)" : "짝 (왼쪽)";
            const effectFile = phaseGameData.dir === "오른쪽" ? "thud.mp3" : "slap.mp3";
            sensorCueBox.innerHTML = `박자 지시: <strong>${soundWord}</strong>`;
            AudioManager.playEffect(effectFile);
          }
        }
        else if (subIndex === 2) {
          if (phaseGameData.subTick % 15 === 0) {
            const counts = [2, 3];
            const chosen = counts[Math.floor(Math.random() * counts.length)];
            const textWord = chosen === 2 ? "두 번!" : "세 번!";
            sensorCueBox.textContent = `지시 횟수: ${textWord} (오른쪽 ${chosen}번, 왼쪽 ${chosen}번)`;
            AudioManager.speak(textWord);
          }
        }
        else if (subIndex === 3) {
          if (phaseGameData.subTick % 12 === 0) {
            phaseGameData.wordIndex = (phaseGameData.wordIndex + 1) % phaseGameData.words.length;
            const w = phaseGameData.words[phaseGameData.wordIndex];
            const isFruit = (w === '사과' || w === '포도');
            sensorCueBox.innerHTML = `단어: <strong>${w}</strong> (${isFruit ? "과일 ➔ 오른쪽" : "동물 ➔ 왼쪽"})`;
            AudioManager.speak(w);
          }
        }
        else if (subIndex === 4) {
          if (phaseGameData.subTick % 12 === 0) {
            const colors = ["노랑", "파랑", "빨강"];
            const c = colors[Math.floor(Math.random() * colors.length)];
            sensorCueBox.innerHTML = `색상: <span style='font-weight:800;'>${c}</span> (빨강일 때만 뒤꿈치 들기)`;
            AudioManager.speak(c);
          }
        }
        else if (subIndex === 5) {
          if (phaseGameData.subTick % 20 === 0) {
            const paths = [["왼쪽", "오른쪽", "오른쪽"], ["오른쪽", "왼쪽", "왼쪽", "오른쪽"]];
            phaseGameData.seq = paths[Math.floor(Math.random() * paths.length)];
            phaseGameData.index = 0;
            sensorCueBox.textContent = `기억 스텝: ${phaseGameData.seq.join(" ➔ ")}`;
            AudioManager.speak(`${phaseGameData.seq.join(", ")}! 순서대로 밟아주세요.`);
          }
        }
        else if (subIndex === 6) {
          if (phaseGameData.subTick === 15) {
            phaseGameData.step = 1;
            phaseGameData.target = 3;
            phaseGameData.count = 0;
            sensorCueBox.textContent = "누적 계산: 2 + 1 = 3회 스쿼트 실시";
            AudioManager.speak("방금 일어난 횟수에 하나를 더하면 몇 번일까요? 그만큼 다시 일어나 보세요.");
          } else if (phaseGameData.subTick === 35) {
            phaseGameData.step = 2;
            phaseGameData.target = 1;
            phaseGameData.count = 0;
            sensorCueBox.textContent = "누적 계산: 3 - 2 = 1회 스쿼트 실시";
            AudioManager.speak("방금 일어난 횟수에서 둘을 빼면 몇 번일까요? 그만큼 다시 일어나 보세요.");
          }
        }
        else if (subIndex === 7) {
          if (phaseGameData.subTick === 27) {
            phaseGameData.rule = "day_stop";
            sensorCueBox.textContent = "[규칙 변경] 낮 ➔ 정지 | 밤 ➔ 걷기";
            AudioManager.speak("어르신! 지금부터는 반대예요. 규칙이 바뀌었어요. 낮에 멈추고 밤에 걸어보세요! 밤!");
          } else if (phaseGameData.subTick % 12 === 0 && phaseGameData.subTick !== 27) {
            const times = ["낮", "밤"];
            const tVal = times[Math.floor(Math.random() * times.length)];
            const isRun = (phaseGameData.rule === 'day_run' && tVal === '낮') || (phaseGameData.rule === 'day_stop' && tVal === '밤');
            sensorCueBox.innerHTML = `상태: <strong>${tVal}</strong> (${isRun ? "걷기" : "멈춤"})`;
            AudioManager.speak(tVal);
          }
        }
        else if (subIndex === 8) {
          if (phaseGameData.subTick % 15 === 0) {
            const items = ["자동차", "포도", "비행기", "사과"];
            const item = items[Math.floor(Math.random() * items.length)];
            const isLarge = (item === '자동차' || item === '비행기');
            sensorCueBox.innerHTML = `단어: <strong>${item}</strong> (${isLarge ? "수박보다 큼 ➔ 오른쪽" : "수박보다 작음 ➔ 왼쪽"})`;
            AudioManager.speak(item);
          }
        }
        else if (subIndex === 9) {
          if (phaseGameData.subTick % 15 === 0) {
            const nums = [3, 7];
            const chosenNum = nums[Math.floor(Math.random() * nums.length)];
            const isLarge = chosenNum > 5;
            sensorCueBox.innerHTML = `숫자: <strong>${chosenNum}</strong> (${isLarge ? "5보다 큼 ➔ 스쿼트 1회" : "5보다 작음 ➔ 제자리 걷기"})`;
            AudioManager.speak(chosenNum.toString());
          }
        }
      }
    }
    else if (currentType === 'C') {
      // 9분 (540초) 동안 7가지 하위 운동을 각 77초씩 진행
      const subIndex = Math.min(6, Math.floor(phaseElapsed / 77));
      
      if (phaseGameData.subIndex !== subIndex) {
        phaseGameData.subIndex = subIndex;
        phaseGameData.subTick = 0;
        
        let introText = "";
        if (subIndex === 0) {
          introText = "첫 번째 본운동은 의자 잡고 까치발과 방향 듣고 체중 이동입니다. 의자를 잡고 뒤꿈치를 올리면서 제가 왼쪽! 하면 왼쪽 발을 꾹 눌러주세요. 왼쪽!";
          sensorCueBox.textContent = "의자 잡고 까치발 상태에서 방향 신호 대기";
          phaseGameData.dir = "왼쪽";
        } else if (subIndex === 1) {
          introText = "두 번째 본운동은 의자 잡고 제자리 걷기와 얼음땡 놀이입니다. 의자 잡고 힘차게 걷다가 얼음! 소리가 나면 양발을 바닥에 딱 붙이고 멈추세요.";
          sensorCueBox.textContent = "제자리 걷기 진행 중 (신호등 대기)...";
        } else if (subIndex === 2) {
          introText = "세 번째 본운동은 앉아서 무릎 펴기와 음절 수 쿵쿵입니다. 의자에 안전하게 앉아 다리를 펴주세요. 비행기는 세 글자죠? 양쪽 다리를 세 번 쭉쭉 펴보세요.";
          sensorCueBox.textContent = "단어: 비행기 (3글자) ➔ 무릎 3번 펴기 (0/3)";
          phaseGameData.word = "비행기";
          phaseGameData.target = 3;
          phaseGameData.count = 0;
        } else if (subIndex === 3) {
          introText = "네 번째 본운동은 앉아서 몸통 비틀기와 카테고리 분류입니다. 들리는 단어가 먹는 것이면 오른쪽, 입는 것이면 왼쪽으로 몸통을 회전해보세요. 사과!";
          sensorCueBox.textContent = "[규칙] 먹는 것 ➔ 오른쪽 | 입는 것 ➔ 왼쪽";
        } else if (subIndex === 4) {
          introText = "다섯 번째 본운동은 앉아서 한 발 버티기와 이야기 듣고 기억하기입니다. 오른쪽 무릎을 쭉 펴고 버티세요. 버티시는 동안 세 가지 물건을 불러드릴 테니 기억하세요. 지갑, 열쇠, 안경.";
          sensorCueBox.textContent = "오른쪽 무릎 펴고 3가지 단어 기억 중...";
          phaseGameData.step = 0; // 0: 버티기, 1: 내리기 완료, 2: 질문
        } else if (subIndex === 5) {
          introText = "여섯 번째 본운동은 좌골 체중 이동과 O/X 상식 퀴즈입니다. 허리를 세우고 앉아, 맞으면 오른쪽 엉덩이를 꾹 누르고, 틀리면 왼쪽 엉덩이를 꾹 눌러보세요. 지구는 둥급니다!";
          sensorCueBox.textContent = "[규칙] 맞으면(O) ➔ 오른쪽 엉덩이 | 틀리면(X) ➔ 왼쪽 엉덩이";
        } else if (subIndex === 6) {
          introText = "마지막 본운동은 앉아서 양발 멀리 보냈다가 모으기와 크기 공간 지각입니다. 동물 이름을 부를 텐데, 그 동물이 나보다 크면 다리를 넓게 쩍 벌려주시고, 나보다 작으면 다리를 모아주세요. 호랑이!";
          sensorCueBox.textContent = "[규칙] 나보다 크면 ➔ 다리 벌리기 | 나보다 작으면 ➔ 다리 모으기";
        }
        
        AudioManager.speak(introText);
      } else {
        phaseGameData.subTick += 1;
        
        // 각 하위 운동별 주기적 TTS 및 지시
        if (subIndex === 0) {
          if (phaseGameData.subTick % 15 === 0) {
            phaseGameData.dir = Math.random() > 0.5 ? "오른쪽" : "왼쪽";
            sensorCueBox.innerHTML = `방향 지시: <strong style="color: #005EA8;">${phaseGameData.dir}</strong>`;
            AudioManager.speak(phaseGameData.dir);
          }
        }
        else if (subIndex === 1) {
          if (phaseGameData.subTick % 15 === 0) {
            sensorCueBox.innerHTML = "<span style='color: #ef4444; font-weight: 800;'>얼음! (멈춤)</span>";
            AudioManager.playEffect('ding_bright.mp3');
            AudioManager.speak("얼음!");
          }
        }
        else if (subIndex === 2) {
          if (phaseGameData.subTick % 20 === 0) {
            const words = ["수박", "텔레비전", "바나나", "책"];
            const w = words[Math.floor(Math.random() * words.length)];
            phaseGameData.word = w;
            phaseGameData.target = w.length;
            phaseGameData.count = 0;
            sensorCueBox.textContent = `단어: ${w} (${w.length}글자) ➔ 무릎 ${w.length}번 펴기 (0/${w.length})`;
            AudioManager.speak(`${w}! ${w.length}글자니까 ${w.length === 2 ? "두 번!" : w.length === 3 ? "세 번!" : w.length === 4 ? "네 번!" : "한 번!"} 쭉쭉 펴보세요.`);
          }
        }
        else if (subIndex === 3) {
          if (phaseGameData.subTick % 15 === 0) {
            const items = ["바지", "사과", "셔츠", "포도"];
            const item = items[Math.floor(Math.random() * items.length)];
            const isFood = (item === '사과' || item === '포도');
            sensorCueBox.innerHTML = `단어: <strong>${item}</strong> (${isFood ? "먹는 것 ➔ 오른쪽" : "입는 것 ➔ 왼쪽"})`;
            AudioManager.speak(item);
          }
        }
        else if (subIndex === 4) {
          if (phaseGameData.subTick === 15) {
            phaseGameData.step = 1;
            sensorCueBox.textContent = "다리를 내리고 편하게 대기하세요.";
            AudioManager.speak("천천히 다리를 내립니다.");
          } else if (phaseGameData.subTick === 30) {
            phaseGameData.step = 2;
            sensorCueBox.textContent = "질문: 방금 부른 세 가지가 맞습니까? (맞으면 ➔ 오른쪽 기울이기, 틀리면 ➔ 왼쪽 기울이기)";
            AudioManager.speak("방금 제가 부른 세 가지가 맞다면 오른쪽으로 기울기, 틀리다면 왼쪽으로 기울여주세요. 지갑, 열쇠, 안경이 맞습니까?");
          }
        }
        else if (subIndex === 5) {
          if (phaseGameData.subTick % 20 === 0) {
            const quizzes = [
              { q: "해는 서쪽에서 뜹니다.", a: false },
              { q: "사과는 빨간 과일입니다.", a: true },
              { q: "일 더하기 일은 삼입니다.", a: false }
            ];
            const quiz = quizzes[Math.floor(Math.random() * quizzes.length)];
            phaseGameData.quizAnswer = quiz.a;
            sensorCueBox.textContent = `퀴즈: ${quiz.q} (O ➔ 오른쪽, X ➔ 왼쪽 엉덩이)`;
            AudioManager.speak(quiz.q);
          }
        }
        else if (subIndex === 6) {
          if (phaseGameData.subTick % 15 === 0) {
            const animals = ["호랑이", "쥐", "코끼리", "나비"];
            const animal = animals[Math.floor(Math.random() * animals.length)];
            const isLarge = (animal === '호랑이' || animal === '코끼리');
            sensorCueBox.innerHTML = `동물: <strong>${animal}</strong> (${isLarge ? "나보다 큼 ➔ 다리 벌리기" : "나보다 작음 ➔ 다리 모으기"})`;
            AudioManager.speak(animal);
          }
        }
      }
    }
  }

  // 7. 가상 센서 클릭 및 실제 센서 이벤트 처리
  window.triggerVirtualSensor = function() {
    console.log("가상 센서 신호 유도됨.");
    
    // A/B유형 1단계 (워밍업) BGM 가사 구간 매칭 체크
    if ((currentType === 'A' || currentType === 'B') && currentPhase === 1) {
      if (!bgmAudio) return;
      const t = bgmAudio.currentTime;
      let matched = false;
      let actionName = "";
      
      // 구간 1: 53초~56초 (멈춤)
      if (t >= 53 && t <= 56) {
        matched = true;
        actionName = "정지 (얼음)";
      }
      // 구간 2: 1분 7초~1분 13초 (67초~73초, 오른쪽 체중이동)
      else if (t >= 67 && t <= 73) {
        matched = true;
        actionName = "오른쪽 체중 이동";
      }
      // 구간 3: 1분 18초~1분 21초 (78초~81초, 왼쪽 체중이동)
      else if (t >= 78 && t <= 81) {
        matched = true;
        actionName = "왼쪽 체중 이동";
      }
      // 구간 4: 1분 56초~2분 2초 (116초~122초, 오른발 앞으로 한발)
      else if (t >= 116 && t <= 122) {
        matched = true;
        actionName = "오른발 앞으로 한발";
      }
      // 구간 5: 2분 4초~2분 5초 (124초~125초, 뒤로 한발)
      else if (t >= 124 && t <= 125) {
        matched = true;
        actionName = "뒤로 한발";
      }
      
      if (matched) {
        AudioManager.playEffect('ding_bright.mp3');
        sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: ${actionName}`;
        console.log(`[A/B유형 1단계] 가사 동작 매칭 완료: ${actionName} (${Math.floor(t)}초)`);
      } else {
        sensorCueBox.innerHTML = `<span style='color: #ef4444; font-weight: 800;'>인식 실패 ❌</span><br>가사의 동작 지시 구간이 아닙니다. (현재 BGM 시간: ${Math.floor(t)}초)`;
        console.log(`[A/B유형 1단계] 가사 매칭 실패 (현재 시간: ${t}초)`);
      }
      return;
    }
    
    // C유형 1단계 (워밍업) BGM 가사 구간 매칭 체크
    if (currentType === 'C' && currentPhase === 1) {
      if (!bgmAudio) return;
      const t = bgmAudio.currentTime;
      let matched = false;
      let actionName = "";
      
      // 구간 1: 55초~61초 (오른쪽 체중 이동)
      if (t >= 55 && t <= 61) {
        matched = true;
        actionName = "오른쪽 체중 이동";
      }
      // 구간 2: 1분 9초~1분 18초 (69초~78초, 왼쪽 체중 이동)
      else if (t >= 69 && t <= 78) {
        matched = true;
        actionName = "왼쪽 체중 이동";
      }
      
      if (matched) {
        AudioManager.playEffect('ding_bright.mp3');
        sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>인식 성공! 🔔</span><br>감지된 동작: ${actionName}`;
        console.log(`[C유형 1단계] 가사 동작 매칭 완료: ${actionName} (${Math.floor(t)}초)`);
      } else {
        sensorCueBox.innerHTML = `<span style='color: #ef4444; font-weight: 800;'>인식 실패 ❌</span><br>가사의 동작 지시 구간이 아닙니다. (현재 BGM 시간: ${Math.floor(t)}초)`;
        console.log(`[C유형 1단계] 가사 매칭 실패 (현재 시간: ${t}초)`);
      }
      return;
    }
    
    // 그 외 유형 및 단계는 기존의 가상 이벤트 핸들러 적용
    let mockEvt = {};
    if (currentPhase === 1) {
      mockEvt = { type: 'motion', action: 'any' };
    } 
    else if (currentPhase === 3) {
      if (currentType === 'A') {
        mockEvt = { type: 'motion', action: 'stop' };
      } else if (currentType === 'B') {
        mockEvt = { type: 'motion', action: phaseGameData.seq[phaseGameData.index] === '왼쪽' ? 'step_left' : 'step_right' };
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
  }

  function handleSensorData(data) {
    if (!isPlaying) return;

    if (currentPhase === 1) {
      if (currentType === 'A') {
        if (waitingSensorAction === 'p1_stop' && data.action === 'stop') {
          AudioManager.playEffect('ding_bright.mp3');
          waitingSensorAction = 'p1_weight_right';
          sensorCueBox.textContent = "정지 인식 완료! ➔ 오른쪽으로 체중을 실어보세요.";
          AudioManager.speak("정지 완료! 아주 좋습니다. 이번에는 양발을 벌리고 오른쪽으로 체중을 꾹 실어보세요.");
        } 
        else if (waitingSensorAction === 'p1_weight_right' && data.action === 'weight_right') {
          AudioManager.playEffect('ding_bright.mp3');
          waitingSensorAction = 'p1_weight_left';
          sensorCueBox.textContent = "오른쪽 체중 이동 성공! ➔ 왼쪽으로 체중을 실어보세요.";
          AudioManager.speak("네, 잘하셨습니다. 이번에는 반대편 왼쪽으로 체중을 꾹 실어볼까요?");
        }
        else if (waitingSensorAction === 'p1_weight_left' && data.action === 'weight_left') {
          AudioManager.playEffect('ding_bright.mp3');
          waitingSensorAction = null;
          sensorCueBox.textContent = "워밍업 준비 완료! 곧 2단계로 넘어갑니다.";
          AudioManager.speak("훌륭합니다! 캘리브레이션이 완료되었습니다. 그럼 2단계 본운동으로 넘어가겠습니다!");
          
          setTimeout(() => {
            if (currentPhase === 1 && isPlaying) jumpToPhase(2);
          }, 4000);
        }
      }
      else if (currentType === 'B') {
        if (waitingSensorAction === 'p1_shake_right' && data.action === 'shake_right') {
          AudioManager.playEffect('ding_bright.mp3');
          waitingSensorAction = 'p1_shake_left';
          sensorCueBox.textContent = "오른쪽 흔들기 인식 완료! ➔ 왼쪽으로 흔들어보세요.";
          AudioManager.speak("오른쪽 인식 성공! 이번에는 왼쪽으로 시계추처럼 몸을 흔 흔들어볼까요?");
        }
        else if (waitingSensorAction === 'p1_shake_left' && data.action === 'shake_left') {
          AudioManager.playEffect('ding_bright.mp3');
          waitingSensorAction = null;
          sensorCueBox.textContent = "시계추 워밍업 완료! 2단계로 이동합니다.";
          AudioManager.speak("참 잘하셨어요! 캘리브레이션이 성공했습니다. 다리를 풀어주는 2단계로 이동해 볼까요?");
          
          setTimeout(() => {
            if (currentPhase === 1 && isPlaying) jumpToPhase(2);
          }, 4000);
        }
      }
    } 
    else if (currentPhase === 3) {
      if (currentType === 'A') {
        const subIndex = phaseGameData.subIndex || 0;
        
        if (subIndex === 0) {
          // 의자 스쿼트 + 정지 신호등 버티기
          AudioManager.playEffect('ding_bright.mp3');
          if (phaseGameData.light === "노란불") {
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>버티기 인식 성공! 🔔</span>";
            AudioManager.speak("네, 잘 버티셨습니다!");
          } else {
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>초록불 스쿼트 성공! 🔔</span>";
            AudioManager.speak("네, 잘 일어서셨습니다!");
          }
        } 
        else if (subIndex === 1) {
          // 일자 걷기 + 타겟 무릎 올리기
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>무릎 올리기 인식 성공! 🔔</span>";
        }
        else if (subIndex === 2) {
          // 청개구리 스텝 순서
          AudioManager.playEffect('ding_bright.mp3');
          phaseGameData.index++;
          if (phaseGameData.index >= phaseGameData.seq.length) {
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>기억 스텝 전체 일치! 🔔</span>";
            AudioManager.speak("아주 잘 기억하셨습니다!");
            phaseGameData.index = 0;
          } else {
            sensorCueBox.textContent = `다음 기억 스텝 입력: ${phaseGameData.seq[phaseGameData.index]}`;
          }
        }
        else if (subIndex === 3) {
          // 조건 변경 제자리 무릎 들어올리기
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>정지 신호 감지 완료! 🔔</span>";
        }
        else if (subIndex === 4) {
          // 뒤꿈치 들기 + N-back 리듬
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>리듬 스쿼트 성공! 🔔</span>";
        }
        else if (subIndex === 5) {
          // 수리 연산 스쿼트
          phaseGameData.count++;
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.textContent = `스쿼트 횟수 감지: ${phaseGameData.count} / ${phaseGameData.answer}회`;
          if (phaseGameData.count >= phaseGameData.answer) {
            sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>수리 연산 (${phaseGameData.answer}회) 스쿼트 성공! 🔔</span>`;
            AudioManager.speak("딩동댕! 정답입니다!");
          }
        }
      }
      else if (currentType === 'B') {
        const subIndex = phaseGameData.subIndex || 0;
        
        if (subIndex === 0) {
          // 제자리 걷기 + 신호등 얼음땡
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>정지 감지 성공! 🔔</span>";
        }
        else if (subIndex === 1) {
          // 좌우 체중 이동 + 쿵짝 박자 맞추기
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>체중 이동 (${phaseGameData.dir}) 성공! 🔔</span>`;
        }
        else if (subIndex === 2) {
          // 무릎들어올리기 + 횟수 기억하기
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>무릎 올리기 횟수 완료! 🔔</span>";
        }
        else if (subIndex === 3) {
          // 사이드 스텝 + 동물/과일 분류기
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>과일/동물 분류 성공! 🔔</span>";
        }
        else if (subIndex === 4) {
          // 발뒤꿈치 들기 + 특정 색깔 잡기
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>색상 밸런스 성공! 🔔</span>";
        }
        else if (subIndex === 5) {
          // 사이드 걷기 + 순서 기억
          AudioManager.playEffect('ding_bright.mp3');
          phaseGameData.index++;
          if (phaseGameData.index >= phaseGameData.seq.length) {
            sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>방향 순서 일치 완료! 🔔</span>";
            AudioManager.speak("참 잘하셨어요!");
            phaseGameData.index = 0;
          } else {
            sensorCueBox.textContent = `다음 기억 스텝 입력: ${phaseGameData.seq[phaseGameData.index]}`;
          }
        }
        else if (subIndex === 6) {
          // 의자 스쿼트 + 더하기 빼기 누적
          phaseGameData.count++;
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.textContent = `누적 스쿼트: ${phaseGameData.count} / ${phaseGameData.target}회`;
          if (phaseGameData.count >= phaseGameData.target) {
            sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>누적 스쿼트 (${phaseGameData.target}회) 완료! 🔔</span>`;
            AudioManager.speak("네, 잘하셨습니다.");
          }
        }
        else if (subIndex === 7) {
          // 제자리 걷기 + 조건변경
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>조건 제자리 걷기 성공! 🔔</span>";
        }
        else if (subIndex === 8) {
          // 좌우 체중 이동 + 크기 비교기
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>크기 비교 이동 성공! 🔔</span>";
        }
        else if (subIndex === 9) {
          // 의자 스쿼트 + 숫자 크기/홀짝 분별
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>숫자 판별 운동 성공! 🔔</span>";
        }
      }
      else if (currentType === 'C') {
        const subIndex = phaseGameData.subIndex || 0;
        
        if (subIndex === 0) {
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>방향 체중이동 성공! 🔔</span>`;
        }
        else if (subIndex === 1) {
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>정지 신호 반응 성공! 🔔</span>";
        }
        else if (subIndex === 2) {
          phaseGameData.count++;
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.textContent = `무릎 펴기 감지: ${phaseGameData.count} / ${phaseGameData.target}회`;
          if (phaseGameData.count >= phaseGameData.target) {
            sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>음절 수 무릎 펴기 (${phaseGameData.target}회) 성공! 🔔</span>`;
            AudioManager.speak("참 잘하셨어요!");
          }
        }
        else if (subIndex === 3) {
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>카테고리 분류 몸통 회전 성공! 🔔</span>";
        }
        else if (subIndex === 4) {
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>기억 퀴즈 답변 완료! 🔔</span>";
        }
        else if (subIndex === 5) {
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = `<span style='color: #059669; font-weight: 800;'>O/X 상식 퀴즈 답변 성공! 🔔</span>`;
        }
        else if (subIndex === 6) {
          AudioManager.playEffect('ding_bright.mp3');
          sensorCueBox.innerHTML = "<span style='color: #059669; font-weight: 800;'>크기/공간 지각 다리 동작 성공! 🔔</span>";
        }
      }
      else if (currentType === 'D') {
        if (data.action === 'any_reaction') {
          AudioManager.playEffect('ding_bright.mp3');
          
          if (phaseGameData.mode === 'elephant') {
            sensorCueBox.textContent = "대성공! 다리를 시원하게 쩍 벌렸습니다! 👏👏👏";
            AudioManager.speak("아이고 시원하게 잘 벌리셨다! 정말 최고예요!");
            phaseGameData.mode = 'ant';
            
            setTimeout(() => {
              if (currentPhase === 3 && isPlaying) {
                sensorCueBox.textContent = "개미처럼 작다! ➔ 다리 쏙 모으기";
                AudioManager.speak("자 이번엔, 개미처럼 아주 작다! 다리를 오므려보세요.");
              }
            }, 4000);
          } else {
            sensorCueBox.textContent = "대성공! 다리를 개미처럼 쏙 모았습니다! 🎉";
            AudioManager.speak("어르신 정말 최고예요! 백점 만점입니다!");
            phaseGameData.mode = 'elephant';
            
            setTimeout(() => {
              if (currentPhase === 3 && isPlaying) {
                sensorCueBox.textContent = "코끼리처럼 커다랗다! ➔ 다리 쩍 벌리기";
                AudioManager.speak("자 다시 한번, 코끼리처럼 커다랗다! 다리를 넓게 벌려주세요.");
              }
            }, 4000);
          }
        }
      }
    }
  }

  // 8. 초기 구동 및 제스처 스타터
  function init() {
    
    // BGM 오디오 재생 시간 업데이트 리스너 바인딩
    if (bgmAudio) {
      bgmAudio.ontimeupdate = () => {
        if (isPlaying) {
          if (bgmAudio.duration) {
            phaseElapsed = Math.floor(bgmAudio.currentTime);
          }
          if (currentPhase === 1 && (currentType === 'A' || currentType === 'B' || currentType === 'C')) {
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
