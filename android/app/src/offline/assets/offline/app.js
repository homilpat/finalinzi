(() => {
  const screens = [...document.querySelectorAll('.screen')];
  const backButton = document.getElementById('backButton');
  let current = 'homeScreen';
  let timerId = null;
  let phaseStartedAt = 0;
  let phaseDuration = 7;

  function show(id) {
    screens.forEach(s => s.classList.toggle('active', s.id === id));
    current = id;
    backButton.classList.toggle('hidden', id === 'homeScreen' || id === 'measureScreen');
    window.scrollTo(0, 0);
  }

  function speak(text) {
    try { if (window.AndroidBridge?.speakGaitCue) window.AndroidBridge.speakGaitCue(text); } catch (_) {}
  }

  function updateProfile() {
    const profile = JSON.parse(localStorage.getItem('giukhaji_profile') || 'null');
    document.getElementById('memberSummary').textContent = profile?.name || '등록 전';
    if (profile) {
      document.getElementById('memberName').value = profile.name || '';
      document.getElementById('memberAge').value = profile.age || '';
    }
  }

  function updateLastResult() {
    const saved = JSON.parse(localStorage.getItem('giukhaji_last_gait') || 'null');
    const card = document.getElementById('lastResultCard');
    card.classList.toggle('hidden', !saved);
    if (!saved) return;
    document.getElementById('lastResultLabel').textContent = saved.label;
    document.getElementById('lastResultProbability').textContent = `${Math.round(saved.probability * 100)}%`;
    document.getElementById('lastResultTime').textContent = new Date(saved.measuredAt).toLocaleString('ko-KR');
  }

  function setPhase(title, eyebrow, help, seconds) {
    document.getElementById('measureTitle').textContent = title;
    document.getElementById('measureEyebrow').textContent = eyebrow;
    document.getElementById('measureHelp').textContent = help;
    phaseDuration = Math.max(1, seconds);
    phaseStartedAt = Date.now();
    clearInterval(timerId);
    const tick = () => {
      const elapsed = (Date.now() - phaseStartedAt) / 1000;
      const remain = Math.max(0, Math.ceil(phaseDuration - elapsed));
      document.getElementById('timerNumber').textContent = remain;
      document.getElementById('timerProgress').style.strokeDashoffset = String(327 * Math.min(1, elapsed / phaseDuration));
    };
    tick(); timerId = setInterval(tick, 200);
  }

  function beginMeasurement() {
    if (!window.AndroidBridge?.startGaitMeasurement) {
      showError('이 휴대폰에서는 센서 연결을 시작할 수 없습니다.'); return;
    }
    show('measureScreen');
    setPhase('휴대폰을 고정해 주세요', '측정 준비', '아직 걷지 않아도 됩니다.', 7);
    speak('휴대폰을 허리에 단단히 고정해 주세요.');
    window.AndroidBridge.startGaitMeasurement(JSON.stringify({wearSec:7, readySec:3, durationSec:20}));
  }

  function showResult(data) {
    clearInterval(timerId);
    const percent = Math.round(Number(data.probability) * 100);
    const impaired = Boolean(data.impaired);
    document.getElementById('resultSymbol').textContent = impaired ? '!' : '✓';
    document.getElementById('resultLabel').textContent = data.label;
    document.getElementById('resultProbability').textContent = `${percent}%`;
    document.getElementById('probabilityBar').style.width = `${percent}%`;
    document.getElementById('metricJerk').textContent = Number(data.v_jerk_rms_median).toFixed(3);
    document.getElementById('metricIqr').textContent = Number(data.v_jerk_rms_iqr).toFixed(3);
    document.getElementById('metricRhythm').textContent = Number(data.v_harmonic_ratio_iqr).toFixed(3);
    document.getElementById('resultCopy').textContent = impaired
      ? '이번 측정에서 운동기능 저하 가능성이 기준보다 높게 나타났습니다. 안전을 위해 반복 측정하거나 전문가와 상담해 주세요.'
      : '이번 측정은 모델 기준의 안정 범위에 있습니다. 같은 조건에서 정기적으로 측정하면 변화 확인에 도움이 됩니다.';
    const saved = {...data, measuredAt: Date.now()};
    localStorage.setItem('giukhaji_last_gait', JSON.stringify(saved));
    updateLastResult(); show('resultScreen');
    speak(`분석이 완료되었습니다. 결과는 ${data.label}입니다.`);
  }

  function showError(message) {
    clearInterval(timerId);
    document.getElementById('errorMessage').textContent = message || '센서 상태를 확인한 뒤 다시 시도해 주세요.';
    show('errorScreen');
  }

  window.onGaitApkEvent = event => {
    if (!event || !event.status) return;
    if (event.status === 'calibrating') {
      setPhase('잠시 움직이지 마세요', '센서 보정', '3초 동안 서 있는 자세를 확인합니다.', Number(event.durationSec || 3));
      speak('잠시 움직이지 말고 서 있어 주세요.');
    } else if (event.status === 'measuring') {
      setPhase('평소처럼 걸어 주세요', '보행 측정 중', '안전한 경로를 따라 20초 동안 걸어 주세요.', Number(event.durationSec || 20));
      speak('지금부터 평소처럼 걸어 주세요.');
    } else if (event.status === 'analyzing') {
      clearInterval(timerId);
      document.getElementById('measureEyebrow').textContent = '온디바이스 AI';
      document.getElementById('measureTitle').textContent = '휴대폰에서 분석 중';
      document.getElementById('measureHelp').textContent = '데이터는 외부로 전송되지 않습니다.';
      document.getElementById('timerNumber').textContent = 'AI';
      document.getElementById('timerProgress').style.strokeDashoffset = '0';
    } else if (event.status === 'complete') showResult(event);
    else if (event.status === 'error') showError(event.message);
  };

  backButton.onclick = () => show('homeScreen');
  document.getElementById('profileButton').onclick = () => show('profileScreen');
  document.getElementById('openGaitButton').onclick = () => show('gaitScreen');
  document.getElementById('startButton').onclick = beginMeasurement;
  document.getElementById('cancelButton').onclick = () => { try { window.AndroidBridge?.stopMeasurement(); } catch (_) {} show('gaitScreen'); };
  document.getElementById('retryButton').onclick = () => show('gaitScreen');
  document.getElementById('errorRetryButton').onclick = () => show('gaitScreen');
  document.getElementById('homeButton').onclick = document.getElementById('errorHomeButton').onclick = () => show('homeScreen');
  document.getElementById('profileForm').onsubmit = event => {
    event.preventDefault();
    const profile = {name: document.getElementById('memberName').value.trim(), age: document.getElementById('memberAge').value};
    localStorage.setItem('giukhaji_profile', JSON.stringify(profile));
    updateProfile(); show('homeScreen');
  };
  updateProfile(); updateLastResult();
})();
