(function () {
  'use strict';

  const data = window.PHYSICAL_MEASURE_DATA || {};
  const total = Number(data.duration_seconds) || 10;
  const countEl = document.getElementById('physicalMeasureCount');
  const statusEl = document.getElementById('physicalMeasureStatus');
  const progressEl = document.getElementById('physicalMeasureProgress');
  const completeBtn = document.getElementById('physicalMeasureCompleteBtn');
  let remain = total;

  function finishMeasurement() {
    if (window.completePhysicalMeasurement) {
      window.completePhysicalMeasurement(data.result || {});
      return;
    }
    window.location.href = '/physical/result';
  }

  function render() {
    if (countEl) countEl.textContent = remain;
    if (statusEl) statusEl.textContent = remain > 0 ? '측정 진행 중' : '측정 완료';
    if (progressEl) progressEl.style.width = `${Math.round(((total - remain) / total) * 100)}%`;
  }

  render();

  const timer = window.setInterval(() => {
    remain = Math.max(0, remain - 1);
    render();
    if (remain <= 0) {
      window.clearInterval(timer);
      if (progressEl) progressEl.style.width = '100%';
    }
  }, 1000);

  if (completeBtn) {
    completeBtn.addEventListener('click', finishMeasurement);
  }
})();
