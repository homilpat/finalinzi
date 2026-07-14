(function () {
  'use strict';

  const activeScreen = document.querySelector('[data-exercise-active]');
  if (!activeScreen) return;

  const progress = document.getElementById('exerciseProgress');
  const timer = document.getElementById('exerciseTimer');
  let pct = 18;

  window.setInterval(() => {
    pct = pct >= 92 ? 18 : pct + 7;
    if (progress) progress.style.width = `${pct}%`;

    if (!timer) return;
    const [minutes, seconds] = timer.textContent.split(':').map(Number);
    if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) return;

    const total = Math.max(0, minutes * 60 + seconds - 1);
    const nextMinutes = Math.floor(total / 60);
    const nextSeconds = String(total % 60).padStart(2, '0');
    timer.textContent = `${nextMinutes}:${nextSeconds}`;
  }, 1000);
})();
