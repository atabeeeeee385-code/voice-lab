const btn = document.getElementById('btn');
const statusEl = document.getElementById('status');
const specCv = document.getElementById('spec'), spec = specCv.getContext('2d');
const pitchCv = document.getElementById('pitch'), pc = pitchCv.getContext('2d');
const f0El = document.getElementById('f0'), clsEl = document.getElementById('cls'), lvlEl = document.getElementById('lvl');

let ctx, analyser, stream, raf, running = false;
const FFT = 2048, MAXHZ = 4000, STEP = 2;      // STEP = pixels per frame
const hist = [];                                 // pitch history
const HIST_MAX = Math.floor(pitchCv.width / STEP);
let f0Buf = [];

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { statusEl.textContent = 'Microphone error: ' + e.message; return; }
  ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  analyser = ctx.createAnalyser();
  analyser.fftSize = FFT; analyser.smoothingTimeConstant = 0;
  src.connect(analyser);
  spec.fillStyle = '#000'; spec.fillRect(0, 0, specCv.width, specCv.height);
  running = true; btn.textContent = 'Stop'; btn.classList.add('stop');
  statusEl.textContent = 'Listening... try "aaaa", then "ffff", then hum a rising note.';
  loop();
}
function stop() {
  running = false; cancelAnimationFrame(raf);
  stream.getTracks().forEach(t => t.stop()); ctx.close();
  btn.textContent = 'Start microphone'; btn.classList.remove('stop');
  statusEl.textContent = 'Stopped.';
}
btn.onclick = () => running ? stop() : start();

function loop() {
  if (!running) return;
  const sr = ctx.sampleRate;
  const td = new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(td);
  const fd = new Float32Array(analyser.frequencyBinCount); analyser.getFloatFrequencyData(fd);

  // Use the newest 1024 samples for pitch (about 23 ms at 44.1 kHz).
  const frame = td.subarray(td.length - 1024);
  const r = Analysis.classify(frame, sr);
  f0Buf.push(r.cls === 'voiced' ? r.f0 : 0); if (f0Buf.length > 5) f0Buf.shift();
  const f0 = r.cls === 'voiced' ? Analysis.median(f0Buf.filter(v => v > 0).length ? f0Buf.filter(v => v > 0) : [r.f0]) : 0;

  // scroll spectrogram left, draw new column on the right
  const W = specCv.width, H = specCv.height;
  spec.drawImage(specCv, -STEP, 0);
  const maxBin = Math.floor(MAXHZ / (sr / 2) * fd.length);
  const col = spec.createImageData(STEP, H);
  for (let y = 0; y < H; y++) {
    const bin = Math.min(maxBin - 1, Math.floor((1 - y / H) * maxBin));
    const t = (fd[bin] + 100) / 70;                 // -100 dB .. -30 dB
    const [R, G, B] = Analysis.color(t);
    for (let k = 0; k < STEP; k++) { const i = (y * STEP + k) * 4; col.data[i] = R; col.data[i+1] = G; col.data[i+2] = B; col.data[i+3] = 255; }
  }
  spec.putImageData(col, W - STEP, 0);
  if (f0 > 0 && f0 < MAXHZ) {                       // pitch marker
    const y = H * (1 - f0 / MAXHZ);
    spec.fillStyle = '#fff'; spec.fillRect(W - STEP, y - 1, STEP, 3);
  }

  hist.push({ f0, cls: r.cls }); if (hist.length > HIST_MAX) hist.shift();
  drawPitch();

  f0El.textContent = f0 ? f0.toFixed(0) + ' Hz' : '-';
  clsEl.textContent = r.cls; clsEl.className = 'value ' + r.cls;
  lvlEl.textContent = 'level ' + r.level.toFixed(3) + '  |  ZCR ' + r.zcr.toFixed(2);
  raf = requestAnimationFrame(loop);
}

function drawPitch() {
  const W = pitchCv.width, H = pitchCv.height;
  pc.fillStyle = '#000'; pc.fillRect(0, 0, W, H);
  pc.strokeStyle = '#21262d'; pc.fillStyle = '#6e7681'; pc.font = '11px sans-serif';
  for (const hz of [100, 200, 300, 400]) {
    const y = H * (1 - (hz - 50) / 450);
    pc.beginPath(); pc.moveTo(0, y); pc.lineTo(W, y); pc.stroke(); pc.fillText(hz + ' Hz', 4, y - 2);
  }
  hist.forEach((h, i) => {
    const x = W - (hist.length - i) * STEP;
    if (h.cls === 'voiced' && h.f0 >= 50 && h.f0 <= 500) {
      pc.fillStyle = '#3fb950'; pc.fillRect(x, H * (1 - (h.f0 - 50) / 450) - 1.5, STEP, 3);
    } else if (h.cls === 'unvoiced') {
      pc.fillStyle = '#f0883e'; pc.fillRect(x, H - 6, STEP, 6);
    }
  });
}
