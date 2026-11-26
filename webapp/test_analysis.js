// Run: node test_analysis.js
const A = require('./analysis.js');
const sr = 16000, N = 1024;
let fails = 0;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };

// 1. pure tones: pitch accuracy
for (const f of [110, 150, 220, 330, 440]) {
  const x = Float32Array.from({ length: N }, (_, i) => 0.3 * Math.sin(2 * Math.PI * f * i / sr));
  const r = A.classify(x, sr);
  ok(r.cls === 'voiced' && Math.abs(r.f0 - f) / f < 0.02, `tone ${f} Hz -> ${r.cls}, f0=${r.f0.toFixed(1)}`);
}
// 2. voiced vowel-like: harmonics of 120 Hz
{
  const x = Float32Array.from({ length: N }, (_, i) => { let s = 0; for (let h = 1; h <= 10; h++) s += Math.sin(2 * Math.PI * 120 * h * i / sr) / h; return 0.15 * s; });
  const r = A.classify(x, sr);
  ok(r.cls === 'voiced' && Math.abs(r.f0 - 120) < 3, `harmonic 120 Hz -> ${r.cls}, f0=${r.f0.toFixed(1)}`);
}
// 3. noise (like "fff") -> unvoiced
{
  let seed = 1; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
  const x = Float32Array.from({ length: N }, () => 0.2 * rnd());
  const r = A.classify(x, sr);
  ok(r.cls === 'unvoiced', `white noise -> ${r.cls} (zcr=${r.zcr.toFixed(2)}, strength=${r.strength.toFixed(2)})`);
}
// 4. silence
{
  const x = new Float32Array(N).fill(0.0005);
  ok(A.classify(x, sr).cls === 'silence', 'near-zero signal -> silence');
}
process.exit(fails ? 1 : 0);
