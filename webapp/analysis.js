// Pure signal-processing helpers. No DOM, so they can be tested in Node.
(function (root) {
  const A = {};

  // Root-mean-square level of a frame.
  A.rms = function (x) {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    return Math.sqrt(s / x.length);
  };

  // Zero-crossing rate: fraction of adjacent samples that change sign.
  A.zcr = function (x) {
    let c = 0;
    for (let i = 1; i < x.length; i++) if ((x[i - 1] >= 0) !== (x[i] >= 0)) c++;
    return c / (x.length - 1);
  };

  // Normalised autocorrelation pitch estimator.
  // Returns {f0, strength}: strength (0..1) is the height of the best peak,
  // which is high for periodic (voiced) sounds and low for noise.
  A.pitch = function (x, sr, fmin = 70, fmax = 500) {
    const n = x.length;
    const lagMin = Math.floor(sr / fmax);
    const lagMax = Math.min(Math.floor(sr / fmin), n - 1);
    let r0 = 0;
    for (let i = 0; i < n; i++) r0 += x[i] * x[i];
    if (r0 < 1e-9) return { f0: 0, strength: 0 };
    const r = new Float32Array(lagMax + 2);
    for (let lag = lagMin - 1; lag <= lagMax + 1; lag++) {
      let s = 0;
      for (let i = 0; i + lag < n; i++) s += x[i] * x[i + lag];
      // compensate the shrinking overlap
      r[lag] = (s / r0) * (n / (n - lag));
    }
    let best = lagMin, bv = -1;
    for (let lag = lagMin; lag <= lagMax; lag++) {
      if (r[lag] > r[lag - 1] && r[lag] >= r[lag + 1] && r[lag] > bv) { bv = r[lag]; best = lag; }
    }
    if (bv < 0) return { f0: 0, strength: 0 };
    // prefer the shortest lag with a nearly-as-high peak (avoids octave errors)
    for (let lag = lagMin; lag < best; lag++) {
      if (r[lag] > r[lag - 1] && r[lag] >= r[lag + 1] && r[lag] > 0.9 * bv) { best = lag; bv = r[lag]; break; }
    }
    // parabolic interpolation around the peak
    const a = r[best - 1], b = r[best], c = r[best + 1];
    const den = a - 2 * b + c;
    const shift = den !== 0 ? 0.5 * (a - c) / den : 0;
    return { f0: sr / (best + shift), strength: Math.min(1, Math.max(0, bv)) };
  };

  // Classify one frame as 'silence' | 'voiced' | 'unvoiced'.
  A.classify = function (x, sr, opts = {}) {
    const silenceRms = opts.silenceRms ?? 0.006;
    const level = A.rms(x);
    const z = A.zcr(x);
    if (level < silenceRms) return { cls: 'silence', f0: 0, level, zcr: z, strength: 0 };
    const p = A.pitch(x, sr);
    const voiced = p.strength > 0.45 && z < 0.25;
    return { cls: voiced ? 'voiced' : 'unvoiced', f0: voiced ? p.f0 : 0, level, zcr: z, strength: p.strength };
  };

  // Median filter to remove single-frame pitch glitches.
  A.median = function (arr) {
    const s = arr.slice().sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  };

  // Magma-like colormap, t in 0..1 -> [r,g,b]
  const stops = [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
  A.color = function (t) {
    t = Math.min(1, Math.max(0, t));
    const f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), u = f - i;
    return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * u));
  };

  if (typeof module !== 'undefined') module.exports = A; else root.Analysis = A;
})(typeof window !== 'undefined' ? window : globalThis);
