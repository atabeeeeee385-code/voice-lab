# Live Voice Spectrogram

A single-page web app. Click start, talk into your microphone, and see:
- a scrolling **spectrogram** (0 to 4 kHz),
- the **pitch** (F0, in Hz) with a white marker on the spectrogram and a pitch track below,
- a live **voiced / unvoiced / silence** label ("aaa" and "mmm" are voiced; "fff" and "sss" are unvoiced).

All processing runs in your browser. Nothing is recorded or uploaded.

## Run
Browsers only allow the microphone on `localhost` or HTTPS, so use a tiny local server:
```bash
cd webapp
python -m http.server 8000
# open http://localhost:8000 in Chrome or Edge
```

## How it works
- Spectrogram: Web Audio `AnalyserNode` (2048-point FFT), drawn as a scrolling canvas with a magma colormap.
- Pitch: normalised autocorrelation on the latest ~23 ms, 70 to 500 Hz, parabolic peak interpolation, median smoothing, and a preference for the shortest strong lag to avoid octave errors.
- Voiced vs unvoiced: a frame is voiced when the autocorrelation peak is strong (> 0.45) and the zero-crossing rate is low (< 0.25). Silence is a simple level threshold.

## Tests
`node test_analysis.js` checks pitch accuracy on synthetic tones (110 to 440 Hz, within 2%), a harmonic voice-like signal, noise (must be unvoiced) and silence. A headless Chrome run with a fake microphone feeding real speech recordings produced `../docs/webapp_screenshot.png`.

## Limits
- Pitch can jump on very noisy rooms or very breathy voices.
- The silence threshold is fixed; a very quiet mic may need it lowered (`silenceRms` in `analysis.js`).
- Tested in Chrome only.
