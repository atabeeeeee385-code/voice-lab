"""MFCC feature extraction in plain NumPy/SciPy (no librosa needed)."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, get_window
from scipy.fftpack import dct

SR = 8000
N_FFT = 256
HOP = 80          # 10 ms
N_MELS = 26
N_MFCC = 13


def load_wav(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype != np.float32 or np.abs(x).max() > 1.5:
        x = x / 32768.0
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype(np.float32)


def _mel(f): return 2595 * np.log10(1 + f / 700)
def _imel(m): return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank():
    pts = _imel(np.linspace(_mel(0), _mel(SR / 2), N_MELS + 2))
    bins = np.floor((N_FFT + 1) * pts / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for i in range(1, N_MELS + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb

_FB = mel_filterbank()


def trim_silence(x, thresh_db=-35):
    frame = 80
    n = len(x) // frame
    if n == 0:
        return x
    e = np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(e / (e.max() + 1e-12) + 1e-12)
    idx = np.where(db > thresh_db)[0]
    if len(idx) == 0:
        return x
    return x[idx[0] * frame:(idx[-1] + 1) * frame]


def mfcc_frames(x):
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    win = get_window("hamming", N_FFT)
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] * win for i in range(n)])
    pw = np.abs(np.fft.rfft(frames, axis=1)) ** 2 / N_FFT
    mel = np.log(pw @ _FB.T + 1e-10)
    return dct(mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]


def clip_features(x, n_seg=5):
    """Fixed-length vector: mean/std of MFCCs over the clip, plus the mean
    MFCCs of n_seg equal time segments (keeps some word-shape information)."""
    x = trim_silence(x)
    m = mfcc_frames(x)
    m = m - m.mean(axis=0, keepdims=True) * 0.0  # keep absolute values
    segs = np.array_split(m, n_seg)
    seg_means = np.concatenate([s.mean(axis=0) for s in segs])
    zcr = np.mean(np.abs(np.diff(np.sign(x))) > 0)
    dur = len(x) / SR
    return np.concatenate([m.mean(axis=0), m.std(axis=0), seg_means, [zcr, dur]])
