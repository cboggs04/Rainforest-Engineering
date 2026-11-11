"""Event-aware mixing with *measured* SNR.

SNR is defined on the event's active window: 20*log10(rms(event[win]) / rms(bg[win])),
where win = samples whose short-time RMS is within `range_db` of the event's loudest frame.
`mix_event` returns the achieved SNR so callers can assert it.
"""
import numpy as np

SR = 20000
N = 100000


def _frame_rms(x, frame=500):  # 25 ms
    n = len(x) // frame
    return np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(1) + 1e-20)


def active_bounds(x, range_db=30.0, frame=500):
    """(start, end) sample bounds of the region within range_db of the loudest frame."""
    r = _frame_rms(x, frame)
    idx = np.where(r >= r.max() * 10 ** (-range_db / 20))[0]
    return int(idx[0] * frame), int((idx[-1] + 1) * frame)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def mix_event(event, background, snr_db, start=None, rng=None, pre_pad=0.05):
    """Place the active part of `event` at `start` (s; random if None) inside `background`.
    Returns (mixture, achieved_snr_db, start_sample, end_sample). Mixture is jointly rescaled
    only if it would clip (SNR unchanged)."""
    rng = rng or np.random.default_rng()
    s, e = active_bounds(event)
    s = max(0, s - int(pre_pad * SR))
    seg = event[s:e]
    L = len(seg)
    if start is None:
        pos = int(rng.integers(0, N - L + 1))
    else:
        pos = int(np.clip(start * SR, 0, N - L))
    bg = background[:N].astype(np.float64)
    win = slice(pos, pos + L)
    gain = rms(seg) / (rms(bg[win]) * 10 ** (snr_db / 20))  # bg scale to hit SNR
    bg = bg * gain
    out = bg.copy()
    out[win] += seg
    achieved = 20 * np.log10(rms(seg) / rms(bg[win]))
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32), float(achieved), pos, pos + L


def simulate_playback(x, rng, sr=SR):
    """Crude speaker -> room -> microphone chain: band-limit, optional soft clipping,
    optional synthetic room reverb, and a noise floor. Output keeps the input RMS."""
    from scipy.signal import butter, sosfilt, fftconvolve
    x = np.asarray(x, dtype=np.float64); r0 = rms(x)
    if r0 < 1e-8:
        return x.astype(np.float32)
    lo, hi = rng.uniform(150, 500), rng.uniform(4000, 9000)
    y = sosfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)
    if rng.random() < 0.5:                                   # small-speaker soft clipping
        pk = np.abs(y).max() + 1e-12; d = rng.uniform(1.5, 4.0)
        y = np.tanh(d * y / pk) / np.tanh(d) * pk
    if rng.random() < 0.7:                                   # room reverb (exponential-decay noise IR)
        n = int(rng.uniform(0.1, 0.6) * sr)
        ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n); ir /= np.sqrt((ir ** 2).sum())
        w = rng.uniform(0.1, 0.5)
        wet = fftconvolve(y, ir)[:len(y)]; wet *= rms(y) / (rms(wet) + 1e-12)
        y = (1 - w) * y + w * wet
    floor = np.abs(y).max() * 10 ** (-rng.uniform(20, 45) / 20)   # peak-to-noise-floor 20-45 dB
    y = y + rng.standard_normal(len(y)) * floor
    y *= r0 / (rms(y) + 1e-12)
    return np.clip(y, -1, 1).astype(np.float32)
