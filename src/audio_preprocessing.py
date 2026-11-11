"""Reference audio preprocessing for the Rainforest detector.

`reference_mel` reproduces the ORIGINAL training pipeline exactly
(Feature Generator.ipynb `mel_features_extractor`).
`firmware_mel` is a float32 NumPy re-implementation of what the ESP32
sketch does, so the two can be compared on identical waveforms.
"""
import numpy as np
import librosa
import scipy.signal

SR = 20000
N_SAMPLES = 100000
HOP = 512
N_MELS = 128
N_FFTS = (2048, 1024, 512)  # channel order
CLASS_NAMES = ["BACKGROUND", "CHAINSAW", "GUNSHOT"]

# FSC22 target (1-based, as in FSC22.csv) -> 3-class label.
# The notebook uses 0-based labels (target-1): background {1-5,20-23}, chainsaw 10, gunshot 14.
BACKGROUND_TARGETS_1B = [2, 3, 4, 5, 6, 21, 22, 23, 24]
CHAINSAW_TARGET_1B = 11
GUNSHOT_TARGET_1B = 15


def to_3class(label_0b):
    if label_0b in (1, 2, 3, 4, 5, 20, 21, 22, 23):
        return 0
    if label_0b == 10:
        return 1
    if label_0b == 14:
        return 2
    return -1


def fix_length(y, n=N_SAMPLES):
    """Centre-pad with zeros (as original `padding`) or centre-crop."""
    if len(y) < n:
        d = n - len(y)
        return np.pad(y, (d // 2, d - d // 2))
    return y[:n]


def reference_mel(audio):
    """(128,196,3) float32 log-Mel exactly as used for training."""
    chans = []
    for n_fft in N_FFTS:
        m = librosa.feature.melspectrogram(
            y=audio, sr=SR, n_mels=N_MELS, n_fft=n_fft, hop_length=HOP)
        chans.append(librosa.power_to_db(m))
    return np.stack(chans, axis=2)


def firmware_mel(audio, pad_mode="reflect", dtype=np.float32):
    """Emulates the ESP32 sketch: periodic Hann, `pad_mode` edge padding,
    float32 power spectrum, librosa mel filters, 10log10(max(x,1e-10)),
    per-channel floor at (max-80 dB)."""
    audio = np.asarray(audio, dtype=dtype)
    chans = []
    for n_fft in N_FFTS:
        win = scipy.signal.get_window("hann", n_fft, fftbins=True).astype(dtype)
        fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=N_MELS).astype(dtype)
        p = np.pad(audio, n_fft // 2, mode=pad_mode)
        n_frames = 1 + len(audio) // HOP
        frames = np.stack([p[i * HOP:i * HOP + n_fft] * win for i in range(n_frames)], axis=1)
        power = np.abs(np.fft.rfft(frames, axis=0)) ** 2
        mel = fb @ power.astype(dtype)
        db = 10 * np.log10(np.maximum(mel, 1e-10))
        db = np.maximum(db, db.max() - 80.0)
        chans.append(db)
    return np.stack(chans, axis=2).astype(np.float32)


def quantize_input(x, scale, zero_point):
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def pickle_folds(labels_0b):
    """Recreate the notebook's 5 'pickle' folds (Feature Generator cell 8):
    StratifiedKFold(5, shuffle=True, random_state=42) over clip labels.
    Returns list of 5 index arrays (order == order stored in mel_features_5_20_foldN)."""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(labels_0b)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return [te.tolist() for _, te in skf.split(np.zeros(len(y)), y)]
