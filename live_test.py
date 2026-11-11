"""Test the models yourself with the Mac microphone (stand-in for the ESP32 mic).

  python live_test.py              # loop: record 5 s from the mic, classify, repeat (Ctrl+C to stop)
  python live_test.py file.wav     # classify a WAV file (first 5 s)
  python live_test.py --export     # write held-out test clips to test_clips/ for playback from a phone/speaker

Recordings are saved to live_recordings/ so any surprising result can be re-analysed.
"""
import sys, os, time
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, soundfile as sf, librosa
from audio_preprocessing import reference_mel, fix_length, CLASS_NAMES

MODELS = {
    "V3-d":   "models/MobileNetV3_Small_Rainforest_V3b_D_s1/best.keras",
    "V3b-B2": "models/MobileNetV3_Small_Rainforest_V3b_B2_s0/best.keras",
    "V1":     "models/MobileNetV3_Small_Rainforest_Finetuned_V1/MobileNetV3_Finetuned_V1_best.keras",
}

def prep(y, sr):
    y = y.mean(1) if y.ndim > 1 else y
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=20000)   # same resampler as training
    return fix_length(y[:100000])

def classify(models, y20k, label=""):
    rms = float(np.sqrt(np.mean(y20k ** 2))); peak = float(np.abs(y20k).max())
    X = reference_mel(y20k)[None]
    print(f"\n{label}  level: RMS {rms:.4f}  peak {peak:.3f}" +
          ("   (very quiet - move closer / louder)" if rms < 0.003 else "") +
          ("   (CLIPPING - turn the speaker down)" if peak > 0.98 else ""))
    print(f"{'model':8s} " + " ".join(f"{c:>11s}" for c in CLASS_NAMES) + "   -> prediction")
    for name, m in models.items():
        p = m.predict(X, verbose=0)[0]
        print(f"{name:8s} " + " ".join(f"{v*100:10.1f}%" for v in p) + f"   -> {CLASS_NAMES[p.argmax()]}")

def load_models():
    import tensorflow as tf
    return {n: tf.keras.models.load_model(p, compile=False) for n, p in MODELS.items() if os.path.exists(p)}

def export():
    import pandas as pd
    meta = pd.read_csv("datasets/external/prepared/meta.csv"); A = np.load("datasets/external/prepared/audio.npy", mmap_mode="r")
    os.makedirs("test_clips", exist_ok=True); rng = np.random.default_rng(0)
    picks = {"chainsaw": ("cs", None, 4), "gunshot": ("gs", None, 6), "hardneg": ("hn", None, 8), "background": ("bg", None, 4)}
    for name, (role, _, n) in picks.items():
        idx = meta.index[(meta.split == "test") & (meta.role == role)]
        for k in rng.choice(idx, min(n, len(idx)), replace=False):
            f = f"test_clips/{name}_{meta.cls[k]}_{os.path.basename(meta.src[k]).split('.')[0]}.wav"
            sf.write(f, np.array(A[k]), 20000)
    print("Wrote", len(os.listdir("test_clips")), "held-out clips (never used in training) to test_clips/")

if __name__ == "__main__":
    if "--export" in sys.argv:
        export(); sys.exit()
    models = load_models()
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            y, sr = sf.read(f); classify(models, prep(y, sr), os.path.basename(f))
        sys.exit()
    import sounddevice as sd
    SR = 48000; os.makedirs("live_recordings", exist_ok=True)
    print("Mic test - play a sound near the Mac. Ctrl+C to stop.")
    try:
        while True:
            input("\nPress Enter, then start the sound (records 5 s)...")
            y = sd.rec(int(5 * SR), samplerate=SR, channels=1, dtype="float32"); sd.wait()
            f = time.strftime("live_recordings/rec_%H%M%S.wav"); sf.write(f, y, SR)
            classify(models, prep(y[:, 0], SR), f)
    except KeyboardInterrupt:
        print("\nstopped")
