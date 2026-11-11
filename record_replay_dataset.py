"""Collect REAL speaker->room->microphone recordings with exact labels.

1) python record_replay_dataset.py make
      -> replay_playlists/{train,test}_playlist.wav  (play these from your phone)
2) python record_replay_dataset.py record test --tag phone_50cm
      -> press Enter, start the playlist on the phone; the Mac mic records the whole thing
3) python record_replay_dataset.py slice test --tag phone_50cm
      -> datasets/replayed/test_phone_50cm/{audio.npy, meta.csv}   (labelled 5 s clips)

Train playlists use only TRAINING-split sources, test playlists only TEST-split sources, so there is no leakage.
Record each playlist in several sessions (different volume / distance / speaker) with different --tag values.
"""
import sys, os, argparse
sys.path.insert(0, "src")
import numpy as np, pandas as pd, soundfile as sf, librosa
from scipy.signal import chirp, fftconvolve

PSR = 44100            # playlist sample rate (phones play this natively)
GAP = 1.0              # silence between clips (s)
CH = 0.5               # sync chirp length (s)
LEAD = 2.0             # silence before first chirp (s)
D = "replay_playlists"

def sync_chirp(sr):
    t = np.arange(int(CH * sr)) / sr
    return (0.5 * chirp(t, 500, CH, 6000) * np.hanning(len(t))).astype(np.float32)

def pick(split, rng):
    from dataset_b import _fsc, _ext
    F, E = (_fsc([1, 2, 3]), _ext("train")) if split == "train" else (_fsc([5]), _ext("test"))
    n = dict(train=dict(gs=40, cs=30, hn=30, bg=20), test=dict(gs=20, cs=12, hn=16, bg=10))[split]
    out = []
    for role, k in n.items():
        pool = F[role] + E[role]
        for j in rng.choice(len(pool), min(k, len(pool)), replace=False):
            out.append((pool[j][0], role, pool[j][1]))
    rng.shuffle(out); return out

def make():
    os.makedirs(D, exist_ok=True); rng = np.random.default_rng(5)
    for split in ("train", "test"):
        clips = pick(split, rng); c = sync_chirp(PSR)
        parts = [np.zeros(int(LEAD * PSR), np.float32), c, np.zeros(int(GAP * PSR), np.float32)]; rows = []
        pos = sum(len(p) for p in parts)
        for x, role, src in clips:
            y = librosa.resample(x, orig_sr=20000, target_sr=PSR)
            pk = np.abs(y).max()
            if pk > 0.9: y = y * 0.9 / pk              # avoid clipping in the file itself
            rows.append(dict(start=pos / PSR, role=role, label={"bg": 0, "hn": 0, "cs": 1, "gs": 2}[role], src=src))
            parts += [y.astype(np.float32), np.zeros(int(GAP * PSR), np.float32)]; pos += len(y) + int(GAP * PSR)
        end_chirp = pos / PSR; parts += [c, np.zeros(int(LEAD * PSR), np.float32)]
        sf.write(f"{D}/{split}_playlist.wav", np.concatenate(parts), PSR)
        m = pd.DataFrame(rows); m.attrs = {}; m.to_csv(f"{D}/{split}_manifest.csv", index=False)
        open(f"{D}/{split}_chirps.txt", "w").write(f"{LEAD} {end_chirp}\n")
        dur = sum(len(p) for p in parts) / PSR
        print(f"{split}: {len(rows)} clips, {dur/60:.1f} min -> {D}/{split}_playlist.wav  {m.role.value_counts().to_dict()}")

def record(split, tag):
    import sounddevice as sd
    dur = sf.info(f"{D}/{split}_playlist.wav").duration + 10
    os.makedirs(f"{D}/recordings", exist_ok=True)
    input(f"Recording {dur/60:.1f} min. Press Enter, then START the {split} playlist on your phone within a few seconds...")
    y = sd.rec(int(dur * 48000), samplerate=48000, channels=1, dtype="float32"); sd.wait()
    f = f"{D}/recordings/{split}_{tag}.wav"; sf.write(f, y, 48000); print("saved", f)

def find(sig, tmpl, lo, hi):
    seg = sig[lo:hi]; c = fftconvolve(seg, tmpl[::-1], mode="valid")
    k = int(np.argmax(np.abs(c))); return lo + k, float(np.abs(c[k]) / (np.linalg.norm(tmpl) * np.linalg.norm(seg[k:k + len(tmpl)]) + 1e-12))

def slice_(split, tag):
    from audio_preprocessing import fix_length
    y, sr = sf.read(f"{D}/recordings/{split}_{tag}.wav"); y = y.mean(1) if y.ndim > 1 else y
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=20000)
    m = pd.read_csv(f"{D}/{split}_manifest.csv"); t0, t1 = map(float, open(f"{D}/{split}_chirps.txt").read().split())
    c = sync_chirp(20000)
    a, qa = find(y, c, 0, min(len(y), int((t0 + 30) * 20000)))                       # start chirp within first 30 s
    exp_gap = (t1 - t0) * 20000
    b, qb = find(y, c, int(a + exp_gap * 0.995), min(len(y), int(a + exp_gap * 1.005 + len(c) * 2)))
    scale = (b - a) / exp_gap
    print(f"start chirp @{a/20000:.2f}s (match {qa:.2f}), end chirp match {qb:.2f}, clock ratio {scale:.6f}")
    if qa < 0.3 or qb < 0.3:
        print("WARNING: weak chirp match - was the whole playlist recorded? Results may be misaligned.")
    out, rows = [], []
    for _, r in m.iterrows():
        s = int(a + (r.start - t0) * 20000 * scale)
        out.append(fix_length(y[s:s + 100000])); rows.append(dict(label=r.label, role=r.role, src=r.src, session=tag))
    od = f"datasets/replayed/{split}_{tag}"; os.makedirs(od, exist_ok=True)
    np.save(f"{od}/audio.npy", np.stack(out).astype(np.float32)); pd.DataFrame(rows).to_csv(f"{od}/meta.csv", index=False)
    lv = [np.sqrt(np.mean(x ** 2)) for x in out]
    print(f"wrote {len(out)} labelled clips to {od}  (median clip RMS {np.median(lv):.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["make", "record", "slice"])
    ap.add_argument("split", nargs="?", choices=["train", "test"]); ap.add_argument("--tag", default="session1")
    a = ap.parse_args(); a.tag = a.tag.strip(" .,;")
    if a.cmd == "make": make()
    elif a.cmd == "record": record(a.split, a.tag)
    else: slice_(a.split, a.tag)
