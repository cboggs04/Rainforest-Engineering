"""Build the leak-free evaluation set from OFFICIAL FSC22 fold 5 only (seeded).
Writes datasets/eval_v1/{X.npy, meta.csv}. Nothing from fold 5 may be used for training/tuning."""
import sys, os, pickle
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from audio_preprocessing import reference_mel, to_3class
from augmentation import mix_event

rng = np.random.default_rng(1234)
meta = pd.read_csv("datasets/fsc22/fsc22/FSC22.csv")
aud = pickle.load(open("datasets/fsc22/Pickle Files/audios_5_20", "rb"))
f5 = meta[meta.fold == 5]
lab = {i: to_3class(meta.target[i] - 1) for i in f5.index}
cs = [i for i in f5.index if lab[i] == 1]; gs = [i for i in f5.index if lab[i] == 2]
bg = [i for i in f5.index if lab[i] == 0]; hn = [i for i in f5.index if lab[i] == -1]
print(len(cs), len(gs), len(bg), len(hn))
A = lambda i: aud[i][0].astype(np.float32)
rows, feats = [], []
def add(x, label, cond, src, bgsrc="", snr=np.nan, ach=np.nan, start=np.nan, gain=1.0):
    feats.append(reference_mel(x)); rows.append(dict(label=label, cond=cond, src=meta.filename[src], bg=bgsrc, snr=snr, achieved=ach, start_s=start, gain=gain))
for i in bg: add(A(i), 0, "clean_bg", i)
for i in hn: add(A(i), 0, "hard_neg", i)               # label 0 = must NOT alert
for i in cs: add(A(i), 1, "clean_cs", i)
for i in gs: add(A(i), 2, "clean_gs", i)
for g in (0.3, 3.0):                                     # gain robustness
    for i in cs: add(np.clip(A(i) * g, -1, 1), 1, "gain_cs", i, gain=g)
    for i in gs: add(np.clip(A(i) * g, -1, 1), 2, "gain_gs", i, gain=g)
for snr in (-5, 0, 5, 10, 15):
    for i in cs + gs:
        L = 1 if i in cs else 2
        for k in range(4):
            b = int(rng.choice(bg)); x, ach, s, e = mix_event(A(i), A(b), snr, rng=rng)
            assert abs(ach - snr) < 0.01
            add(x, L, "mix_cs" if L == 1 else "mix_gs", i, meta.filename[b], snr, ach, s / 20000)
X = np.stack(feats).astype(np.float32); df = pd.DataFrame(rows)
os.makedirs("datasets/eval_v1", exist_ok=True)
np.save("datasets/eval_v1/X.npy", X); df.to_csv("datasets/eval_v1/meta.csv", index=False)
print(X.shape); print(df.cond.value_counts())
