"""Stress test from TEST sources only (FSC22 fold 5 + external test): replayed (speaker/room/mic) audio,
very quiet events, and loud noise beds. Uses a different seed than training."""
import sys, os
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from dataset_b import _fsc, _ext, features, mixable, mix_retry, to_rms
from dataset import gain
from augmentation import simulate_playback
rng = np.random.default_rng(2024)
F, E = _fsc([5]), _ext("test"); P = {k: F[k] + E[k] for k in F}
bgs = [x for x, _ in P["bg"] if mixable(x)]
aud, rows = [], []
def add(x, l, cond, src): aud.append(x); rows.append(dict(label=l, cond=cond, src=src))
for grp, l in (("cs", 1), ("gs", 2)):
    for x, s in P[grp]:
        for _ in range(2): add(simulate_playback(x, rng), l, f"replayed_{grp}", s)
        for g in (0.03, 0.1): add(gain(x, g), l, f"quiet_{grp}", s)
        if grp == "cs":
            for _ in range(3):
                m, _ = mix_retry(x, bgs, rng.uniform(-5, 5), rng); add(to_rms(m, 0.1), 1, "loudbed_cs", s)
neg = P["hn"] + P["bg"]
for j in rng.choice(len(neg), 250, replace=False): add(simulate_playback(neg[j][0], rng), 0, "replayed_neg", neg[j][1])
for j in rng.choice(len(neg), 150, replace=False): add(gain(neg[j][0], 0.1), 0, "quiet_neg", neg[j][1])
for j in rng.choice(len(bgs), 100, replace=False): add(to_rms(bgs[j], 0.1), 0, "loud_bg", "bed")
X = features(aud); meta = pd.DataFrame(rows)
os.makedirs("datasets/eval_stress", exist_ok=True); np.save("datasets/eval_stress/X.npy", X); meta.to_csv("datasets/eval_stress/meta.csv", index=False)
print(X.shape); print(meta.cond.value_counts().to_dict())
