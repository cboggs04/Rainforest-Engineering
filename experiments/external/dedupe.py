"""Flag external target clips that appear (near-)verbatim inside FSC22 same-class clips (normalised cross-correlation)."""
import sys, pickle
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.signal import fftconvolve
from augmentation import active_bounds
A = np.load("datasets/external/prepared/audio.npy"); df = pd.read_csv("datasets/external/prepared/meta.csv")
fs = pickle.load(open("datasets/fsc22/Pickle Files/audios_5_20", "rb"))
meta = pd.read_csv("datasets/fsc22/fsc22/FSC22.csv")
ref = {"cs": [fs[i][0] for i in meta.index if meta.target[i] == 11], "gs": [fs[i][0] for i in meta.index if meta.target[i] == 15]}
def best_ncc(tmpl, x):
    num = fftconvolve(x, tmpl[::-1], mode="valid")
    cs = np.cumsum(np.r_[0, x.astype(np.float64) ** 2]); L = len(tmpl)
    den = np.sqrt(np.maximum(cs[L:] - cs[:-L], 1e-12)) * np.sqrt((tmpl.astype(np.float64) ** 2).sum() + 1e-12)
    return float(np.max(np.abs(num) / den))
best = []
for k, r in df.iterrows():
    if r.role not in ("cs", "gs"): best.append(np.nan); continue
    s, e = active_bounds(A[k], 30); t = A[k][s:e]
    if len(t) > 20000: m = (s + e) // 2; t = A[k][m - 10000:m + 10000]   # 1 s centre for long clips
    if len(t) < 2000: t = A[k][max(0, s - 1000):s + 3000]
    best.append(max(best_ncc(t, x) for x in ref[r.role]))
df["fsc22_ncc"] = best; df["dup_fsc22"] = df.fsc22_ncc > 0.8
df.to_csv("datasets/external/prepared/meta.csv", index=False)
d = df[df.role.isin(["cs", "gs"])]
print(d.groupby("role").fsc22_ncc.describe().round(3)); print("flagged duplicates:\n", d.groupby("role").dup_fsc22.sum())
print(d.sort_values("fsc22_ncc", ascending=False).head(8)[["src", "fsc22_ncc"]])
