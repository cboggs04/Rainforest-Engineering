"""Independent external test set from held-out EXTERNAL test sources (never in any training split)."""
import sys, os
sys.path.insert(0, "src")
import numpy as np
from dataset_b import _ext, condition_set
P = _ext("test"); print({k: len(v) for k, v in P.items()})
X, meta = condition_set(P, seed=99)
os.makedirs("datasets/eval_ext", exist_ok=True); np.save("datasets/eval_ext/X.npy", X); meta.to_csv("datasets/eval_ext/meta.csv", index=False)
print(X.shape); print(meta.cond.value_counts())
