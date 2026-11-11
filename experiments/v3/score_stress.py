"""Score existing checkpoints on datasets/eval_stress."""
import sys, os, json
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
from scoring import score_simple
meta = pd.read_csv("datasets/eval_stress/meta.csv"); X = np.load("datasets/eval_stress/X.npy"); out = {}
for tag in sys.argv[1:]:
    p = f"models/MobileNetV3_Small_Rainforest_V3b_{tag}/best.keras" if tag.startswith("B") else tag
    out[tag] = score_simple(meta, tf.keras.models.load_model(p, compile=False).predict(X, batch_size=32, verbose=0))
json.dump(out, open("experiments/v3/results/stress_B2.json", "w"), indent=1)
print(pd.DataFrame(out).round(2).to_string())
