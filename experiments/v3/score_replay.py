"""Score checkpoints on a REAL replay session: python experiments/v3/score_replay.py test_phone_50cm TAG [TAG...]
TAG = V3b model tag (e.g. B2_s0, C_s1) or a .keras path."""
import sys, os, json
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
from dataset import features
sess = sys.argv[1]; d = f"datasets/replayed/{sess}"
A = np.load(f"{d}/audio.npy"); m = pd.read_csv(f"{d}/meta.csv"); X = features(list(A))
rows = []
for tag in sys.argv[2:]:
    path = tag if tag.endswith(".keras") else f"models/MobileNetV3_Small_Rainforest_V3b_{tag}/best.keras"
    p = tf.keras.models.load_model(path, compile=False).predict(X, verbose=0, batch_size=32); a = p.argmax(1)
    for t in (0.5, 0.8):
        pr = np.where((a != 0) & (p.max(1) >= t), a, 0)
        rows.append(dict(model=os.path.basename(os.path.dirname(path)).replace("MobileNetV3_Small_Rainforest_", ""), thr=t,
            cs_recall=(pr[m.role == "cs"] == 1).mean(), gs_recall=(pr[m.role == "gs"] == 2).mean(),
            cs_to_gs=(pr[m.role == "cs"] == 2).mean(), neg_false_alarm=(pr[m.role.isin(["hn", "bg"])] != 0).mean()))
r = pd.DataFrame(rows); print(f"{sess}: {m.role.value_counts().to_dict()}"); print(r.round(2).to_string(index=False))
r.to_csv(f"experiments/v3/results/replay_{sess}_{'_'.join(t.replace('/','-')[-20:] for t in sys.argv[2:4])}.csv", index=False)
