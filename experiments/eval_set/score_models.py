import sys, os
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
from scoring import score
from evaluation import TFLiteModel
meta = pd.read_csv("datasets/eval_v1/meta.csv"); X = np.load("datasets/eval_v1/X.npy")
M = {"Original": "models/MobileNetV3_Small_Rainforest_3Class_BN09/MobileNetV3_Small_Rainforest_3Class_BN09_base_fold5.keras",
     "V1": "models/MobileNetV3_Small_Rainforest_Finetuned_V1/MobileNetV3_Finetuned_V1_best.keras",
     "V2": "models/MobileNetV3_Small_Rainforest_Finetuned_V2/MobileNetV3_Finetuned_V2_best.keras"}
res = {}
for n, p in M.items():
    res[n] = score(meta, tf.keras.models.load_model(p, compile=False).predict(X, batch_size=32, verbose=0))
res["V1-INT8"] = score(meta, TFLiteModel("models/ESP32_Deployment_V1/MobileNetV3_Rainforest_V1_INT8.tflite").predict(X))
tab = pd.DataFrame({n: {k: f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" if v[1] == v[1] else f"{v[0]:.2f}" for k, v in r.items()} for n, r in res.items()})
print(tab.to_string())
tab.to_csv(sys.argv[1] if len(sys.argv) > 1 else "experiments/eval_set/baseline_scores.csv")
