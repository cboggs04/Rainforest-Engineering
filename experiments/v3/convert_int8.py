"""Full-INT8 conversion + validation of the V3 candidate. Writes models/ESP32_Deployment_V3/ (never overwrites)."""
import sys, os, json
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
from dataset_b import condition_set, _fsc, _ext
from dataset import features
# usage: python experiments/v3/convert_int8.py SRC.keras NAME   (e.g. ... V3b_F_s0/best.keras V3f)
SRC, NAME = sys.argv[1], sys.argv[2]
OUT = "models/ESP32_Deployment_V3"; TFL = f"{OUT}/MobileNetV3_Rainforest_{NAME}_INT8.tflite"; HDR = f"{OUT}/rainforest_model_{NAME}.h"
os.makedirs(OUT, exist_ok=True)
EXISTS = os.path.exists(TFL)  # never overwritten: if present, only re-validate it
# calibration: validation conditions (FSC22 fold 4 + external val) + real replay TRAIN clips; no test data
rng = np.random.default_rng(42)
Xv1, _ = condition_set(_fsc([4]), 7, snrs=(-5, 5, 15), per=1)
Xv2, _ = condition_set(_ext("val"), 8, snrs=(-5, 5, 15), per=1)
Xr = features(list(np.load("datasets/replayed/train_phone_50cm/audio.npy")))
cal = np.concatenate([Xv1[rng.choice(len(Xv1), 150, replace=False)], Xv2[rng.choice(len(Xv2), 100, replace=False)], Xr[rng.choice(len(Xr), 50, replace=False)]])
del Xv1, Xv2, Xr
import tensorflow as tf
model = tf.keras.models.load_model(SRC, compile=False)
if EXISTS:
    blob = open(TFL, "rb").read()
else:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([c[None].astype(np.float32)] for c in cal)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
    blob = conv.convert(); open(TFL, "wb").write(blob)
it = tf.lite.Interpreter(model_path=TFL, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES); it.allocate_tensors()
zero_scale = [t["name"] for t in it.get_tensor_details() if len(t["quantization_parameters"]["scales"]) and t["quantization_parameters"]["scales"].min() < 1e-9]
ops = sorted({o["op_name"] for o in it._get_ops_details()} - {"DELEGATE"})
i, o = it.get_input_details()[0], it.get_output_details()[0]
FIRMWARE_OPS = {"ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "HARD_SWISH", "MEAN", "MUL", "SOFTMAX"}
info = dict(source=SRC, bytes=len(blob), MiB=round(len(blob) / 2**20, 3), ops=ops, ops_not_in_firmware=sorted(set(ops) - FIRMWARE_OPS), n_ops=len(it._get_ops_details()), zero_scale_tensors=zero_scale,
            input=dict(shape=i["shape"].tolist(), dtype=str(i["dtype"]), scale=float(i["quantization"][0]), zero_point=int(i["quantization"][1])),
            output=dict(shape=o["shape"].tolist(), dtype=str(o["dtype"]), scale=float(o["quantization"][0]), zero_point=int(o["quantization"][1])))
print(json.dumps(info, indent=1))
# float vs INT8 on every test set
from evaluation import TFLiteModel
from scoring import score, score_simple
q = TFLiteModel(TFL, reference=True); val = {}
for n in ("eval_v1", "eval_ext", "eval_stress", "replay_test"):
    if n == "replay_test":
        X = features(list(np.load("datasets/replayed/test_phone_50cm/audio.npy"))); meta = pd.read_csv("datasets/replayed/test_phone_50cm/meta.csv")
    else:
        X = np.load(f"datasets/{n}/X.npy"); meta = pd.read_csv(f"datasets/{n}/meta.csv")
    pf = model.predict(X, batch_size=32, verbose=0); pq = q.predict(X); del X
    agree = float((pf.argmax(1) == pq.argmax(1)).mean()); d = np.abs(pf - pq)
    r = dict(n=len(pf), class_agreement=round(agree, 4), mean_abs_prob_diff=round(float(d.mean()), 4), max_abs_prob_diff=round(float(d.max()), 3))
    if n in ("eval_v1", "eval_ext"):
        for tag, p in (("float", pf), ("int8", pq)):
            s = score(meta, p); r[tag] = {k: round(float(s[k][0]), 3) for k in ("bg false-alarm", "hardneg false-alarm", "mix_gs +5dB recall", "mix_cs -5dB recall", "_cs->gs (all cs rows)")}
    elif n == "eval_stress":
        for tag, p in (("float", pf), ("int8", pq)):
            s = score_simple(meta, p); r[tag] = {k: round(s[k], 3) for k in ("replayed_gs", "quiet_gs", "replayed_neg", "loud_bg")}
    else:
        for tag, p in (("float", pf), ("int8", pq)):
            a = p.argmax(1); r[tag] = dict(gs=round(float((a[meta.role == "gs"] == 2).mean()), 3), cs=round(float((a[meta.role == "cs"] == 1).mean()), 3),
                                           neg_FA=round(float((a[meta.role.isin(["hn", "bg"])] != 0).mean()), 3))
    val[n] = r; print(n, json.dumps(r))
info["validation"] = val; json.dump(info, open(f"{OUT}/conversion_report_{NAME}.json", "w"), indent=1)
# C header: same symbol names as V1 header so the sketch only changes its #include
hx = ",\n".join("    " + ", ".join(f"0x{b:02x}" for b in blob[k:k + 12]) for k in range(0, len(blob), 12))
open(HDR, "w").write(f"// V3 model ({os.path.basename(TFL)}), source {SRC}\n// input scale {info['input']['scale']:.6f} zero_point {info['input']['zero_point']}\n"
                     f"#ifndef RAINFOREST_MODEL_H\n#define RAINFOREST_MODEL_H\n\n#include <stdint.h>\n\nalignas(16) const unsigned char rainforest_model[] = {{\n{hx}\n}};\n\n"
                     f"const unsigned int rainforest_model_len = {len(blob)};\n\n#endif\n")
print("wrote", TFL, HDR)
