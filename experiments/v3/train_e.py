"""V3-e: V3-d recipe trained FROM SCRATCH (INT8-friendly). python experiments/v3/train_e.py SEED
Data = V3-c set (ext data, neg mixes, playback sim, gain x0.03-3, loud beds) + real replay train sessions x5.
Then INT8 conversion (reference-kernel validation) on every test set incl. the real replay test session."""
import sys, os, json, glob, time
sys.path.insert(0, "src")
SEED = int(sys.argv[1]); t0 = time.time()
DISTILL = len(sys.argv) > 2 and sys.argv[2] == "distill"   # V3-f: soft targets from the V3-d ensemble
TAG = f"{'F' if DISTILL else 'E'}_s{SEED}"
import numpy as np, pandas as pd
import dataset_b
from dataset_b import build_train, condition_set, _fsc, _ext, mixable, mix_retry, rgain
from dataset import features, gain
Xc, yc, _ = build_train(SEED, ext=True, negmix=True, v3c=True)
rng = np.random.default_rng(SEED + 500)
bgs = [x for x, _ in _fsc([1, 2, 3])["bg"] + _ext("train")["bg"] if mixable(x)]
dataset_b.GAIN_LO = 0.3; aud, yr = [], []
for d in sorted(glob.glob("datasets/replayed/train_*")):
    A = np.load(f"{d}/audio.npy"); m = pd.read_csv(f"{d}/meta.csv")
    for x, l in zip(A, m.label):
        aud.append(x); yr.append(l)
        for _ in range(2): aud.append(gain(x, rgain(rng))); yr.append(l)
        for _ in range(2):
            mx, _ = mix_retry(x, bgs, rng.uniform(-5, 20), rng); aud.append(mx); yr.append(l)
Xr = features(aud); del aud
Xtr = np.concatenate([Xc, Xr]); ytr = np.concatenate([yc, np.array(yr)]); del Xc, Xr
Xv1, mv1 = condition_set(_fsc([4]), SEED + 100, snrs=(-5, 5, 15), per=1)
Xv2, mv2 = condition_set(_ext("val"), SEED + 200, snrs=(-5, 5, 15), per=1)
Xva = np.concatenate([Xv1, Xv2]); yva = np.concatenate([mv1.label, mv2.label]).astype(int); del Xv1, Xv2
print(TAG, "train", Xtr.shape, np.bincount(ytr), f"{time.time()-t0:.0f}s", flush=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from scoring import score, score_simple
from evaluation import TFLiteModel
tf.keras.utils.set_random_seed(SEED)
def get_model():
    inp = tf.keras.Input(shape=(128, 196, 3))
    x = tf.keras.layers.Rescaling(2.0 / 140.0, (200.0 / 140.0) - 1.0, name="mel_normalization")(inp)
    base = tf.keras.applications.MobileNetV3Small(input_shape=(128, 196, 3), weights=None, include_top=False, include_preprocessing=False)
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization): l.momentum = 0.9
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base(x))
    return tf.keras.Model(inp, tf.keras.layers.Dense(3, activation="softmax", name="prediction")(x), name="MobileNetV3_Small_Rainforest")
EPOCHS = int(os.environ.get("EPOCHS", 30))  # seed 0 of V3-f used 30; seeds 1-2 use 45
sched = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, max(1, len(Xtr) // 32), 0.5 ** (1 / EPOCHS), staircase=True)
model = get_model(); model.compile(optimizer=tf.keras.optimizers.Adam(sched), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw = dict(zip([0, 1, 2], compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=ytr)))
if DISTILL:
    teach = np.mean([tf.keras.models.load_model(f"models/MobileNetV3_Small_Rainforest_V3b_D_s{s}/best.keras", compile=False).predict(Xtr, batch_size=64, verbose=0) for s in range(3)], axis=0)
    print("teacher agreement with hard labels", round(float((teach.argmax(1) == ytr).mean()), 3), flush=True)
    ytarget = 0.5 * np.eye(3, dtype=np.float32)[ytr] + 0.5 * teach.astype(np.float32); del teach
    sw = np.array([cw[int(c)] for c in ytr], dtype=np.float32)
    model.compile(optimizer=tf.keras.optimizers.Adam(sched), loss="categorical_crossentropy", metrics=["accuracy"])
out = f"models/MobileNetV3_Small_Rainforest_V3b_{TAG}"; os.makedirs(out, exist_ok=True); best = {"v": -1}
class BalAcc(tf.keras.callbacks.Callback):
    def on_epoch_end(self, ep, logs=None):
        v = balanced_accuracy_score(yva, self.model.predict(Xva, batch_size=64, verbose=0).argmax(1)); logs["val_balacc"] = v
        if v > best["v"]: best.update(v=v, ep=ep); self.model.save(f"{out}/best.keras")
        print(f"   ep{ep} val_balacc {v:.3f} (best {best['v']:.3f} @ep{best['ep']}) t={time.time()-t0:.0f}s", flush=True)
fit_kw = dict(x=Xtr, y=ytarget, sample_weight=sw) if DISTILL else dict(x=Xtr, y=ytr, class_weight=cw)
model.fit(**fit_kw, batch_size=32, epochs=EPOCHS, verbose=0,
          callbacks=[BalAcc(), tf.keras.callbacks.EarlyStopping("val_balacc", mode="max", patience=8, verbose=1)])
del Xtr
bm = tf.keras.models.load_model(f"{out}/best.keras", compile=False)
# INT8 conversion (calibration = validation conditions + replay TRAIN clips, no test data)
cr = np.random.default_rng(42)
Xrt = features(list(np.load("datasets/replayed/train_phone_50cm/audio.npy")))
cal = np.concatenate([Xva[cr.choice(len(Xva), 250, replace=False)], Xrt[cr.choice(len(Xrt), 50, replace=False)]]); del Xrt
conv = tf.lite.TFLiteConverter.from_keras_model(bm); conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = lambda: ([c[None].astype(np.float32)] for c in cal)
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
tfl = f"{out}/int8_candidate.tflite"; open(tfl, "wb").write(conv.convert()); q = TFLiteModel(tfl, reference=True)
res = {}
def evalset(name, X, meta, kind):
    pf = bm.predict(X, batch_size=32, verbose=0); pq = q.predict(X)
    np.save(f"experiments/v3/results/{TAG}_{name}_probs.npy", pf); np.save(f"experiments/v3/results/{TAG}_{name}_int8_probs.npy", pq)
    r = dict(agreement=float((pf.argmax(1) == pq.argmax(1)).mean()), mean_abs_dp=float(np.abs(pf - pq).mean()))
    for t, p in (("float", pf), ("int8", pq)):
        if kind == "full": r[t] = {k: float(v[0]) for k, v in score(meta, p).items()}
        elif kind == "simple": r[t] = score_simple(meta, p)
        else:
            a = p.argmax(1); r[t] = dict(gs=float((a[meta.role == "gs"] == 2).mean()), cs=float((a[meta.role == "cs"] == 1).mean()),
                                         cs_to_gs=float((a[meta.role == "cs"] == 2).mean()), neg_FA=float((a[meta.role.isin(["hn", "bg"])] != 0).mean()))
    res[name] = r; print(name, "agreement", round(r["agreement"], 3), flush=True)
for name in ("eval_v1", "eval_ext"):
    evalset(name, np.load(f"datasets/{name}/X.npy"), pd.read_csv(f"datasets/{name}/meta.csv"), "full")
evalset("eval_stress", np.load("datasets/eval_stress/X.npy"), pd.read_csv("datasets/eval_stress/meta.csv"), "simple")
evalset("replay_test", features(list(np.load("datasets/replayed/test_phone_50cm/audio.npy"))), pd.read_csv("datasets/replayed/test_phone_50cm/meta.csv"), "replay")
json.dump(dict(cfg=TAG[0], seed=SEED, best_epoch=best["ep"], best_val_balacc=best["v"], scores=res), open(f"experiments/v3/results/{TAG}.json", "w"), indent=1)
print("DONE", TAG, f"{time.time()-t0:.0f}s")
