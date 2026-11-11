"""V3-b: python experiments/v3/train_b.py {B0|B1|B2} SEED
B0 = FSC22 + external data, balanced mixes, scratch | B1 = B0 + non-target-event negative mixes | B2 = B1 + ImageNet init"""
import sys, os, json, time
sys.path.insert(0, "src")
cfg, SEED = sys.argv[1], int(sys.argv[2])
C = dict(B0=dict(negmix=False, pre=False), B1=dict(negmix=True, pre=False), B2=dict(negmix=True, pre=True), C=dict(negmix=True, pre=True, v3c=True))[cfg]
import numpy as np, pandas as pd
from dataset_b import build_train, condition_set, _fsc, _ext
t0 = time.time()
Xtr, ytr, itr = build_train(SEED, ext=True, negmix=C["negmix"], v3c=C.get("v3c", False))
Xv1, mv1 = condition_set(_fsc([4]), SEED + 100, snrs=(-5, 5, 15), per=1)
Xv2, mv2 = condition_set(_ext("val"), SEED + 200, snrs=(-5, 5, 15), per=1)
Xva = np.concatenate([Xv1, Xv2]); yva = np.concatenate([mv1.label, mv2.label]).astype(int)
print(cfg, SEED, "train", Xtr.shape, np.bincount(ytr), itr.kind.value_counts().to_dict(), "val", Xva.shape, np.bincount(yva), f"{time.time()-t0:.0f}s", flush=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from scoring import score
tf.keras.utils.set_random_seed(SEED)
def get_model(pre):
    inp = tf.keras.Input(shape=(128, 196, 3))
    x = tf.keras.layers.Rescaling(2.0 / 140.0, (200.0 / 140.0) - 1.0, name="mel_normalization")(inp)
    base = tf.keras.applications.MobileNetV3Small(input_shape=(128, 196, 3), weights="imagenet" if pre else None, include_top=False, include_preprocessing=False)
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization): l.momentum = 0.9
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base(x))
    return tf.keras.Model(inp, tf.keras.layers.Dense(3, activation="softmax", name="prediction")(x), name="MobileNetV3_Small_Rainforest")
EPOCHS, BS = 30, 32; lr0 = 3e-4 if C["pre"] else 1e-3
sched = tf.keras.optimizers.schedules.ExponentialDecay(lr0, max(1, len(Xtr) // BS), 0.5 ** (1 / EPOCHS), staircase=True)
model = get_model(C["pre"]); model.compile(optimizer=tf.keras.optimizers.Adam(sched), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw = dict(zip([0, 1, 2], compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=ytr)))
tag = f"{cfg}_s{SEED}"; out = f"models/MobileNetV3_Small_Rainforest_V3b_{tag}"; os.makedirs(out, exist_ok=True)
best = {"v": -1}
class BalAcc(tf.keras.callbacks.Callback):
    def on_epoch_end(self, ep, logs=None):
        v = balanced_accuracy_score(yva, self.model.predict(Xva, batch_size=64, verbose=0).argmax(1)); logs["val_balacc"] = v
        if v > best["v"]: best.update(v=v, ep=ep); self.model.save(f"{out}/best.keras")
        print(f"   ep{ep} val_balacc {v:.3f} (best {best['v']:.3f} @ep{best.get('ep')}) t={time.time()-t0:.0f}s", flush=True)
model.fit(Xtr, ytr, batch_size=BS, epochs=EPOCHS, class_weight=cw, verbose=0,
          callbacks=[BalAcc(), tf.keras.callbacks.EarlyStopping("val_balacc", mode="max", patience=8, verbose=1)])
del Xtr; bm = tf.keras.models.load_model(f"{out}/best.keras", compile=False)
os.makedirs("experiments/v3/results", exist_ok=True); res = {}
for name in ("eval_v1", "eval_ext"):
    meta = pd.read_csv(f"datasets/{name}/meta.csv"); X = np.load(f"datasets/{name}/X.npy")
    p = bm.predict(X, batch_size=32, verbose=0); np.save(f"experiments/v3/results/{tag}_{name}_probs.npy", p)
    res[name] = {k: list(map(float, v)) for k, v in score(meta, p).items()}
from scoring import score_simple
meta = pd.read_csv("datasets/eval_stress/meta.csv"); X = np.load("datasets/eval_stress/X.npy")
p = bm.predict(X, batch_size=32, verbose=0); np.save(f"experiments/v3/results/{tag}_eval_stress_probs.npy", p)
res["eval_stress"] = score_simple(meta, p)
json.dump(dict(cfg=cfg, seed=SEED, best_epoch=best["ep"], best_val_balacc=best["v"], n_train=len(ytr), scores=res), open(f"experiments/v3/results/{tag}.json", "w"), indent=1)
print("DONE", tag, f"{time.time()-t0:.0f}s")
