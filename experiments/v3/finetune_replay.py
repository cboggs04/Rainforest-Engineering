"""V3-d: fine-tune V3-c (cfg C) on REAL replayed train clips. python experiments/v3/finetune_replay.py SEED
Data = full V3-c training set (rehearsal, same seed) + real replay train sessions x5 (orig, 2 gain copies, 2 rainforest mixes).
Epoch chosen on the standard FSC22-fold4 + external-val set; real replay TEST session is never used here."""
import sys, os, json, glob, time
sys.path.insert(0, "src")
SEED = int(sys.argv[1]); t0 = time.time()
import numpy as np, pandas as pd
from dataset_b import build_train, condition_set, _fsc, _ext, mixable, mix_retry, rgain
from dataset import features, gain
import dataset_b
Xc, yc, _ = build_train(SEED, ext=True, negmix=True, v3c=True)
rng = np.random.default_rng(SEED + 500)
bgs = [x for x, _ in _fsc([1, 2, 3])["bg"] + _ext("train")["bg"] if mixable(x)]
dataset_b.GAIN_LO = 0.3
aud, yr = [], []
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
print("train", Xtr.shape, np.bincount(ytr), "replay-derived", len(yr), f"{time.time()-t0:.0f}s", flush=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from scoring import score, score_simple
tf.keras.utils.set_random_seed(SEED)
model = tf.keras.models.load_model(f"models/MobileNetV3_Small_Rainforest_V3b_C_s{SEED}/best.keras", compile=False)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw = dict(zip([0, 1, 2], compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=ytr)))
out = f"models/MobileNetV3_Small_Rainforest_V3b_D_s{SEED}"; os.makedirs(out, exist_ok=True); best = {"v": -1}
class BalAcc(tf.keras.callbacks.Callback):
    def on_epoch_end(self, ep, logs=None):
        v = balanced_accuracy_score(yva, self.model.predict(Xva, batch_size=64, verbose=0).argmax(1))
        if v > best["v"]: best.update(v=v, ep=ep); self.model.save(f"{out}/best.keras")
        print(f"   ep{ep} val_balacc {v:.3f} (best {best['v']:.3f} @ep{best['ep']}) t={time.time()-t0:.0f}s", flush=True)
model.fit(Xtr, ytr, batch_size=32, epochs=8, class_weight=cw, verbose=0, callbacks=[BalAcc()])
del Xtr
bm = tf.keras.models.load_model(f"{out}/best.keras", compile=False); res = {}
for name in ("eval_v1", "eval_ext"):
    meta = pd.read_csv(f"datasets/{name}/meta.csv"); X = np.load(f"datasets/{name}/X.npy")
    p = bm.predict(X, batch_size=32, verbose=0); np.save(f"experiments/v3/results/D_s{SEED}_{name}_probs.npy", p)
    res[name] = {k: list(map(float, v)) for k, v in score(meta, p).items()}; del X
meta = pd.read_csv("datasets/eval_stress/meta.csv"); X = np.load("datasets/eval_stress/X.npy")
p = bm.predict(X, batch_size=32, verbose=0); np.save(f"experiments/v3/results/D_s{SEED}_eval_stress_probs.npy", p)
res["eval_stress"] = score_simple(meta, p)
json.dump(dict(cfg="D", seed=SEED, best_epoch=best["ep"], best_val_balacc=best["v"], scores=res), open(f"experiments/v3/results/D_s{SEED}.json", "w"), indent=1)
print("DONE D", SEED, f"{time.time()-t0:.0f}s")
