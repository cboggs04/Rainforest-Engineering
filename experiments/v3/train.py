"""V3-a ablation: python experiments/v3/train.py {clean|hn|mix}"""
import sys, os, json, time
sys.path.insert(0, "src")
variant = sys.argv[1]; SEED = 42
import numpy as np
from dataset import build
t0 = time.time()
Xtr, ytr, itr = build([1, 2, 3], variant, SEED)
Xva, yva, iva = build([4], variant, SEED + 1, val=True)
print(variant, "train", Xtr.shape, np.bincount(ytr), "val", Xva.shape, np.bincount(yva), f"{time.time()-t0:.0f}s", flush=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf, pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from scoring import score
tf.keras.utils.set_random_seed(SEED)

def get_model():  # identical to notebook get_model()
    inp = tf.keras.Input(shape=(128, 196, 3))
    x = tf.keras.layers.Rescaling(2.0 / 140.0, (200.0 / 140.0) - 1.0, name="mel_normalization")(inp)
    base = tf.keras.applications.MobileNetV3Small(input_shape=(128, 196, 3), weights=None, include_top=False, include_preprocessing=False)
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization): l.momentum = 0.9
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base(x))
    return tf.keras.Model(inp, tf.keras.layers.Dense(3, activation="softmax", name="prediction")(x), name="MobileNetV3_Small_Rainforest")

EPOCHS, BS = 60, 32
steps = max(1, len(Xtr) // BS)
sched = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, steps, (5e-4 / 1e-3) ** (1 / EPOCHS), staircase=True)
model = get_model()
model.compile(optimizer=tf.keras.optimizers.Adam(sched), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw = dict(zip([0, 1, 2], compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=ytr)))
out = f"models/MobileNetV3_Small_Rainforest_V3a_{variant}"; os.makedirs(out, exist_ok=True)
best = {"v": -1}
class BalAcc(tf.keras.callbacks.Callback):
    def on_epoch_end(self, ep, logs=None):
        v = balanced_accuracy_score(yva, self.model.predict(Xva, batch_size=64, verbose=0).argmax(1))
        logs["val_balacc"] = v
        if v > best["v"]:
            best.update(v=v, ep=ep); self.model.save(f"{out}/best.keras")
        print(f"   val_balacc {v:.3f} (best {best['v']:.3f} @ep{best.get('ep')})", flush=True)
hist = model.fit(Xtr, ytr, batch_size=BS, epochs=EPOCHS, validation_data=(Xva, yva), class_weight=cw, verbose=2,
                 callbacks=[BalAcc(), tf.keras.callbacks.EarlyStopping("val_balacc", mode="max", patience=12, verbose=1)])
best_model = tf.keras.models.load_model(f"{out}/best.keras", compile=False)
meta = pd.read_csv("datasets/eval_v1/meta.csv"); Xte = np.load("datasets/eval_v1/X.npy")
prob = best_model.predict(Xte, batch_size=32, verbose=0)
res = score(meta, prob)
os.makedirs("experiments/v3/results", exist_ok=True)
pd.Series({k: v[0] for k, v in res.items()}).to_csv(f"experiments/v3/results/{variant}_scores.csv")
json.dump(dict(variant=variant, best_epoch=best["ep"], best_val_balacc=best["v"], seed=SEED, n_train=len(ytr), class_counts=np.bincount(ytr).tolist(),
               scores={k: list(map(float, v)) for k, v in res.items()}), open(f"experiments/v3/results/{variant}.json", "w"), indent=1)
np.save(f"experiments/v3/results/{variant}_test_probs.npy", prob)
print("DONE", variant, f"{time.time()-t0:.0f}s")
