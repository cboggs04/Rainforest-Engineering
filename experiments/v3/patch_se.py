"""Function-preserving patch for dead squeeze-excite blocks (ReLU output == 0 for all inputs):
conv1 -> kernel 0, bias c (ReLU output becomes constant c); conv2 bias -= W2^T c  => identical gate, no zero-range tensor.
python experiments/v3/patch_se.py SRC.keras DST.keras"""
import sys, os
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
from dataset_b import condition_set, _fsc
src, dst = sys.argv[1], sys.argv[2]
Xv, _ = condition_set(_fsc([4]), 7, snrs=(-5, 5, 15), per=1); X = Xv[np.random.default_rng(0).choice(len(Xv), 300, replace=False)]; del Xv
m = tf.keras.models.load_model(src, compile=False)
base = [l for l in m.layers if isinstance(l, tf.keras.Model)][0]
relus = [l.name for l in base.layers if l.name.endswith("_squeeze_excite_relu")]
probe = tf.keras.Model(base.input, [base.get_layer(n).output for n in relus])
acts = probe.predict(m.get_layer("mel_normalization")(X), batch_size=50, verbose=0)
ref = m.predict(X, batch_size=50, verbose=0); C = 1.0; patched = []
for n, a in zip(relus, acts):
    dead = np.abs(np.asarray(a)).reshape(len(X), -1).max(0) < 1e-7       # per-channel dead on all calibration inputs
    if not dead.any(): continue
    pre = n[: -len("_relu")]
    c1, c2 = base.get_layer(pre + "_conv"), base.get_layer(pre + "_conv_1")
    k1, b1 = c1.get_weights(); k2, b2 = c2.get_weights()
    k1[..., dead] = 0.0; b1[dead] = C
    b2 = b2 - C * k2[0, 0, dead, :].sum(0)
    c1.set_weights([k1, b1]); c2.set_weights([k2, b2]); patched.append((pre, int(dead.sum()), dead.size))
new = m.predict(X, batch_size=50, verbose=0)
print("patched:", patched)
print(f"float output change on 300 val clips: max|dp| {np.abs(new-ref).max():.2e}, argmax changes {(new.argmax(1)!=ref.argmax(1)).sum()}")
m.save(dst); print("saved", dst)
