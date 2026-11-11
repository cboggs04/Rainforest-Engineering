import sys, os, json, pickle
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf, soundfile as sf, librosa
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from audio_preprocessing import *
from evaluation import TFLiteModel

import pandas as _pd
from audio_preprocessing import pickle_folds
folds = pickle_folds(_pd.read_csv("datasets/fsc22/fsc22/FSC22.csv").target.values - 1)
meta = pd.read_csv("datasets/fsc22/fsc22/FSC22.csv"); y_all = np.array([t - 1 for t in meta.target])
aud = pickle.load(open("datasets/fsc22/Pickle Files/audios_5_20", "rb"))
v1 = tf.keras.models.load_model("models/MobileNetV3_Small_Rainforest_Finetuned_V1/MobileNetV3_Finetuned_V1_best.keras", compile=False)
tfl = TFLiteModel("models/ESP32_Deployment_V1/MobileNetV3_Rainforest_V1_INT8.tflite")
print("INT8 input scale/zp", tfl.scale, tfl.zp, "out", tfl.oscale, tfl.ozp)

idx = [i for i in folds[4] if to_3class(y_all[i]) >= 0]
y = np.array([to_3class(y_all[i]) for i in idx])
Xc = np.stack([reference_mel(aud[i][0]) for i in idx])
print("\n[A] INT8 vs Keras on clean val (n=%d)" % len(y))
pk = v1.predict(Xc, verbose=0); pq = tfl.predict(Xc)
for n, p in (("Keras", pk), ("INT8", pq)):
    cm = confusion_matrix(y, p.argmax(1), labels=[0, 1, 2]); print(n, "acc %.3f balacc %.3f" % ((p.argmax(1) == y).mean(), balanced_accuracy_score(y, p.argmax(1)))); print(cm)
print("Keras/INT8 agreement %.3f  mean|dp| %.4f max|dp| %.3f" % ((pk.argmax(1) == pq.argmax(1)).mean(), np.abs(pk - pq).mean(), np.abs(pk - pq).max()))
dis = np.where(pk.argmax(1) != pq.argmax(1))[0]; print("disagreements:", [(meta.filename[idx[i]], pk[i].round(2).tolist(), pq[i].round(2).tolist()) for i in dis])

print("\n[B] Firmware-emulation features vs reference (clean val audio)")
for pm in ("reflect", "constant"):
    Xf = np.stack([firmware_mel(aud[i][0], pad_mode=pm) for i in idx])
    d = np.abs(Xf - Xc); 
    pf = tfl.predict(Xf)
    print(f"pad={pm}: mean|dB diff| {d.mean():.4f}  max {d.max():.2f}  pred agreement w/ reference INT8: {(pf.argmax(1)==pq.argmax(1)).mean():.3f}  acc {(pf.argmax(1)==y).mean():.3f}")
    # which frames differ
    dd = d.mean(axis=(0, 1, 3)); print("   mean diff by frame (first5,last5):", dd[:5].round(2), dd[-5:].round(2), " middle:", dd[10:-10].mean().round(5))

print("\n[C] Gain sensitivity (INT8, firmware features) on clean-val chainsaw/gunshot clips")
tgt = [k for k in range(len(idx)) if y[k] > 0]
for g in (0.01, 0.03, 0.1, 0.3, 1, 3, 10):
    X = np.stack([firmware_mel(np.clip(aud[idx[k]][0] * g, -1, 1), "constant") for k in tgt])
    p = tfl.predict(X).argmax(1); yt = y[tgt]
    print(f"gain x{g:<5} chainsaw recall {np.mean(p[yt==1]==1):.2f}  gunshot recall {np.mean(p[yt==2]==2):.2f}  cs->gs {np.mean(p[yt==1]==2):.2f}  gs->cs {np.mean(p[yt==2]==1):.2f}")
bgk = [k for k in range(len(idx)) if y[k] == 0]
print("   background->gunshot false alarm vs gain:")
for g in (0.1, 1, 10, 30):
    X = np.stack([firmware_mel(np.clip(aud[idx[k]][0] * g, -1, 1), "constant") for k in bgk[:60]])
    p = tfl.predict(X).argmax(1); print(f"   x{g}: bg acc {np.mean(p==0):.2f}  ->chainsaw {np.mean(p==1):.2f}  ->gunshot {np.mean(p==2):.2f}")

print("\n[D] Level statistics: training audio vs available recordings")
rms = np.array([np.sqrt((aud[i][0] ** 2).mean()) for i in idx])
for c, n in enumerate(CLASS_NAMES): print(f"  FSC22 {n}: median RMS {np.median(rms[y==c]):.4f}  p10-p90 {np.percentile(rms[y==c],10):.4f}-{np.percentile(rms[y==c],90):.4f}")
for f in ("datasets/mixed_tests/microphone_test.wav", "datasets/mixed_tests/chainsaw_rain_0dB.wav"):
    w, sr = sf.read(f); w = w if w.ndim == 1 else w[:, 0]
    w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=20000)
    w = w[:100000] if len(w) >= 100000 else fix_length(w)
    X = reference_mel(w)[None]; print(f"  {f.split('/')[-1]}: RMS {np.sqrt((w**2).mean()):.4f}  Keras {v1.predict(X,verbose=0)[0].round(3)}  INT8 {tfl.predict(X)[0].round(3)}")
