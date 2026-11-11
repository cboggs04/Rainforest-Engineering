"""Phase 1/2: leakage audit + reproducible baseline for Original / V1 / V2 (Keras)."""
import sys, os, json, pickle
sys.path.insert(0, "src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import tensorflow as tf
from audio_preprocessing import to_3class, CLASS_NAMES

OUT = "experiments/audit/results"; os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv("datasets/fsc22/fsc22/FSC22.csv")
y_all = np.array([t - 1 for t in meta.target])
import pandas as _pd
from audio_preprocessing import pickle_folds
folds = pickle_folds(_pd.read_csv("datasets/fsc22/fsc22/FSC22.csv").target.values - 1)
aud_y = np.array([a[1] for a in pickle.load(open("datasets/fsc22/Pickle Files/audios_5_20", "rb"))])
assert (aud_y == y_all).all(), "pickle order != FSC22.csv order"
print("pickle order matches FSC22.csv: True")

# --- leakage: V1 clean train = pickle folds 2,3,4 ; noisy val = official-fold-5 sources
pf = {i + 1: set(f) for i, f in enumerate(folds)}
meta["pickle_fold"] = 0
for k, idxs in pf.items():
    meta.loc[list(idxs), "pickle_fold"] = k
ct = pd.crosstab(meta.fold, meta.pickle_fold)
print("\nOfficial FSC22 fold (rows) vs notebook pickle fold (cols):\n", ct)
v1 = pd.read_csv("datasets/fsc22_augmented/mel_features_v1/feature_metadata.csv")
sf = meta.set_index("filename")
val = v1[(v1.threat_source.map(sf.fold) == 5) & (v1.background_source.map(sf.fold) == 5)]
srcs = val.threat_source.unique()
pfold_of = sf.loc[srcs, "pickle_fold"]
print(f"\nNoisy-val: {len(val)} clips from {len(srcs)} unique threat sources")
print("pickle fold of those sources:", pfold_of.value_counts().sort_index().to_dict())
leak = pfold_of.isin([2, 3, 4]).sum()
print(f"=> {leak}/{len(srcs)} noisy-val threat sources were ALSO in V1's clean training set (pickle folds 2-4)")
# same-freesound-recording check (middle id)
meta["fsid"] = meta.filename.str.split("-").str[1]
print("distinct freesound ids:", meta.fsid.nunique(), "of", len(meta), "clips")

# --- clean val (pickle fold 5)
def clean_set(fold):
    idx = np.array(sorted(pf[fold])); feats = pickle.load(open(f"datasets/fsc22/Pickle Files/mel_features_5_20/mel_features_5_20_fold{fold}", "rb"))
    # feats are in folds[fold-1] order
    order = folds[fold - 1]; X, y, nm = [], [], []
    for j, i in enumerate(order):
        l = to_3class(y_all[i])
        if l < 0: continue
        X.append(feats[j][0]); y.append(l); nm.append(meta.filename[i])
    return np.asarray(X, np.float32), np.array(y), nm
Xc, yc, nmc = clean_set(5)
print("\nclean val (pickle fold 5):", Xc.shape, np.bincount(yc))
fsid_c = {meta.fsid[meta.filename == n].iloc[0] for n in nmc}
tr_ids = {meta.fsid[i] for f in (2, 3, 4) for i in folds[f - 1] if to_3class(y_all[i]) >= 0}
print(f"clean val clips whose freesound recording also appears in train folds 2-4: "
      f"{sum(meta.fsid[meta.filename == n].iloc[0] in tr_ids for n in nmc)}/{len(nmc)}")
# official-fold version (leak-free)
# noisy val
Xn = np.stack([np.load(f"datasets/fsc22_augmented/mel_features_v1/{f}") for f in val.feature_filename]).astype(np.float32)
yn = val.label.map({"chainsaw": 1, "gunshot": 2}).values

paths = {
 "Original(BN09 fold5)": "models/MobileNetV3_Small_Rainforest_3Class_BN09/MobileNetV3_Small_Rainforest_3Class_BN09_base_fold5.keras",
 "V1": "models/MobileNetV3_Small_Rainforest_Finetuned_V1/MobileNetV3_Finetuned_V1_best.keras",
 "V2": "models/MobileNetV3_Small_Rainforest_Finetuned_V2/MobileNetV3_Finetuned_V2_best.keras"}
summary = {}
for name, p in paths.items():
    m = tf.keras.models.load_model(p, compile=False)
    for sname, X, y in (("clean_val", Xc, yc), ("noisy_val", Xn, yn)):
        pr = m.predict(X, verbose=0, batch_size=32); pd_ = pr.argmax(1)
        cm = confusion_matrix(y, pd_, labels=[0, 1, 2])
        rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
        print(f"\n=== {name} / {sname}  n={len(y)}  acc={np.mean(pd_==y):.3f} balacc={balanced_accuracy_score(y,pd_):.3f}")
        print(cm); print("recall bg/cs/gs:", np.round(rec, 3))
        summary[f"{name}|{sname}"] = dict(n=len(y), acc=float(np.mean(pd_ == y)), bal_acc=float(balanced_accuracy_score(y, pd_)), recall=rec.tolist(), cm=cm.tolist())
        np.save(f"{OUT}/probs_{name.split('(')[0]}_{sname}.npy", pr)
json.dump(summary, open(f"{OUT}/baseline_keras.json", "w"), indent=1)
