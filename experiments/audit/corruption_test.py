"""Can ESP32-style capture faults turn chainsaws into 'gunshots'?"""
import sys, os, json, pickle
sys.path.insert(0, "src"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
from audio_preprocessing import *
from evaluation import TFLiteModel
import pandas as _pd
from audio_preprocessing import pickle_folds
folds = pickle_folds(_pd.read_csv("datasets/fsc22/fsc22/FSC22.csv").target.values - 1)
meta = pd.read_csv("datasets/fsc22/fsc22/FSC22.csv"); y_all = np.array([t - 1 for t in meta.target])
aud = pickle.load(open("datasets/fsc22/Pickle Files/audios_5_20", "rb"))
tfl = TFLiteModel("models/ESP32_Deployment_V1/MobileNetV3_Rainforest_V1_INT8.tflite")
# ALL 75 chainsaw clips (train+val: this is a robustness probe, not an accuracy claim)
cs = [i for i in range(len(aud)) if y_all[i] == 10]
rng = np.random.default_rng(0)
def run(name, fn):
    X = np.stack([firmware_mel(np.clip(fn(aud[i][0].copy()), -1, 1), "constant") for i in cs])
    p = tfl.predict(X); a = p.argmax(1)
    print(f"{name:42s} ->bg {np.mean(a==0):.2f}  ->chainsaw {np.mean(a==1):.2f}  ->GUNSHOT {np.mean(a==2):.2f}  mean P(gun) {p[:,2].mean():.2f}")
def rand_zero(frac):
    def f(y): y[rng.random(len(y)) < frac] = 0; return y
    return f
def block_zero(frac, blk):
    def f(y):
        nb = len(y)//blk; sel = rng.random(nb) < frac
        for b in np.where(sel)[0]: y[b*blk:(b+1)*blk] = 0
        return y
    return f
def dc(v):  return lambda y: y + v
print(f"n={len(cs)} chainsaw clips (clean, INT8, firmware-equivalent features)")
run("unmodified", lambda y: y)
for fr in (0.05, 0.28): run(f"random single-sample zeros {fr:.0%}", rand_zero(fr))
for blk in (16, 128, 512): run(f"28% block dropouts (block={blk})", block_zero(0.28, blk))
run("DC offset +0.02", dc(0.02)); run("DC offset +0.1", dc(0.1))
run("1-bit-shift style: x2", lambda y: y*2); run("x0.5", lambda y: y*.5)
run("hard clip at 0.3 (x gain 4)", lambda y: np.clip(y*4, -.3, .3))
# 18-bit quantisation of 24-bit-left-justified mic: only a resolution check
run("quantise to 18 bit", lambda y: np.round(y*2**17)/2**17)
