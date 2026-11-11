"""V3-b data: FSC22 (official folds) + external (source-grouped splits)."""
import pickle, gc
import numpy as np, pandas as pd
from audio_preprocessing import to_3class
from augmentation import mix_event, simulate_playback, rms
from dataset import features, gain

EXT = "datasets/external/prepared/"

def _fsc(folds):
    m = pd.read_csv("datasets/fsc22/fsc22/FSC22.csv"); m = m[m.fold.isin(folds)]
    A = pickle.load(open("datasets/fsc22/Pickle Files/audios_5_20", "rb"))
    out = {"bg": [], "hn": [], "cs": [], "gs": []}
    for i in m.index:
        l = to_3class(m.target[i] - 1); k = {0: "bg", -1: "hn", 1: "cs", 2: "gs"}[l]
        out[k].append((A[i][0].astype(np.float32), m.filename[i]))
    del A; gc.collect(); return out

def _ext(split):
    meta = pd.read_csv(EXT + "meta.csv"); A = np.load(EXT + "audio.npy", mmap_mode="r")
    out = {"bg": [], "hn": [], "cs": [], "gs": []}
    for k in meta.index[meta.split == split]:
        out[meta.role[k]].append((np.array(A[k]), meta.src[k]))
    return out

def mixable(x, max_silent=0.2):
    """Usable as a noise bed: not dominated by digital silence (zero padding)."""
    return (np.abs(x) < 1e-6).mean() < max_silent

def mix_retry(x, bgs, snr, rng, tries=25):
    """mix_event with a fresh background until the target SNR is achieved (a bed with silence
    inside the event window cannot reach it). First attempt uses the same RNG draws as before."""
    for _ in range(tries):
        m, ach, s, e = mix_event(x, bgs[rng.integers(len(bgs))], snr, rng=rng)
        if abs(ach - snr) < 0.01:
            return m, ach
    raise RuntimeError(f"could not reach SNR {snr}")

GAIN_LO = 0.3
def rgain(rng): return 10 ** rng.uniform(np.log10(GAIN_LO), np.log10(3))

def to_rms(x, target):
    return np.clip(x * target / (rms(x) + 1e-12), -1, 1).astype(np.float32)

def build_train(seed, ext=True, negmix=True, n_mix_fsc_gs=8, hn_cap=700, v3c=False):
    global GAIN_LO
    GAIN_LO = 0.03 if v3c else 0.3
    rng = np.random.default_rng(seed); F = _fsc([1, 2, 3]); E = _ext("train") if ext else {k: [] for k in F}
    P = {k: F[k] + E[k] for k in F}
    hn = P["hn"]; hn = [hn[i] for i in rng.permutation(len(hn))[:hn_cap]] if len(hn) > hn_cap else hn
    aud, y, kind = [], [], []
    def add(x, l, k): aud.append(x); y.append(l); kind.append(k)
    for x, _ in P["bg"]: add(x, 0, "bg")
    for x, _ in hn: add(x, 0, "hn")
    for x, _ in P["cs"]: add(x, 1, "clean")
    for x, _ in P["gs"]: add(x, 2, "clean")
    for grp, l in (("cs", 1), ("gs", 2)):                                        # gain copies of targets
        for x, _ in P[grp]: add(gain(x, rgain(rng)), l, "gain")
    for grp in (hn, P["bg"]):                                                   # 30% gain copies of negatives
        for j in rng.permutation(len(grp))[: int(.3 * len(grp))]: add(gain(grp[j][0], rgain(rng)), 0, "gain")
    bgs = [x for x, _ in P["bg"] if mixable(x)]
    # balanced mixes: ~1000 per target class regardless of clip counts
    for grp, l in (("cs", 1), ("gs", 2)):
        n = len(P[grp]); per = max(2, int(round(1000 / n)))
        for x, _ in P[grp]:
            for _ in range(per):
                snr = rng.uniform(-5, 20); m, ach = mix_retry(x, bgs, snr, rng)
                add(gain(m, rgain(rng)), l, "mix")
    if negmix:                                                                  # non-target events in rainforest noise -> label 0
        for j in rng.integers(len(hn), size=700):
            snr = rng.uniform(-5, 20); m, _ = mix_retry(hn[j][0], bgs, snr, rng)
            add(gain(m, rgain(rng)), 0, "negmix")
    if v3c:
        tgt = [(x, 1) for x, _ in P["cs"]] + [(x, 2) for x, _ in P["gs"]]
        for k in range(450):                                    # loud mixtures (e.g. heavy rain close to the mic)
            x, l = tgt[rng.integers(len(tgt))] if k < 300 else (None, 0)
            bed = bgs[rng.integers(len(bgs))]
            if x is None: m = bed
            else: m, _ = mix_retry(x, bgs, rng.uniform(-5, 5), rng)
            add(to_rms(m, 10 ** rng.uniform(np.log10(0.05), np.log10(0.2))), l, "loudbed")
        for j in range(len(aud)):                               # speaker->room->mic chain on 30% of ALL examples
            if rng.random() < 0.3:
                aud[j] = simulate_playback(aud[j], rng); kind[j] += "+chain"
    del F, E, P; gc.collect()
    X = features(aud); del aud; gc.collect()
    return X, np.array(y), pd.DataFrame({"kind": kind})

def condition_set(P, seed, snrs=(-5, 0, 5, 10, 15), per=2, gains=True):
    """Eval/val set with the same condition names as eval_v1 (clean_*, gain_*, mix_*, hard_neg, clean_bg)."""
    rng = np.random.default_rng(seed); aud, rows = [], []
    bgs = [x for x, _ in P["bg"] if mixable(x)]
    def add(x, l, cond, src, snr=np.nan, g=1.0): aud.append(x); rows.append(dict(label=l, cond=cond, src=src, snr=snr, gain=g))
    for x, s in P["bg"]: add(x, 0, "clean_bg", s)
    for x, s in P["hn"]: add(x, 0, "hard_neg", s)
    for grp, l, n in (("cs", 1, "cs"), ("gs", 2, "gs")):
        for x, s in P[grp]:
            add(x, l, f"clean_{n}", s)
            if gains:
                for g in (0.3, 3.0): add(gain(x, g), l, f"gain_{n}", s, g=g)
            for snr in snrs:
                for _ in range(per):
                    m, ach = mix_retry(x, bgs, snr, rng)
                    add(m, l, f"mix_{n}", s, snr)
    X = features(aud); return X, pd.DataFrame(rows)
