"""Leak-free dataset construction on OFFICIAL FSC22 folds. Fold roles for V3-a:
train = folds 1-3, val = fold 4, test = fold 5 (datasets/eval_v1, never used here)."""
import pickle
import numpy as np, pandas as pd
from multiprocessing.pool import ThreadPool
from audio_preprocessing import reference_mel, to_3class
from augmentation import mix_event

_aud = None
def _load():
    global _aud
    if _aud is None:
        _aud = pickle.load(open("datasets/fsc22/Pickle Files/audios_5_20", "rb"))
    return _aud

def clips_by_role(folds):
    m = pd.read_csv("datasets/fsc22/fsc22/FSC22.csv"); m = m[m.fold.isin(folds)]
    lab = {i: to_3class(m.target[i] - 1) for i in m.index}
    g = lambda v: [i for i in m.index if lab[i] == v]
    return dict(bg=g(0), cs=g(1), gs=g(2), hn=g(-1)), m

def features(audios, procs=8):
    with ThreadPool(procs) as p:
        return np.stack(p.map(reference_mel, audios, chunksize=16)).astype(np.float32)

def gain(x, g): return np.clip(x * g, -1, 1).astype(np.float32)

def build(folds, variant, seed, n_mix=12, val=False):
    """Returns X, y (0=non-alert,1=chainsaw,2=gunshot), info DataFrame.
    variant: 'clean' (bg only as class 0), 'hn' (+hard negatives as class 0), 'mix' (hn + mixes + gain copies).
    val=True builds the fixed condition set used for model selection (always includes hn + mixes)."""
    rng = np.random.default_rng(seed); A = _load(); roles, m = clips_by_role(folds)
    a = lambda i: A[i][0].astype(np.float32)
    aud, y, info = [], [], []
    def add(x, label, kind): aud.append(x); y.append(label); info.append(kind)
    use_hn = val or variant in ("hn", "mix")
    for i in roles["bg"]: add(a(i), 0, "bg")
    if use_hn:
        for i in roles["hn"]: add(a(i), 0, "hn")
    for i in roles["cs"]: add(a(i), 1, "clean")
    for i in roles["gs"]: add(a(i), 2, "clean")
    if val:
        for snr in (-5, 0, 5, 10, 15):
            for i in roles["cs"] + roles["gs"]:
                for _ in range(2):
                    x, ach, _, _ = mix_event(a(i), a(int(rng.choice(roles["bg"]))), snr, rng=rng)
                    add(x, 1 if i in roles["cs"] else 2, f"mix{snr}")
    elif variant == "mix":
        for grp in ("bg", "hn", "cs", "gs"):                # random-gain copies of every clean clip
            lbl = {"bg": 0, "hn": 0, "cs": 1, "gs": 2}[grp]
            for i in roles[grp]:
                add(gain(a(i), 10 ** rng.uniform(np.log10(0.3), np.log10(3))), lbl, "gain")
        for i in roles["cs"] + roles["gs"]:
            lbl = 1 if i in roles["cs"] else 2
            for _ in range(n_mix):
                snr = rng.uniform(-5, 20)
                x, ach, _, _ = mix_event(a(i), a(int(rng.choice(roles["bg"]))), snr, rng=rng)
                assert abs(ach - snr) < 0.01
                add(gain(x, 10 ** rng.uniform(np.log10(0.3), np.log10(3))), lbl, "mix")
    X = features(aud)
    return X, np.array(y), pd.DataFrame({"kind": info})
