import numpy as np, pandas as pd

def _rate(df, pred, cond, target):
    d = df[df.cond.isin(cond)]
    return d, (pred[d.index] == target)

def score(meta, prob, n_boot=500, seed=0):
    """Per-condition metrics. Recall conditions: fraction predicted correctly; false alarm
    conditions (clean_bg/hard_neg): fraction predicted as chainsaw/gunshot. 95% CI by
    bootstrap over SOURCE clips (mixes of one clip are not independent)."""
    pred = prob.argmax(1); rng = np.random.default_rng(seed); out = {}
    def ci(d, hit):
        g = pd.Series(hit.values, index=d.src.values).groupby(level=0).mean()
        v = g.values; b = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
        return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
    specs = {
     "clean_cs recall": (["clean_cs"], 1), "clean_gs recall": (["clean_gs"], 2),
     "gain_cs recall": (["gain_cs"], 1), "gain_gs recall": (["gain_gs"], 2),
     "bg false-alarm": (["clean_bg"], None), "hardneg false-alarm": (["hard_neg"], None)}
    for snr in (-5, 0, 5, 10, 15):
        specs[f"mix_cs {snr:+d}dB recall"] = (["mix_cs"], 1, snr)
        specs[f"mix_gs {snr:+d}dB recall"] = (["mix_gs"], 2, snr)
    for k, s in specs.items():
        d = meta[meta.cond.isin(s[0])]
        if len(s) == 3: d = d[d.snr == s[2]]
        p = pred[d.index]
        hit = (p != 0) if s[1] is None else (p == s[1])
        out[k] = ci(d, pd.Series(hit, index=d.index))
    cs2gs = ((pred[meta[meta.label == 1].index] == 2)).mean(); gs2cs = ((pred[meta[meta.label == 2].index] == 1)).mean()
    out["_cs->gs (all cs rows)"] = (cs2gs, np.nan, np.nan); out["_gs->cs (all gs rows)"] = (gs2cs, np.nan, np.nan)
    return out


def score_simple(meta, prob):
    """Per-condition accuracy: targets must get their class, label-0 rows must NOT alert."""
    pred = prob.argmax(1); out = {}
    for c, d in meta.groupby("cond"):
        p = pred[d.index]; l = d.label.values
        out[c] = float(np.mean(np.where(l == 0, p == 0, p == l)))
        if (l == 1).any(): out[c + " cs->gs"] = float(np.mean(p[l == 1] == 2))
    return out
