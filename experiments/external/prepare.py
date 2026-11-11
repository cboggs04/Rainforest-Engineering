"""Extract external clips -> 20 kHz mono, centre-padded/cropped to 5 s, same as original pipeline.
Outputs datasets/external/prepared/{audio.npy, meta.csv}.  role: cs / gs / hn (must not alert) / bg."""
import sys, io, glob, zipfile
sys.path.insert(0, "src")
import numpy as np, pandas as pd, soundfile as sf, librosa, pyarrow.parquet as pq
from audio_preprocessing import fix_length
rng = np.random.default_rng(7)
audio, rows = [], []
def add(y, sr, **kw):
    y = y.mean(1) if y.ndim > 1 else y
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=20000)
    audio.append(fix_length(y).astype(np.float32)); rows.append(kw)

# ---- ESC-50
z = zipfile.ZipFile("datasets/external/esc50_dl/esc50.zip"); P = "ESC-50-master/"
m = pd.read_csv(io.BytesIO(z.read(P + "meta/esc50.csv")))
ESC_HN = ["engine", "hand_saw", "fireworks", "vacuum_cleaner", "washing_machine", "helicopter", "airplane", "train", "clapping",
          "door_wood_knock", "door_wood_creaks", "footsteps", "laughing", "glass_breaking", "coughing", "siren", "car_horn", "dog",
          "cat", "rooster", "hen", "crow", "cow", "pig", "sheep", "keyboard_typing", "mouse_click", "clock_tick", "can_opening", "crying_baby", "sneezing", "snoring", "breathing"]
ESC_BG = ["rain", "wind", "thunderstorm", "crickets", "insects", "frog", "chirping_birds", "sea_waves", "water_drops", "pouring_water", "crackling_fire"]
for _, r in m.iterrows():
    role = "cs" if r.category == "chainsaw" else "hn" if r.category in ESC_HN else "bg" if r.category in ESC_BG else None
    if role is None: continue
    y, sr = sf.read(io.BytesIO(z.read(P + "audio/" + r.filename)))
    add(y, sr, src="esc50/" + r.filename, role=role, cls=r.category, fsid=r.src_file, fold=r.fold, ds="esc50")

# ---- UrbanSound8K
US_HN = {"drilling": 120, "jackhammer": 120, "engine_idling": 120, "dog_bark": 100, "car_horn": 100, "siren": 100, "street_music": 100, "children_playing": 100}
cnt = {}
for f in sorted(glob.glob("datasets/external/us8k_dl/*.parquet")):
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g, columns=["audio", "slice_file_name", "fsID", "fold", "class"]).to_pylist():
            c = r["class"]
            if c == "gun_shot": role = "gs"
            elif c in US_HN:
                cnt[c] = cnt.get(c, 0) + 1
                if cnt[c] > US_HN[c]: continue          # first N in file order (already shuffled across folds)
                role = "hn"
            else: continue
            y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
            add(y, sr, src="us8k/" + r["slice_file_name"], role=role, cls=c, fsid=r["fsID"], fold=r["fold"], ds="us8k")
A = np.stack(audio); df = pd.DataFrame(rows)
np.save("datasets/external/prepared/audio.npy", A); df.to_csv("datasets/external/prepared/meta.csv", index=False)
print(A.shape); print(df.groupby(["ds", "role"]).size())
print("gs sources (fsID):", df[df.role == "gs"].fsid.nunique(), " cs sources:", df[df.role == "cs"].fsid.nunique())
