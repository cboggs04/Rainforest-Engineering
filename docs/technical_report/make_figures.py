"""Figures for the technical report. Run with the project venv:
    python docs/technical_report/make_figures.py
All numbers are read from saved result files (no hand-typed metrics)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

dep = json.load(open(OUT / "deployed_V3f_metrics.json"))
v1 = json.load(open("experiments/v3/results/ref_eval_ext.json"))["V1"]
ext = {k: v[0] for k, v in dep["eval_ext"].items()}
v1e = {k: v[0] for k, v in v1.items()}


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---- Fig 1: system architecture ----
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
def box(x, y, w, h, text, fc="#f0efec", ec="#b9b8b2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color=INK)
def arrow(x1, y1, x2, y2, label=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    if label: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.4, label, ha="center", fontsize=6.3, color=INK2)
ax.text(1, 43.5, "Detector node (XIAO ESP32S3, core 1: audio + model)", fontsize=7.5, color=INK, weight="bold")
box(1, 31, 12, 8, "SPH0645\nI2S mic\n20 kHz")
box(17, 31, 15, 8, "Health check\n(dropouts)\n+ DC removal")
box(36, 31, 15, 8, "3-channel\nlog-Mel\n128x196x3")
box(55, 31, 15, 8, "MobileNetV3-S\nINT8 (V3-f)\n~1 s")
box(74, 31, 12, 8, "Threshold\n+ cooldown")
for a, b in ((13.8, 17), (32.8, 36), (51.8, 55), (70.8, 74)): arrow(a, 35, b, 35)
ax.text(1, 25.5, "LoRa mesh (core 0 task on every node)", fontsize=7.5, color=INK, weight="bold")
box(74, 14, 12, 8, "LoRa mesh\nSX1262\n915 MHz SF9", fc="#e3edf9", ec=BLUE)
arrow(80, 30.2, 80, 22.8)
box(48, 14, 17, 8, "Other nodes\n(relay, hop count)", fc="#e3edf9", ec=BLUE)
arrow(73.2, 18, 65.8, 18, "LoRa")
box(22, 14, 17, 8, "Gateway node\n(gw 1)", fc="#e3edf9", ec=BLUE)
arrow(47.2, 18, 39.8, 18, "LoRa")
box(0, 1, 20, 8, "Mac\ngateway_bridge.py\n(USB @JSON log)")
box(24, 1, 15, 8, "Phone hotspot\nWiFi 2.4 GHz")
box(45, 1, 15, 8, "ntfy.sh\n(HTTPS POST)")
box(66, 1, 15, 8, "ntfy app\nphone alert", fc="#fbe7dd", ec=ORANGE)
arrow(24, 13.2, 13, 9.8, "USB"); arrow(31, 13.2, 31, 9.8, "WiFi"); arrow(39.8, 5, 45, 5); arrow(60.8, 5, 66, 5)
fig.savefig(OUT / "fig_architecture.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: V1 vs V3-f on unseen external recordings ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
panels = [("False alarms (lower is better)", [("Background", "bg false-alarm"), ("Hard negatives", "hardneg false-alarm")]),
          ("Recall in rainforest noise (higher is better)", [("Gunshot +5 dB", "mix_gs +5dB recall"), ("Chainsaw -5 dB", "mix_cs -5dB recall")])]
for ax, (title, items) in zip(axes, panels):
    xs = range(len(items)); w = 0.36
    for off, (lab, src, col) in zip((-w / 2 - 0.01, w / 2 + 0.01), (("V1 (previous)", v1e, ORANGE), ("V3-f INT8 (deployed)", ext, BLUE))):
        vals = [src[k] for _, k in items]
        bars = ax.bar([x + off for x in xs], vals, w, color=col, label=lab)
        for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels([n for n, _ in items]); ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); style(ax)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_v1_vs_v3f.png", dpi=220); plt.close(fig)

# ---- Fig 3: recall vs SNR ----
snrs = [-5, 0, 5, 10, 15]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, (cls, name) in zip(axes, (("gs", "Gunshot"), ("cs", "Chainsaw"))):
    for src, lab, col in ((v1e, "V1", ORANGE), (ext, "V3-f INT8", BLUE)):
        ys = [src[f"mix_{cls} {s:+d}dB recall"] for s in snrs]
        ax.plot(snrs, ys, color=col, lw=2, marker="o", ms=4.5, label=lab)
        ax.text(15.6, ys[-1], lab, color=INK2, fontsize=7, va="center")
    ax.set_title(f"{name} recall vs event SNR (external test set)", fontsize=8.5, color=INK, loc="left")
    ax.set_xticks(snrs); ax.set_xlabel("event-window SNR (dB)"); ax.set_xlim(-6, 19.5); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); style(ax)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_snr.png", dpi=220); plt.close(fig)

# ---- Fig 4: float vs INT8 agreement by checkpoint ----
rows = [("B0", 0.948, "scratch"), ("B1", 0.978, "scratch"), ("B2", 0.882, "pre"), ("V3-c", 0.863, "pre"), ("V3-d", 0.815, "pre"),
        ("V3-e", json.load(open("experiments/v3/results/E_s0.json"))["scores"]["eval_ext"]["agreement"], "scratch"),
        ("V3-f", json.load(open("models/ESP32_Deployment_V3/conversion_report_V3f.json"))["validation"]["eval_ext"]["class_agreement"], "scratch")]
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for i, (n, v, kind) in enumerate(rows):
    col = BLUE if kind == "scratch" else ORANGE
    ax.bar(i, v, 0.6, color=col, label={"scratch": "trained from scratch", "pre": "ImageNet-pretrained"}[kind] if n in ("B0", "B2") else None)
    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=7.5, color=INK)
ax.axhline(0.95, color=INK2, lw=0.8, ls=(0, (4, 3))); ax.text(6.42, 0.95, "95%\ntarget", fontsize=7, color=INK2, va="center", ha="left"); ax.set_xlim(-0.6, 7.1)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_ylabel("float/INT8 class agreement")
ax.set_title("INT8 conversion fidelity by checkpoint (external test set)", fontsize=8.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.16)); style(ax)
fig.tight_layout(); fig.savefig(OUT / "fig_int8_agreement.png", dpi=220); plt.close(fig)
print("figures written to", OUT)
