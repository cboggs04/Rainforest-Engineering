"""Builds docs/Rainforest_Technical_Report.pdf.
Run: ~/.venvs/report-tools/bin/python docs/technical_report/build_report.py
(after docs/technical_report/make_figures.py). Metrics are read from result files."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether, ListFlowable, ListItem, NextPageTemplate)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = ROOT / "docs" / "Rainforest_Technical_Report.pdf"

FD = ROOT / ".venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf"
pdfmetrics.registerFont(TTFont("DV", str(FD / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FD / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(FD / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FD / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK, INK2, RULE, FILL = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#cfcec9"), colors.HexColor("#f0efec")
ACCENT = colors.HexColor("#1c5cab")
S = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=24, leading=30, textColor=INK, spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=12.5, leading=17, textColor=INK2, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=15, leading=19, textColor=ACCENT, spaceBefore=16, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=11.5, leading=15, textColor=INK, spaceBefore=9, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=13.4, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8, leading=10.5, textColor=INK2),
    "cap": ParagraphStyle("cap", fontName="DVI", fontSize=8, leading=10.5, textColor=INK2, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=8, leading=10.2, textColor=INK),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.6, leading=10, textColor=INK, backColor=FILL,
                           borderPadding=(4, 5, 4, 5), spaceBefore=3, spaceAfter=8, leftIndent=4),
    "callout": ParagraphStyle("callout", fontName="DV", fontSize=9.2, leading=13, textColor=INK, backColor=colors.HexColor("#e9f1fb"),
                              borderPadding=(6, 7, 6, 7), spaceBefore=4, spaceAfter=10, leftIndent=6, rightIndent=6),
}

# ---------------- metrics (read from files) ----------------
dep = json.load(open(HERE / "deployed_V3f_metrics.json"))
conv = json.load(open(ROOT / "models/ESP32_Deployment_V3/conversion_report_V3f.json"))
base = json.load(open(ROOT / "experiments/audit/results/baseline_keras.json"))
refx = json.load(open(ROOT / "experiments/v3/results/ref_eval_ext.json"))
E = {k: v[0] for k, v in dep["eval_ext"].items()}
F5 = {k: v[0] for k, v in dep["eval_v1"].items()}
ST = dep["eval_stress"]; RP5, RP8 = dep["replay_test@0.5"], dep["replay_test@0.8"]
V1X = {k: v[0] for k, v in refx["V1"].items()}
pct = lambda x, d=0: f"{100 * x:.{d}f}%"


def P(t, st="body"): return Paragraph(t, S[st])
def bullets(items, st="body"):
    return ListFlowable([ListItem(P(i, st), leftIndent=12, value="•") for i in items], bulletType="bullet", start="•",
                        leftIndent=12, bulletFontName="DV", bulletFontSize=8)
def code(t): return Paragraph(t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])
def fig(name, width, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(HERE / name)).getSize()
    return KeepTogether([Image(str(HERE / name), width=width, height=width * ih / iw), P(caption, "cap")])
def table(rows, widths, header=True, zebra=True, bold_first_col=False):
    data = [[c if not isinstance(c, str) else P(c, "cellb" if (header and r == 0) or (bold_first_col and j == 0) else "cell")
             for j, c in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2) if header else ("TOPPADDING", (0, 0), (0, 0), 3),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), FILL))
    if zebra:
        for r in range(2 if header else 1, len(rows), 2): st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8f8f6")))
    t.setStyle(TableStyle(st)); return t


W = 6.9 * inch
story = []

# ---------------- title page ----------------
story += [Spacer(1, 1.5 * inch),
          P("Rainforest Engineering", "subtitle"),
          P("Acoustic Threat Detection on the Edge", "title"),
          P("Chainsaw and gunshot detection with an INT8 MobileNetV3 on the XIAO ESP32-S3, a LoRa mesh, and WiFi phone alerts", "subtitle"),
          Spacer(1, 0.3 * inch),
          P("Technical report · November 10, 2026", "small"),
          P("Project lead: Charlie Boggs · Repository: github.com/cboggs04/Rainforest-Engineering (branch v3-model-and-lora-mesh)", "small"),
          Spacer(1, 0.5 * inch),
          P(f"<b>In one paragraph.</b> A low-power acoustic node listens to the forest in 5-second windows, converts each window to a "
            f"three-resolution log-Mel spectrogram, and classifies it as BACKGROUND, CHAINSAW or GUNSHOT with a 1.19 MiB INT8 "
            f"MobileNetV3-Small running on the ESP32-S3. Detections travel over a self-relaying LoRa mesh to a gateway, which "
            f"pushes a notification to a phone over WiFi. This report documents how the model was audited, rebuilt and validated; "
            f"how the firmware captures and preprocesses audio; and how the two-board network and notifications work. "
            f"Compared with the previous V1 model on recordings it never saw during training, the deployed V3-f model cuts "
            f"hard-negative false alarms from {pct(V1X['hardneg false-alarm'])} to {pct(E['hardneg false-alarm'])} and raises gunshot "
            f"recall at +5 dB SNR from {pct(V1X['mix_gs +5dB recall'])} to {pct(E['mix_gs +5dB recall'])}, while running in INT8 on the "
            f"device with {pct(conv['validation']['eval_ext']['class_agreement'], 1)} agreement with its float version.", "callout"),
          NextPageTemplate("body"), PageBreak()]

# ---------------- contents ----------------
toc = ["1. Summary of results", "2. First system and initial models (V1, V2)", "3. System overview", "4. Data", "5. Audio features and preprocessing",
       "6. Evaluation methodology", "7. Model development", "8. INT8 quantization and deployment", "9. ESP32 firmware: audio pipeline",
       "10. Hardware debugging and live tests", "11. LoRa mesh network", "12. Gateway, WiFi and phone notifications",
       "13. Results comparison", "14. Limitations and risks", "15. Recommended next steps", "Appendix A. Reproducing the work",
       "Appendix B. Device command reference"]
story += [P("Contents", "h1"), bullets(toc), PageBreak()]

# ---------------- 1. summary ----------------
story += [P("1. Summary of results", "h1"),
          P("The first stage of the project produced a working prototype: a fine-tuned MobileNetV3-Small (\"V1\") deployed as INT8 on an ESP32 (Section 2), "
            "which classified a chainsaw played from a phone as a gunshot with 97% confidence. The second stage, the bulk of this report, had "
            "four parts:"),
          bullets([
              "<b>Audit.</b> All earlier V1/V2 numbers were reproduced exactly, but the evaluation behind them was too small "
              "(22 noisy clips) and partly leaked training material. Two firmware preprocessing faults and a capture fault were found.",
              "<b>Model rebuild.</b> Leak-free test sets were built (FSC22 official fold 5, held-out external recordings, a stress set and "
              "real phone-replay recordings). A sequence of controlled experiments led to V3-f: a scratch-trained MobileNetV3-Small, "
              "distilled from an ImageNet-pretrained teacher so that it keeps most of the teacher's accuracy and still quantizes cleanly to INT8.",
              "<b>Firmware.</b> The ESP32 pipeline now matches training preprocessing exactly (zero padding), rejects recordings with I2S "
              "dropouts, and removes the microphone's DC offset. The deployed model uses the same 8 operators and input shape as V1.",
              "<b>Network.</b> A custom LoRa flooding mesh (relay, de-duplication, TTL, ACK and retry) runs on the ESP32's second core. "
              "A gateway node forwards alerts to a Mac over USB and to a phone through WiFi push notifications (ntfy).",
          ]),
          P("Headline numbers for the deployed model (V3-f, INT8, 50% alert threshold) on test material it never trained on:", "body"),
          table([["Test set", "Gunshot alerted", "Chainsaw alerted", "False alarms", "Chainsaw sent as gunshot"],
                 ["External recordings (ESC-50 / UrbanSound8K held-out sources)", pct(dep['eval_ext@0.5']['gs']), pct(dep['eval_ext@0.5']['cs']), pct(dep['eval_ext@0.5']['neg_FA']), pct(dep['eval_ext@0.5']['cs_to_gs'], 1)],
                 ["FSC22 official fold 5", pct(dep['eval_v1@0.5']['gs']), pct(dep['eval_v1@0.5']['cs']), pct(dep['eval_v1@0.5']['neg_FA']), pct(dep['eval_v1@0.5']['cs_to_gs'], 1)],
                 [f"Real phone-to-microphone replays ({RP5['n_gs']} gunshots, {RP5['n_cs']} chainsaws, {RP5['n_neg']} others)", pct(RP5['gs']), pct(RP5['cs']), pct(RP5['neg_FA']), pct(RP5['cs_to_gs'], 1)]],
                [2.9 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 1.1 * inch]),
          P("Rows mix clean, gain-shifted and noisy clips (including -5 dB SNR), so they are deliberately harder than a quiet-room demo. "
            "Per-condition results are in Section 13.", "cap"),
          fig("fig_v1_vs_v3f.png", W, "Figure 1. V1 versus the deployed V3-f (INT8) on held-out external recordings that neither model trained on. "
              "False alarms count clips predicted as CHAINSAW or GUNSHOT; recall counts correctly classified noisy target clips."),
          Spacer(1, 6)]

# ---------------- 2. system overview ----------------
story += [P("2. First system and initial models (V1, V2)", "h1"),
          P("The first development stage produced a working end-to-end prototype and two fine-tuned models. Figures marked <b>re-measured</b> "
            "were later reproduced exactly with the reproducible evaluation scripts (Section 13); the others come from the original development "
            "experiments, several of which used very small sets."),
          P("2.1 Objective and design choices", "h2"),
          P("The goal is a low-cost, battery-powered acoustic monitor for rainforest conservation that detects chainsaws (possible illegal "
            "logging) and gunshots (possible poaching) while ignoring natural background sound. Classification happens on the device so that only a "
            "small LoRa message, not raw audio, has to be transmitted. The chosen chain is SPH0645 I2S microphone → XIAO ESP32-S3 → "
            "MobileNetV3-Small INT8 → Wio-SX1262 LoRa → remote ESP32 receiver. Microphone recording, Mel-spectrogram "
            "generation, INT8 inference and LoRa transmission were all demonstrated on the ESP32."),
          P("2.2 Base model and preprocessing", "h2"),
          bullets([
              "Starting point: the open-source Forest-Sound-Analysis-on-Edge project (Neural-Dreamers), which classifies FSC22 forest sounds. It was "
              "adapted to three classes (BACKGROUND, CHAINSAW, GUNSHOT) and to ESP32 deployment.",
              "Features: three log-Mel spectrograms (n_fft 2048 / 1024 / 512, hop 512, 128 bands) of 5-second, 20 kHz audio, stacked into a 128 × 196 × 3 input.",
              "The Python augmentation pipeline was checked against the original saved features: for a correctly aligned waveform the regenerated "
              "features matched exactly (mean and maximum absolute difference 0.0).",
              "MobileNetV3-Small trained from scratch with five-fold cross-validation (the \"BN09\" base models, batch-norm momentum 0.9).",
          ]),
          P("2.3 Fine-tuned models V1 and V2", "h2"),
          table([["Model", "Approach", "Reported results"],
                 ["Original", "3-class base model (fold-5 checkpoint)", "Clean accuracy 91.5%; noisy validation 63.6% (<b>re-measured</b>)"],
                 ["V1", "Fine-tuned with noisy augmentation: chainsaw and gunshot clips mixed into rainforest background at -5, 0, +5 and +10 dB using whole-recording RMS",
                  "Noisy validation 72.7%; noisy gunshot recall 63.6%; noisy chainsaw recall 81.8% (<b>re-measured</b>). The strongest candidate from this stage"],
                 ["V2", "Added event-aware gunshot augmentation (below)", "Noisy validation 54.5% (<b>re-measured</b>): worse than V1"]],
                [0.8 * inch, 3.2 * inch, 2.9 * inch]),
          P("<b>Event-aware gunshot augmentation (V2).</b> randomize_gunshot_timing() found the strongest waveform peak and cut a window from "
            "0.15 s before to 1.5 s after it (about 1.65 s), then placed it at a random position in the 5-second clip; mix_gunshot_event_aware() "
            "set the SNR on that event window (-5, 0, +5, +10 dB). V2 recognised few of its own training examples:"),
          KeepTogether([table([["Event SNR", "Detected as gunshot (of 45)", "Detection rate"],
                 ["-5 dB", "3", "6.7%"], ["0 dB", "6", "13.3%"], ["+5 dB", "14", "31.1%"], ["+10 dB", "14", "31.1%"],
                 ["All 180", "37 (139 predicted background, 4 chainsaw)", "20.6% (mean gunshot probability 23.6%)"]],
                [1.2 * inch, 3.2 * inch, 2.5 * inch]),
          P("This was a training-set diagnostic, which suggested the training procedure had not learned the new augmentation distribution.", "cap")]),
          P("2.4 Mixing and SNR experiments", "h2"),
          P("For one gunshot/background pair at a nominal +10 dB, the two mixing methods produced different event loudness: the actual event-window "
            "SNR was 16.28 dB with V1 mixing and 10.00 dB with V2 mixing (6.28 dB apart). The V1 model called the V1-mixed clip GUNSHOT (76.2%) "
            "and the V2-mixed clip CHAINSAW (46.5%, gunshot 40.0%). Increasing the SNR helped one recording but not five:"),
          table([["Event SNR", "One recording: gunshot probability", "Five recordings: detection rate", "Five recordings: mean gunshot probability"],
                 ["+10 dB", "40.0%", "60%", "58.6%"], ["+15 dB", "65.4%", "60%", "56.6%"],
                 ["+20 dB", "95.8%", "40%", "52.0%"], ["+25 dB", "98.5%", "40%", "44.1%"]],
                [1.0 * inch, 2.0 * inch, 1.9 * inch, 2.0 * inch]),
          P("The conclusion was that making gunshots louder is not a general fix and that results must be checked across many source recordings. "
            "That conclusion led directly to the measured-SNR mixing and larger test sets in Section 6.", "cap"),
          P("2.5 INT8 conversion and ESP32 deployment of V1", "h2"),
          bullets([
              "V1 was converted to full INT8 TensorFlow Lite: 1,248,792 bytes (1.19 MiB); input int8 [1,128,196,3] with scale about 0.5427 and zero "
              "point 56; output int8 [1,3] with scale 0.00390625 and zero point -128; operators ADD, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, "
              "HARD_SWISH, MEAN, MUL and SOFTMAX.",
              "On 52 development samples the INT8 and Keras models agreed on 50 (96.15%), with a mean probability difference of 0.033 and a maximum of 0.356.",
              "Firmware (Arduino, libraries ESP_I2S, arduinoFFT, TensorFlowLite_ESP32 and RadioLib): 20 kHz 32-bit mono I2S capture, on-device Mel "
              "spectrograms using librosa filter banks exported as sparse coefficients (mel_filterbanks.h), and a 4 MiB tensor arena in the 8 MB PSRAM. "
              "Measured times: capture 5.0 s, Mel preprocessing 1.18 s, inference about 1.0 s (997 ms on synthetic input).",
              "Microphone wiring: BCLK→D0, LRCL/WS→D1, DOUT→D2, SEL→GND, 3V and GND; the wiring was known to be occasionally unreliable.",
              "A LoRa link to a second ESP32 was working, with messages such as TEST|NODE_01|CHAINSAW|81.2|1; a separate LoRa device in the classroom "
              "caused unwanted packets.",
          ]),
          P("2.6 The live misclassification", "h2"),
          P("When a chainsaw recording was played from a phone near the microphone, the ESP32 reported BACKGROUND 2.73%, CHAINSAW 0.00% and "
            "GUNSHOT 97.27%. The capture had RMS 0.132 and peak 0.822, and 27,940 of its 100,000 samples were exactly zero. A USB-serial recorder "
            "was also started (record_esp32_audio.py) to capture the exact ESP32 audio for comparison with Python; its first "
            "recording was entirely zeros. Whether the error came from the model or from audio acquisition was unknown. Section 10 shows the "
            "most likely cause was capture dropouts."),
          Spacer(1, 6)]

story += [KeepTogether([P("3. System overview", "h1"),
          fig("fig_architecture.png", W, "Figure 2. End-to-end system. Each node records and classifies audio on core 1 while a separate "
              "task on core 0 runs the LoRa mesh, so the radio keeps receiving and relaying during audio capture and inference.")]),
          P("2.1 Hardware", "h2"),
          table([["Component", "Part", "Role / notes"],
                 ["Microcontroller", "Seeed Studio XIAO ESP32-S3 (8 MB flash, 8 MB OPI PSRAM)", "Dual core; core 1 runs audio + model, core 0 runs the mesh and WiFi"],
                 ["Microphone", "SPH0645LM4H I2S MEMS microphone", "BCLK→D0, LRCL/WS→D1, DOUT→D2, SEL→GND (left slot), 3V, GND; 18-bit data in 32-bit slots"],
                 ["Radio", "Wio-SX1262 (board-to-board connector)", "915 MHz, 125 kHz bandwidth, SF9, CR 4/7, sync word 0x12, +10 dBm"],
                 ["Gateway uplink", "ESP32 WiFi (2.4 GHz) + phone hotspot", "HTTPS POST to ntfy.sh; phone app shows the alert"],
                 ["Development host", "MacBook Air (M1), Python 3.11, TensorFlow 2.21", "Training on CPU; arduino-cli for builds and uploads"]],
                [1.25 * inch, 2.45 * inch, 3.2 * inch]),
          P("2.2 Software components", "h2"),
          table([["Component", "Location", "Purpose"],
                 ["Preprocessing, mixing, datasets, scoring", "src/", "Reusable Python modules; exact reproduction of training features"],
                 ["Experiments and results", "experiments/", "Audit, evaluation sets, V3 training and conversion scripts; full log in experiments/audit/REPORT.md"],
                 ["Mesh node firmware", "firmware/Rainforest_Mesh_Node/", "Detector + LoRa mesh + gateway (USB JSON and WiFi notifications)"],
                 ["Single-node detector", "firmware/Rainforest_Detection_ESP32/", "Detector without the mesh (with all audio fixes)"],
                 ["Diagnostics", "firmware/Rainforest_Mic_Slot_Test/", "Reads both I2S slots with a pull-up to diagnose microphone wiring"],
                 ["Mac tools", "live_test.py, gateway_bridge.py, record_replay_dataset.py", "Mac-microphone testing, USB gateway, real-replay data collection"]],
                [1.9 * inch, 2.3 * inch, 2.7 * inch]),
          Spacer(1, 6)]

# ---------------- 3. data ----------------
story += [P("4. Data", "h1"),
          P("3.1 FSC22", "h2"),
          P("FSC22 is a forest sound dataset of 1,950 five-second clips in 26 classes (75 clips each), distributed with an official "
            "5-fold assignment (15 clips per class per fold). The project maps FSC22 to three classes: <b>BACKGROUND</b> = rain, "
            "thunderstorm, water drops, wind, silence, insect, frog, bird chirping, wing flapping (9 classes); <b>CHAINSAW</b>; "
            "<b>GUNSHOT</b>. The remaining 15 classes (for example speaking, footsteps, axe, handsaw, generator, firework, tree falling) "
            "were originally discarded. In V3 they are used as <b>hard negatives</b>: sounds that must <i>not</i> raise an alert."),
          P("Audio was resampled to 20 kHz mono with librosa and centre-padded or cropped to exactly 100,000 samples. "
            "There are only 75 chainsaw and 75 gunshot clips in total, which is the main reason extra data was needed."),
          P("3.2 External datasets", "h2"),
          table([["Dataset", "Used for", "Clips used", "License"],
                 ["ESC-50", "Chainsaw (40 clips, 22 source recordings); hard negatives such as engine, hand saw, fireworks, vacuum; extra background", "40 + 1,320 + 440", "CC BY-NC 3.0"],
                 ["UrbanSound8K (Hugging Face mirror)", "Gunshot (374 slices, 117 source recordings); hard negatives such as drilling, jackhammer, engine idling, siren, dog bark", "374 + 860", "CC BY-NC 4.0"]],
                [1.5 * inch, 3.5 * inch, 0.95 * inch, 0.95 * inch]),
          P("Both licenses are non-commercial. Models trained on these data should not be used commercially without reviewing the licenses.", "cap"),
          P("<b>De-duplication against FSC22.</b> FSC22 and these datasets are all partly sourced from Freesound, so an identical recording "
            "could appear in an external training set and in the FSC22 test fold. Every external chainsaw and gunshot clip was "
            "compared with every FSC22 clip of the same class by normalised cross-correlation of the waveform. 31 gunshot slices and 2 "
            "chainsaw clips matched an FSC22 clip almost exactly (correlation above 0.8, up to 0.9999). Every clip from those source "
            "recordings was excluded (63 gunshot and 4 chainsaw clips)."),
          P("<b>Source-grouped splits.</b> External clips were split into train / validation / test by Freesound source ID, so slices of one "
            "recording never appear on both sides. Gunshots: 217 / 28 / 66 clips (59 / 15 / 25 sources). "
            "Chainsaws: 18 / 5 / 13 clips (11 / 3 / 6 sources)."),
          P("3.3 Real phone-replay recordings", "h2"),
          P("Because demonstrations use sounds played from a phone, the project collected real speaker-to-microphone recordings. "
            "record_replay_dataset.py builds a playlist of labelled clips framed by a frequency-sweep sync chirp; the phone plays it while the "
            "Mac records; the script finds both chirps, corrects the phone/Mac clock drift (measured 8 ppm) and cuts the recording back into "
            "labelled 5-second clips. Alignment was verified on a simulated recording (median error 0.35 ms). Training playlists use only "
            "training sources and test playlists only test sources. One training session (120 clips) and one test session (58 clips) were recorded "
            "with the phone at about 50 cm."),
          Spacer(1, 6)]

# ---------------- 4. features ----------------
story += [P("5. Audio features and preprocessing", "h1"),
          P("The model's input is a 128 × 196 × 3 tensor: three log-Mel spectrograms of the same 5-second window computed with different FFT "
            "sizes, so the network sees both fine frequency detail and fine time detail. The exact training pipeline was recovered from the "
            "original notebooks and reimplemented in src/audio_preprocessing.py, which reproduces the saved training features to 0.0000 dB."),
          table([["Parameter", "Value"],
                 ["Sample rate / window", "20,000 Hz, mono, 5 s = 100,000 samples (centre zero-pad or crop)"],
                 ["Amplitude normalisation", "None: librosa float scaling only, so absolute loudness matters"],
                 ["STFT", "Periodic Hann window; hop 512; n_fft 2048 / 1024 / 512 for channels 0 / 1 / 2; centred frames with <b>zero padding</b> (librosa 0.11 default); 196 frames"],
                 ["Mel filter bank", "128 bands, librosa Slaney normalisation (sparse coefficients exported to mel_filterbanks.h)"],
                 ["Power and dB", "Magnitude squared; 10·log10(max(x, 1e-10)); floor at (per-channel maximum - 80 dB)"],
                 ["Model input scaling", "Inside the model: x·(2/140) + (200/140 - 1), mapping roughly -100…+40 dB to -1…+1"],
                 ["INT8 input quantisation (V3-f)", f"scale {conv['input']['scale']:.6f}, zero point {conv['input']['zero_point']} (read by the firmware at run time)"]],
                [1.9 * inch, 5.0 * inch]),
          P("4.1 Preprocessing mismatches found", "h2"),
          bullets([
              "<b>Edge padding.</b> The firmware reflected the signal at the window edges while training used zero padding. The first and last "
              "frames differed by up to 2.75 dB on average. After switching the firmware to zero padding, a NumPy emulation of the firmware "
              "matches the training features exactly.",
              "<b>DC offset.</b> The SPH0645 output carries a constant offset of about -0.105 to -0.125 of full scale on these boards, "
              "larger than any offset in the training audio. The firmware now subtracts each recording's mean before the spectrogram; in a quiet "
              "room the true sound level is then 0.006-0.009 RMS, close to quiet FSC22 background clips.",
              "<b>Loudness dependence.</b> Because features are absolute dB levels, the model is sensitive to gain. For V1, chainsaw recall fell "
              "from 73% at original level to 40% at 10× gain, and background-to-gunshot false alarms rose from 5% to 22% (45% at 30×). "
              "Later models were trained with random gain (0.03× to 3×).",
          ]),
          Spacer(1, 6)]

# ---------------- 5. evaluation ----------------
story += [P("6. Evaluation methodology", "h1"),
          P("5.1 Problems with the original evaluation", "h2"),
          bullets([
              "The \"noisy validation\" set had <b>22 clips</b> (one clip = 4.5 percentage points) from 17 sources, with no background clips, so "
              "false alarms were never measured on it. V1's advantage over the original model rested on two clips.",
              "Clean folds were a random stratified shuffle of clips, while noisy mixes used FSC22's official folds. <b>6 of the 17</b> source "
              "clips behind the noisy validation set were also in V1's clean training data.",
              "The original and V1 models had seen about 80% of the FSC22 fold-5 clips during training. Their FSC22 scores are therefore inflated "
              "and marked as such in this report.",
          ]),
          P("5.2 New test sets", "h2"),
          table([["Test set", "Built from", "Contents"],
                 ["FSC22 fold 5 (eval_v1)", "FSC22 official fold 5 only", "135 background, 225 hard negatives, 15 chainsaw and 15 gunshot clips clean and at 0.3× / 3× gain, 600 mixes (5 SNRs × 4 random backgrounds and positions per clip): 1,050 clips"],
                 ["External (eval_ext)", "Held-out ESC-50 / UrbanSound8K sources", "109 background, 551 hard negatives, 13 chainsaw (6 sources), 66 gunshot (25 sources), gain copies, 790 mixes: 1,687 clips"],
                 ["Stress (eval_stress)", "Test sources of both", "Simulated speaker/room/mic replay, very quiet events (0.03×, 0.1×), loud rain beds: 1,020 clips"],
                 ["Real replay", "One recorded session", "58 real phone-to-Mac-mic clips (20 gunshot, 12 chainsaw, 26 non-target)"]],
                [1.4 * inch, 1.7 * inch, 3.8 * inch]),
          P("5.3 Mixing with a measured SNR", "h2"),
          P("Target sounds are mixed into rainforest background at a controlled event signal-to-noise ratio. The event's active window is "
            "the region within 30 dB of its loudest 25 ms frame; the background is scaled so that the RMS ratio inside that window equals the "
            "target SNR, and the achieved SNR is recomputed and asserted for every clip (within 0.01 dB). This fixes an inconsistency in the "
            "earlier experiments, where two mixing methods with the same nominal +10 dB produced event SNRs 6.3 dB apart. Background beds that "
            "are mostly digital silence (zero-padded clips) are excluded, and if a bed has silence inside the event window a different bed is drawn."),
          P("5.4 Metrics", "h2"),
          P("Recall is reported per condition (clean, gain-shifted, each SNR). False alarms are the share of background or hard-negative clips "
            "predicted as CHAINSAW or GUNSHOT. Chainsaw-as-gunshot is tracked separately because it was the original failure. Mixes of one source "
            "clip are not independent, so 95% confidence intervals are bootstrapped over source clips. Models are always selected on validation "
            "data (FSC22 fold 4 plus external validation sources), never on test results."),
          Spacer(1, 6)]

# ---------------- 6. model development ----------------
story += [P("7. Model development", "h1"),
          P("All V3 models keep the same architecture as V1 (MobileNetV3-Small, 128 × 196 × 3 input, three outputs) so the firmware is unchanged. "
            "Each step changed one factor, and most configurations were run with three random seeds."),
          table([["Version", "What changed", "Main finding"],
                 ["Original / V1 / V2", "First stage (Section 2): scratch training on random folds; V1 and V2 fine-tuned on noisy mixes", "Reproduced exactly; V1 generalises poorly to unseen recordings (gunshot recall 11-28% in noise)"],
                 ["V3a", "Retrained on official folds 1-3: clean / + hard negatives / + mixing", "Mixing lifts gunshot-in-noise recall from 37% to 85% (+10 dB) but hurts chainsaws; hard negatives alone push towards \"no alert\""],
                 ["V3b B0", "+ external data (ESC-50, UrbanSound8K), balanced mixes", "Better on both classes; external hard-negative false alarms halved"],
                 ["V3b B1", "+ non-target events mixed into rainforest noise", "Background false alarms down (25% → 18% external)"],
                 ["V3b B2", "+ ImageNet-pretrained weights", "Best float model: 6% background and 5% hard-negative false alarms, 90% gunshot recall at +5 dB (external, 3 seeds)"],
                 ["V3-c", "+ gain down to 0.03×, simulated speaker chain, loud beds", "Quiet and replayed sounds much better (quiet gunshot 82% → 95%); false alarms roughly doubled"],
                 ["V3-d", "V3-c fine-tuned on 120 real replay clips", "Best float model for demos: real-replay gunshot 78%, chainsaw 86%, false alarms 8%. <b>Fails INT8</b>"],
                 ["V3-e", "V3-d recipe trained from scratch", "Quantises well, but chainsaw → gunshot rises to 6-11%; rejected"],
                 ["V3-f (deployed)", "V3-e + knowledge distillation from the V3-d ensemble", "Quantises well (96.7-98.3%) and keeps most of the teacher's accuracy"]],
                [1.05 * inch, 2.55 * inch, 3.3 * inch]),
          P("6.1 Training details", "h2"),
          bullets([
              "Optimiser Adam with an exponential learning-rate decay (1e-3 halving over the run for scratch models; 3e-4 for pretrained), "
              "batch size 32, balanced class weights, early stopping on validation balanced accuracy.",
              "Training data for V3-f: FSC22 folds 1-3 and external training sources: clean clips, random-gain copies, about 1,000 event-aware "
              "mixes per target class (SNR -5 to +20 dB, random position), 700 non-target-in-noise mixes, 450 loud-bed mixes, a simulated "
              "speaker/room/microphone chain on 30% of all examples, and the real replay training session expanded fivefold: 6,233 examples in total.",
              "Distillation: the targets are 50% the true label and 50% the averaged probabilities of the three V3-d seeds (teacher agreement "
              "with the true labels: 95%).",
              "Seed selection: the three V3-f seeds were scored on one shared validation set with a priority score (mean recall minus false "
              "alarms minus twice chainsaw-as-gunshot). Seeds 0 and 1 tied (0.522 vs 0.519); seed 0 was kept for its lower false alarms.",
          ]),
          P("6.2 What the experiments showed", "h2"),
          bullets([
              "The original gunshot weakness was a training-distribution problem: gunshots mixed into rainforest noise were barely represented.",
              "Extra data and ImageNet pretraining were the two largest accuracy gains. Pretraining, however, made the model unsuitable for INT8 (Section 8).",
              "Simulated speaker effects helped, but real phone replays remained harder than simulations. One real recording session gave the largest single gain on demo-style audio.",
              "Loud rain combined with a chainsaw can be mistaken for a gunshot; mixes with loud beds were added, but this remains a weaker condition (62% recall in the stress test).",
          ]),
          Spacer(1, 6)]

# ---------------- 7. INT8 ----------------
story += [P("8. INT8 quantization and deployment", "h1"),
          P("The ESP32 runs the model with TensorFlow Lite Micro in full INT8. Conversion uses 300 representative calibration examples drawn from "
            "validation conditions and real replay training clips (never test data). Each converted model is checked with TensorFlow Lite's "
            "reference kernels, the closest desktop equivalent of TensorFlow Lite Micro, by comparing its class decisions with the float model on every test set."),
          fig("fig_int8_agreement.png", W, "Figure 3. Share of clips where the INT8 model picks the same class as the float model. B0-V3-d were measured on 400 "
              "external test clips; V3-e and V3-f on the full external set. Scratch-trained models convert cleanly; every ImageNet-pretrained model degrades."),
          P("7.1 Diagnosing the INT8 failure", "h2"),
          bullets([
              "V3-d agreed with its float version on only 72-79% of clips; in INT8 its external gunshot recall fell from 86% to 57% and real-replay gunshot recall from 75% to 55%.",
              "Converting one checkpoint from each stage isolated the cause: only the pretrained models degrade. They also contain 4 always-zero "
              "squeeze-excite tensors and 8-10 layers with very wide activation ranges, against 1 such layer in the scratch models.",
              "A function-preserving patch of the dead squeeze-excite channels changed nothing (81.5% → 81.2%), so wide activation ranges are the likely cause. "
              "16-bit activations would test this, but the converter does not support them for HARD_SWISH.",
              "Training from scratch fixed quantisation but lost accuracy (V3-e). Distillation recovered most of it while keeping scratch-style activation ranges (V3-f).",
          ]),
          P("7.2 Deployed model", "h2"),
          table([["Property", "Value"],
                 ["File", "models/ESP32_Deployment_V3/MobileNetV3_Rainforest_V3f_INT8.tflite and rainforest_model_V3f.h"],
                 ["Size", f"{conv['bytes']:,} bytes (1.19 MiB), the same as V1"],
                 ["Operators", f"{', '.join(conv['ops'])}: {conv['n_ops']} nodes, all already registered by the firmware"],
                 ["Input / output", f"int8 [1,128,196,3], scale {conv['input']['scale']:.6f}, zero point {conv['input']['zero_point']} / int8 [1,3], scale {conv['output']['scale']}, zero point {conv['output']['zero_point']}"],
                 ["Float/INT8 agreement", " · ".join(f"{n}: {pct(v['class_agreement'], 1)}" for n, v in (("FSC22 fold 5", conv['validation']['eval_v1']), ("external", conv['validation']['eval_ext']), ("stress", conv['validation']['eval_stress']), ("real replay", conv['validation']['replay_test'])))],
                 ["Firmware footprint", "Detector sketch 1.76 MB flash (52%); mesh + WiFi sketch 2.47 MB (74%); 4 MiB tensor arena in PSRAM"]],
                [1.5 * inch, 5.4 * inch]),
          Spacer(1, 6)]

# ---------------- 8. firmware ----------------
story += [KeepTogether([P("9. ESP32 firmware: audio pipeline", "h1"),
          P("Each detection cycle runs on core 1 and takes about 7-8 seconds:"),
          table([["Step", "What happens", "Time"],
                 ["1. Capture", "I2S standard mode, 20 kHz, 32-bit mono slots, read in 512-sample blocks into PSRAM; samples scaled by 1/2^31", "5.0 s"],
                 ["3. Health check", "Rejects silence (no non-zero samples), clipping (>1% near full scale) and capture dropouts (any run of ≥32 exact zeros)", "<10 ms"],
                 ["4. DC removal", "Subtracts the recording's mean (after the health check, which needs the raw zeros)", "<10 ms"],
                 ["5. Log-Mel", "Three STFTs (2048/1024/512) with arduinoFFT, zero padding, sparse Mel filters, dB with 80 dB floor", "about 1.2 s"],
                 ["6. Inference", "Quantise input, run INT8 MobileNetV3-Small with TensorFlow Lite Micro, dequantise the 3 outputs", "about 1.0 s"],
                 ["7. Decision", "Alert if the top class is CHAINSAW or GUNSHOT with probability ≥ threshold (0.50 demo / 0.80 field); 30 s cooldown per class", "-"]],
                [1.0 * inch, 4.9 * inch, 1.0 * inch])]),
          P("Mel and inference times are from the V1 measurements; the V3-f model has the same architecture and operator count. "
            "Because capture and processing are sequential, the node is not listening for about 2.2 s of each cycle.", "cap"),
          P("8.1 Alert threshold", "h2"),
          P("The threshold trades detections against false alarms. Measured on the deployed model:"),
          table([["Test set", "Threshold", "Chainsaw alerted", "Gunshot alerted", "False alarms", "Chainsaw sent as gunshot"]] +
                [[name, f"{t:.2f}", pct(dep[f'{k}@{t}']['cs']), pct(dep[f'{k}@{t}']['gs']), pct(dep[f'{k}@{t}']['neg_FA'], 1), pct(dep[f'{k}@{t}']['cs_to_gs'], 1)]
                 for name, k in (("External", "eval_ext"), ("FSC22 fold 5", "eval_v1"), ("Real replay", "replay_test")) for t in (0.5, 0.8)],
                [1.3 * inch, 0.8 * inch, 1.15 * inch, 1.1 * inch, 1.0 * inch, 1.35 * inch]),
          P("0.50 is used for demonstrations; 0.80 halves false alarms and is recommended for field use.", "cap"),
          Spacer(1, 6)]

# ---------------- 9. hardware debugging ----------------
story += [P("10. Hardware debugging and live tests", "h1"),
          P("9.1 The original chainsaw-as-gunshot error", "h2"),
          P("The live failure (chainsaw classified as GUNSHOT at 97.3%) came from a capture with 27,940 exact-zero samples (28%). "
            "Simulating that fault on all 75 FSC22 chainsaw clips, by zeroing 28% of the audio in 512-sample blocks (the firmware's read size), "
            "turned 79% of them into GUNSHOT (mean probability 0.77). Random single-sample zeros, DC offset, 2× or 0.5× gain, clipping and "
            "18-bit quantisation did not. Blocks of digital silence create sharp edges in the spectrogram that look like impulses. The firmware now "
            "detects and skips such recordings instead of classifying them."),
          P("9.2 Microphone wiring diagnosis", "h2"),
          P("Two boards produced all-zero recordings. A diagnostic sketch read both I2S slots and then enabled the ESP32's internal pull-up on "
            "the data pin. Both slots then read 0xFFFFFFFF, which means nothing was driving the data line: the microphone was not powered, not clocked, "
            "or not connected, rather than a software or slot-selection problem. Intermittent dropouts (zero runs of 369 to 9,921 samples) matched a "
            "loose connection. After soldering the microphone, captures showed 100% non-zero samples and no dropouts."),
          KeepTogether([P("9.3 Live tests on the ESP32", "h2"),
          bullets([
              "Quiet room: every capture classified BACKGROUND at 95-99.6%; a startup spike (peak 0.99) after DC removal was correctly ignored.",
              "Phone playback (before soldering): chainsaw windows classified CHAINSAW at 83.2% and 99.6%; one gunshot window GUNSHOT at 89.5% "
              "(the adjacent window, where the shot was split across windows, reached only 19.5%); no false alerts at the 0.80 threshold; "
              "6 of 19 windows skipped because of dropouts, all caught by the new check.",
              "Mac-microphone tests with YouTube clips showed that phone speakers remove much of a gunshot's low-frequency energy and that "
              "recorded levels were 12-40× quieter than training audio. This finding led to V3-c and V3-d.",
          ])]),
          Spacer(1, 6)]

# ---------------- 10. mesh ----------------
story += [P("11. LoRa mesh network", "h1"),
          P("Every board runs identical firmware and plays three roles: it detects locally, relays other nodes' messages, and (if gateway mode is "
            "on) delivers alerts off the mesh. Node IDs are derived from each chip's MAC address; gateway mode and other settings are stored in "
            "flash and survive power cycles. The mesh is a managed flood, the same basic approach Meshtastic uses, which is appropriate for tens of "
            "nodes reporting rare events."),
          P("10.1 Packet format (24 bytes, about 0.26 s on air at SF9 / 125 kHz)", "h2"),
          table([["Field", "Bytes", "Purpose"],
                 ["magic 'RF', version, network ID", "4", "Ignore other LoRa devices (a stray classroom LoRa device was seen earlier)"],
                 ["type", "1", "DETECTION, ACK or HEARTBEAT"],
                 ["origin, message ID, attempt", "5", "De-duplication key (origin, ID, attempt) so each copy is relayed once"],
                 ["TTL, hops, sender", "4", "Hop limit (5), hop count and last transmitter"],
                 ["class, confidence", "2", "CHAINSAW / GUNSHOT (high bit marks a test trigger); confidence × 255"],
                 ["origin uptime", "4", "Seconds since the origin booted"],
                 ["reference origin, reference ID", "4", "ACKs: which detection is confirmed"]],
                [2.1 * inch, 0.6 * inch, 4.2 * inch]),
          P("10.2 Protocol behaviour", "h2"),
          bullets([
              "<b>Relaying:</b> a node that receives an unseen message with TTL > 1 waits a random 50-400 ms, checks the channel is free (channel "
              "activity detection, up to 5 tries), decrements TTL and retransmits. A 64-entry cache suppresses duplicates.",
              "<b>Reliability:</b> the origin keeps up to 4 pending detections and retries (up to 4 attempts, about 6 s × attempt plus jitter) until "
              "a gateway ACK returns. Retries carry a new attempt number so relays forward them again; the gateway reports each detection once but ACKs every attempt.",
              "<b>Heartbeats</b> every 2-2.3 minutes show which nodes are alive.",
              "<b>Concurrency:</b> the SX1262 receives in the background and signals through an interrupt; the mesh task on core 0 handles packets, retries, "
              "Serial commands and the BOOT-button test trigger, so audio capture and inference on core 1 never block the radio.",
          ]),
          P("10.3 Measured results (two boards on one desk)", "h2"),
          P("Test detections reached the gateway in about 0.3 s and the ACK returned in under 1 s, on the first attempt, at RSSI -27 to -30 dBm "
            "and SNR 11 dB, with no corrupted or foreign packets. Multi-hop relaying is implemented but needs a third board to test. Real-world range "
            "has not yet been measured."),
          Spacer(1, 6)]

# ---------------- 11. gateway ----------------
story += [P("12. Gateway, WiFi and phone notifications", "h1"),
          P("A node with gateway mode on delivers each detection it receives in two ways:"),
          bullets([
              "<b>USB:</b> one JSON line per event (prefixed @JSON) with class, confidence, origin, hops, RSSI and SNR. gateway_bridge.py on the Mac "
              "prints these, appends them to gateway_log.jsonl and can forward them to any web endpoint.",
              "<b>WiFi push notification:</b> an HTTPS POST to ntfy.sh on a private, randomly generated topic; the ntfy phone app shows the alert. "
              "Priority is 5 (urgent) for gunshots, 4 for chainsaws and 3 for tests. A \"gateway online\" message confirms setup at each boot.",
          ]),
          P("Notifications run in their own task on core 0, so a slow or missing WiFi connection never delays the radio. Alerts queue (up to 8) until "
            "WiFi returns, and the gateway reconnects every 20 seconds. WiFi settings are entered once over USB (ssid, pass) and stored in flash."),
          P("11.1 Lessons from setup", "h2"),
          bullets([
              "The ESP32-S3 supports only 2.4 GHz WiFi; an iPhone hotspot needs \"Maximize Compatibility\" turned on.",
              "The hotspot's real name contained a curly apostrophe and a trailing space (\"Charlie’s iPhone \"). A scan command lists exactly what the "
              "board sees, and the ssid command accepts quotes to keep spaces.",
              "Networks that need a login page or per-user credentials (for example eduroam) are not supported; a phone hotspot is the reliable option.",
          ]),
          P("11.2 Security notes", "h2"),
          bullets([
              "The ntfy topic is private only because it is hard to guess; anyone who knows it can read the alerts.",
              "HTTPS certificate checking is disabled in this prototype (setInsecure), so a malicious network could intercept alerts; pinning the ntfy certificate would fix this.",
              "Mesh packets are not authenticated. A message authentication code with a shared key would stop spoofed alerts in a real deployment.",
          ]),
          Spacer(1, 6)]

# ---------------- 12. comparison ----------------
b = lambda n, k: base[f"{n}|{k}"]
story += [P("13. Results comparison", "h1"),
          P("Reproduced measurements, by model. FSC22 numbers for the original, V1 and V2 models are inflated because those models trained on about 80% "
            "of the test clips; the external set is clean for every model. A dash means not measured."),
          table([["Measurement", "Original", "V1", "V2", "V3-f INT8 (deployed)"],
                 ["Original evaluation: clean accuracy (random fold 5, n=165)", pct(b('Original(BN09 fold5)', 'clean_val')['acc'], 1), pct(b('V1', 'clean_val')['acc'], 1), pct(b('V2', 'clean_val')['acc'], 1), "-"],
                 ["Original evaluation: noisy accuracy (n=22)", pct(b('Original(BN09 fold5)', 'noisy_val')['acc'], 1), pct(b('V1', 'noisy_val')['acc'], 1), pct(b('V2', 'noisy_val')['acc'], 1), "-"],
                 ["External: background false alarms", "-", pct(V1X['bg false-alarm']), "-", pct(E['bg false-alarm'])],
                 ["External: hard-negative false alarms", "-", pct(V1X['hardneg false-alarm']), "-", pct(E['hardneg false-alarm'])],
                 ["External: clean chainsaw / gunshot recall", "-", f"{pct(V1X['clean_cs recall'])} / {pct(V1X['clean_gs recall'])}", "-", f"{pct(E['clean_cs recall'])} / {pct(E['clean_gs recall'])}"],
                 ["External: gunshot recall at -5 / +5 / +15 dB", "-", " / ".join(pct(V1X[f'mix_gs {s}dB recall']) for s in ('-5', '+5', '+15')), "-", " / ".join(pct(E[f'mix_gs {s}dB recall']) for s in ('-5', '+5', '+15'))],
                 ["External: chainsaw recall at -5 / +5 / +15 dB", "-", " / ".join(pct(V1X[f'mix_cs {s}dB recall']) for s in ('-5', '+5', '+15')), "-", " / ".join(pct(E[f'mix_cs {s}dB recall']) for s in ('-5', '+5', '+15'))],
                 ["External: chainsaw sent as gunshot", "-", pct(V1X['_cs->gs (all cs rows)'], 1), "-", pct(E['_cs->gs (all cs rows)'], 1)],
                 ["FSC22 fold 5: background / hard-negative false alarms", "inflated", "inflated", "inflated", f"{pct(F5['bg false-alarm'])} / {pct(F5['hardneg false-alarm'])}"],
                 ["Stress: replayed gunshot / chainsaw recall", "-", "-", "-", f"{pct(ST['replayed_gs'])} / {pct(ST['replayed_cs'])}"],
                 ["Stress: quiet gunshot / chainsaw recall", "-", "-", "-", f"{pct(ST['quiet_gs'])} / {pct(ST['quiet_cs'])}"],
                 ["Real replay: gunshot / chainsaw / false alarms", "-", "-", "-", f"{pct(RP5['gs'])} / {pct(RP5['cs'])} / {pct(RP5['neg_FA'])}"],
                 ["Model size", "-", "1,248,792 B", "-", f"{conv['bytes']:,} B"],
                 ["Float/INT8 agreement", "-", "97.0% (clean val)", "-", pct(conv['validation']['eval_ext']['class_agreement'], 1) + " (external)"]],
                [2.55 * inch, 0.8 * inch, 1.15 * inch, 0.6 * inch, 1.8 * inch], bold_first_col=False),
          fig("fig_snr.png", W, "Figure 4. Recall against event SNR on the external test set. V3-f is far better on gunshots. On chainsaws it is better at "
              "low SNR but slightly worse than V1 at +10 and +15 dB, a remaining weakness (13 clips from 6 sources, so each point is noisy)."),
          Spacer(1, 6)]

# ---------------- 13. limitations ----------------
story += [P("14. Limitations and risks", "h1"),
          bullets([
              "<b>Small target test sets.</b> FSC22 fold 5 has 15 chainsaw and 15 gunshot source clips; the external set 13 chainsaw clips from 6 recordings; the real replay test 12 chainsaws. A single clip moves some results by 5-8 points.",
              "<b>No real gunshots or chainsaws recorded in the field.</b> All tests use dataset recordings, played digitally or through a phone speaker.",
              "<b>Chainsaw weaknesses.</b> Clean FSC22 chainsaw recall is 60% (9 of 15), real replay 58%, and loud rain plus chainsaw 62%; the float-only V3-d was better on all three.",
              "<b>False alarms.</b> 10-20% of hard negatives (depending on the set) still alert at the 0.50 threshold; field use should use 0.80 and ideally require two consecutive windows to agree.",
              "<b>Listening gaps.</b> Sequential capture and processing leave the node deaf for about 30% of each cycle, and a short gunshot can be split across two windows.",
              "<b>Untested in deployment conditions.</b> Battery runtime, LoRa range outdoors, multi-hop relaying and weatherproofing are untested.",
              "<b>Regulatory.</b> Single-channel 915 MHz operation should be checked against local rules (for example FCC Part 15 in the US) before any field deployment.",
              "<b>Licensing.</b> Training data includes CC BY-NC material (ESC-50, UrbanSound8K).",
          ]),
          P("15. Recommended next steps", "h1"),
          bullets([
              "Record more real replay sessions (different distances, volumes, rooms and speakers), then retrain V3-f; this gave the largest gain on demo audio.",
              "Collect hard negatives in the demonstration environment (talking, doors, chairs) to reduce false alarms.",
              "Add two-window confirmation and double-buffered capture (record the next window while processing the current one).",
              "Add a \"node offline\" phone alert when heartbeats stop, and report battery voltage in heartbeats.",
              "Measure battery life, outdoor LoRa range and three-node relaying.",
              "Investigate quantisation-aware training, which could deploy the stronger pretrained V3-d accuracy in INT8.",
              "Authenticate mesh packets and pin the ntfy TLS certificate before any real deployment.",
          ]),
          Spacer(1, 6)]

# ---------------- appendices ----------------
story += [P("Appendix A. Reproducing the work", "h1"),
          P("Environment: Python 3.11 virtual environment with the exact package versions in requirements.txt (TensorFlow 2.21, Keras 3.15, librosa 0.11). "
            "Datasets are not in git; FSC22, ESC-50 and UrbanSound8K must be downloaded to datasets/. All scripts use fixed random seeds."),
          code("""source .venv/bin/activate
python experiments/audit/baseline_eval.py          # reproduce Original/V1/V2 baselines
python experiments/eval_set/build_eval.py          # FSC22 fold-5 test set
python experiments/external/prepare.py             # extract ESC-50 / UrbanSound8K clips
python experiments/external/dedupe.py              # remove clips duplicated in FSC22
python experiments/eval_set/build_eval_ext.py      # external test set
python experiments/eval_set/build_eval_stress.py   # stress test set
python experiments/v3/train_b.py B2 0              # V3b configurations (B0/B1/B2, seed)
python experiments/v3/finetune_replay.py 1         # V3-d (needs replay recordings)
python experiments/v3/train_e.py 0 distill         # V3-f (distilled, scratch)
python experiments/v3/convert_int8.py models/MobileNetV3_Small_Rainforest_V3b_F_s0/best.keras V3f"""),
          P("Firmware build and upload ($PORT is the board's /dev/cu.usbmodem… port):", "body"),
          code("""cp models/ESP32_Deployment_V3/rainforest_model_V3f.h firmware/Rainforest_Mesh_Node/rainforest_model.h
arduino-cli compile -b esp32:esp32:XIAO_ESP32S3:PSRAM=opi firmware/Rainforest_Mesh_Node
arduino-cli upload  -b esp32:esp32:XIAO_ESP32S3:PSRAM=opi -p $PORT firmware/Rainforest_Mesh_Node"""),
          KeepTogether([P("Appendix B. Device command reference", "h1"),
          P("Type into the board's USB serial connection at 115200 baud (or into gateway_bridge.py). Settings marked * are saved in flash."),
          table([["Command", "Effect"],
                 ["t / g / BOOT button", "Send a test CHAINSAW / GUNSHOT detection through the whole chain"],
                 ["s", "Status: node ID, gateway mode, radio, packet counters, WiFi and ntfy topic"],
                 ["gw 1 / gw 0 *", "Gateway mode on / off"],
                 ["id N *", "Set node ID (default: derived from the chip's MAC address)"],
                 ["ssid \"Name\" * / pass PASSWORD *", "WiFi network (quotes keep spaces) and password, for gateway phone alerts"],
                 ["ntfy TOPIC *", "Change the ntfy topic (a random private topic is generated on first boot)"],
                 ["notify", "Send a test phone notification"],
                 ["scan", "List the 2.4 GHz WiFi networks the board can see"],
                 ["wifi off", "Forget WiFi settings"]],
                [2.3 * inch, 4.6 * inch])]),
          ]


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7.5); c.setFillColor(INK2)
    c.drawString(0.8 * inch, 0.5 * inch, "Rainforest Engineering · Acoustic Threat Detection on the Edge")
    c.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"Page {doc.page}")
    c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(0.8 * inch, 0.65 * inch, letter[0] - 0.8 * inch, 0.65 * inch)
    c.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.75 * inch, bottomMargin=0.85 * inch,
                      title="Rainforest Engineering: Acoustic Threat Detection on the Edge", author="Charlie Boggs",
                      subject="Technical report: model, ESP32 firmware, LoRa mesh and notifications")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="title", frames=[frame]), PageTemplate(id="body", frames=[frame], onPage=on_page)])
doc.build(story)
print("wrote", OUT)
