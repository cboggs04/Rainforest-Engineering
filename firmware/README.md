# ESP32 firmware (Seeed XIAO ESP32S3 + SPH0645 + Wio-SX1262)

| Sketch | Purpose |
|---|---|
| `Rainforest_Mesh_Node/` | **Current.** Detector + LoRa flooding mesh (relay, ACK/retry, heartbeats) + gateway mode (USB `@JSON` lines and WiFi phone alerts via ntfy). |
| `Rainforest_Detection_ESP32/` | Single-node detector (no mesh), with the zero-padding, dropout and DC-removal fixes. |
| `Rainforest_Mic_Slot_Test/` | Diagnostic: reads both I2S slots with a pull-up on the data pin (all `ffffffff` = mic not driving the line). |

## Before flashing
The 7.9 MB model header is not in git. Copy the V3-f header into the sketch folder:
```
cp models/ESP32_Deployment_V3/rainforest_model_V3f.h firmware/Rainforest_Mesh_Node/rainforest_model.h
```
If it is missing, regenerate it from the versioned model:
`python experiments/v3/convert_int8.py models/MobileNetV3_Small_Rainforest_V3b_F_s0/best.keras V3f`

## Build / upload (arduino-cli)
```
arduino-cli compile -b esp32:esp32:XIAO_ESP32S3:PSRAM=opi firmware/Rainforest_Mesh_Node
arduino-cli upload  -b esp32:esp32:XIAO_ESP32S3:PSRAM=opi -p /dev/cu.usbmodemXXXX firmware/Rainforest_Mesh_Node
```
Requires the esp32 core (3.3.x) and libraries arduinoFFT, TensorFlowLite_ESP32, RadioLib. OPI PSRAM must be enabled.

## Mesh node setup (Serial 115200, saved in flash)
`gw 1` gateway mode · `ssid "Exact Name"` / `pass PW` WiFi (2.4 GHz) · `ntfy TOPIC` · `notify` test alert ·
`scan` list WiFi networks · `t`/`g` test detections (or BOOT button) · `s` status · `id N` node ID.
On the Mac, `python gateway_bridge.py` reads the gateway's `@JSON` alerts over USB.
