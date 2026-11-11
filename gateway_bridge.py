"""Mac-side gateway for the Rainforest LoRa mesh.

The ESP32 in gateway mode prints every detection it receives as a line
"@JSON {...}" on USB serial. This script reads those lines, shows them,
appends them to a JSON-lines log, and optionally forwards each one to a web
endpoint (--webhook URL, HTTP POST of the JSON). Anything you type is sent to
the board, so mesh commands (t, g, s, id N, gw 1|0) work from here too.

  python gateway_bridge.py                      # auto-detect the board
  python gateway_bridge.py --port /dev/cu.usbmodem101 --debug
  python gateway_bridge.py --webhook https://example.com/hook
"""
import argparse, json, sys, threading, time, urllib.request
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports


def find_port():
    ports = [p.device for p in list_ports.comports() if "usbmodem" in p.device or "usbserial" in p.device]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        sys.exit("No ESP32 found. Plug in the gateway board or pass --port.")
    sys.exit(f"Several boards found {ports}; pass --port for the gateway one.")


def post(url, event):
    try:
        req = urllib.request.Request(url, data=json.dumps(event).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:  # never let a network problem stop the gateway
        print(f"   ! webhook failed: {e}")


def show(ev):
    t = ev["received_at"][11:19]
    if ev["type"] == "detection":
        tag = " [TEST]" if ev.get("test") else ""
        where = "this gateway's own mic" if ev.get("local") else f"{ev['hops']} hop(s), via node {ev['via']}, RSSI {ev['rssi']} dBm, SNR {ev['snr']} dB"
        print(f"{t}  ALERT{tag}: {ev['class']} {ev['confidence']*100:.0f}%  from node {ev['origin']}  ({where}, attempt {ev['attempt']})")
    elif ev["type"] == "heartbeat":
        print(f"{t}  heartbeat from node {ev['origin']} (up {ev['origin_uptime_s']//60} min, {ev['hops']} hop(s))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log", default="gateway_log.jsonl"); ap.add_argument("--webhook")
    ap.add_argument("--debug", action="store_true", help="also print the board's normal log lines")
    a = ap.parse_args()
    port = a.port or find_port()
    ser = serial.Serial(port, a.baud, timeout=1)
    log = Path(a.log)
    print(f"Gateway bridge on {port}. Logging to {log.resolve()}.")
    print("Waiting for alerts: the detector nodes listen on their own and alerts appear here as they arrive.")
    print("(Optional: type t = test chainsaw, g = test gunshot, s = status, then Enter.)")

    def keyboard():
        for line in sys.stdin:
            ser.write((line.strip() + "\n").encode())
    threading.Thread(target=keyboard, daemon=True).start()

    while True:
        try:
            raw = ser.readline().decode(errors="replace").strip()
        except serial.SerialException as e:
            print(f"Serial error ({e}); reconnecting in 3 s"); time.sleep(3)
            ser = serial.Serial(port, a.baud, timeout=1); continue
        if not raw:
            continue
        if raw.startswith("@JSON "):
            try:
                ev = json.loads(raw[6:])
            except json.JSONDecodeError:
                print(f"   ! bad JSON from board: {raw}"); continue
            ev["received_at"] = datetime.now().isoformat(timespec="seconds")
            with log.open("a") as f:
                f.write(json.dumps(ev) + "\n")
            show(ev)
            if a.webhook and ev["type"] == "detection":
                threading.Thread(target=post, args=(a.webhook, ev), daemon=True).start()
        elif a.debug or raw.startswith(("[mesh]", "[wifi]", "MESH", "GATEWAY")):
            print(f"   {raw}")


if __name__ == "__main__":
    main()
