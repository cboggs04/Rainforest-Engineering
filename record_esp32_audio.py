
import time
import wave
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

# ==========================================
# RECEIVE AUDIO FROM XIAO ESP32S3
# ==========================================

SAMPLE_RATE = 20000
DURATION_SECONDS = 5
EXPECTED_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
EXPECTED_BYTES = EXPECTED_SAMPLES * 2

OUTPUT_FILE = Path("esp32_recording.wav")

BAUD_RATE = 115200


def find_port():
    ports = list(list_ports.comports())

    print("Available serial ports:")

    for p in ports:
        print(f"  {p.device} | {p.description}")

    candidates = [
        p.device for p in ports
        if "usbmodem" in p.device.lower()
        or "usbserial" in p.device.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    return input(
        "\nEnter your ESP32 serial port: "
    ).strip()


def read_exact(ser, count):
    data = bytearray()

    while len(data) < count:
        chunk = ser.read(count - len(data))

        if not chunk:
            raise TimeoutError(
                f"Received {len(data)} of {count} bytes"
            )

        data.extend(chunk)

    return bytes(data)


def main():
    port = find_port()

    print("\nConnecting to:", port)

    with serial.Serial(
        port,
        BAUD_RATE,
        timeout=10,
        write_timeout=10
    ) as ser:

        # Allow USB serial to settle
        time.sleep(2)

        ser.reset_input_buffer()

        print("\nGet your phone ready.")
        input("Press Enter to start recording...")

        # Tell ESP32 to record
        ser.write(b"R")
        ser.flush()

        print("Recording for five seconds...")

        # ESP32 records before sending header
        header = ser.readline().decode(
            "ascii", errors="replace"
        ).strip()

        if not header.startswith("AUDIO_BEGIN "):
            raise RuntimeError(
                f"Unexpected ESP32 response: {header!r}"
            )

        count = int(header.split()[1])

        if count != EXPECTED_SAMPLES:
            raise RuntimeError(
                f"Unexpected sample count: {count}"
            )

        print("Receiving audio...")

        raw_audio = read_exact(
            ser,
            EXPECTED_BYTES
        )

    # ======================================
    # SAVE WAV
    # ======================================

    with wave.open(str(OUTPUT_FILE), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(raw_audio)

    samples = np.frombuffer(
        raw_audio,
        dtype="<i2"
    ).astype(np.float32) / 32768.0

    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    zero_fraction = float(np.mean(samples == 0))

    print("\nRECORDING COMPLETE")
    print("=" * 40)
    print("Saved:", OUTPUT_FILE.resolve())
    print("Duration:", DURATION_SECONDS, "seconds")
    print("Sample rate:", SAMPLE_RATE)
    print("RMS:", round(rms, 6))
    print("Peak:", round(peak, 6))
    print("Zero samples:", f"{zero_fraction * 100:.2f}%")

    print("\nOpen the WAV file to listen.")


if __name__ == "__main__":
    main()
