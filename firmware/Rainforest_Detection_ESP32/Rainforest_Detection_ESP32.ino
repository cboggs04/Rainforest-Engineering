
#include <Arduino.h>
#include <ESP_I2S.h>
#include <arduinoFFT.h>
#include <SPI.h>
#include <RadioLib.h>
#include "esp_heap_caps.h"

#include "rainforest_model.h"
#include "mel_filterbanks.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <math.h>
#include <stdint.h>

// ==========================================
// RAINFOREST DETECTOR + LORA TRANSMITTER
// XIAO ESP32S3 + SPH0645 + Wio-SX1262
// ==========================================

// Microphone pins
#define MIC_BCLK D0
#define MIC_WS   D1
#define MIC_DIN  D2

// Wio-SX1262 B2B pins
#define LORA_NSS    41
#define LORA_DIO1   39
#define LORA_RST    42
#define LORA_BUSY   40
#define LORA_RF_SW  38
#define LORA_SCK     7
#define LORA_MISO    8
#define LORA_MOSI    9

// LoRa settings - receiver must match
constexpr float LORA_FREQ = 915.0f;
constexpr float LORA_BW = 125.0f;
constexpr uint8_t LORA_SF = 9;
constexpr uint8_t LORA_CR = 7;
constexpr uint8_t LORA_SYNC = 0x12;
constexpr int8_t LORA_POWER = 10;
constexpr uint16_t LORA_PREAMBLE = 8;
constexpr float LORA_TCXO = 1.8f;

// Audio settings
constexpr int SAMPLE_RATE = 20000;
constexpr int TOTAL_SAMPLES = 100000;
constexpr int BLOCK_SAMPLES = 512;
constexpr int N_MELS = 128;
constexpr int N_FRAMES = 196;
constexpr int N_CHANNELS = 3;
constexpr int HOP = 512;
constexpr int MAX_FFT = 2048;

constexpr size_t FEATURE_COUNT =
    (size_t)N_MELS * N_FRAMES * N_CHANNELS;

constexpr size_t ARENA_SIZE = 4 * 1024 * 1024;

constexpr float ALERT_THRESHOLD = 0.80f;

// A real microphone signal never holds exactly 0 for this many
// consecutive samples (1.6 ms at 20 kHz); such runs are I2S dropouts.
constexpr int MAX_ZERO_RUN = 32;
constexpr unsigned long ALERT_COOLDOWN_MS = 30000;

// Hardware
I2SClass I2S;
SX1262 radio = new Module(
    LORA_NSS, LORA_DIO1, LORA_RST, LORA_BUSY
);

// Memory
float* audio_buffer = nullptr;
float* mel_tensor = nullptr;
uint8_t* tensor_arena = nullptr;

static int32_t raw_samples[BLOCK_SAMPLES];

static float fft_real[MAX_FFT];
static float fft_imag[MAX_FFT];
static float power_spectrum[MAX_FFT / 2 + 1];

static float hann_2048[2048];
static float hann_1024[1024];
static float hann_512[512];

// TensorFlow Lite
tflite::MicroMutableOpResolver<8> resolver;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

bool system_ready = false;
bool lora_ready = false;

uint32_t packet_number = 0;
unsigned long last_alert_time = 0;
bool alert_sent_before = false;

const char* class_names[] = {
    "BACKGROUND",
    "CHAINSAW",
    "GUNSHOT"
};

// ==========================================
// HANN WINDOW
// ==========================================

void makeHann(float* window, int size) {
  for (int i = 0; i < size; i++) {
    window[i] =
        0.5f - 0.5f *
        cosf(2.0f * PI * i / size);
  }
}

// ==========================================
// REFLECT PADDING
// ==========================================

int reflectIndex(int index) {
  while (index < 0 || index >= TOTAL_SAMPLES) {
    if (index < 0) index = -index;

    if (index >= TOTAL_SAMPLES) {
      index = 2 * TOTAL_SAMPLES - 2 - index;
    }
  }
  return index;
}

// ==========================================
// INITIALIZE LORA
// ==========================================

bool initializeLoRa() {
  Serial.println("Initializing Wio-SX1262...");

  SPI.begin(
      LORA_SCK,
      LORA_MISO,
      LORA_MOSI,
      LORA_NSS
  );

  pinMode(LORA_RF_SW, OUTPUT);
  digitalWrite(LORA_RF_SW, HIGH);

  int state = radio.begin(
      LORA_FREQ,
      LORA_BW,
      LORA_SF,
      LORA_CR,
      LORA_SYNC,
      LORA_POWER,
      LORA_PREAMBLE,
      LORA_TCXO
  );

  if (state != RADIOLIB_ERR_NONE) {
    Serial.printf(
        "LORA ERROR: Initialization code %d\n",
        state
    );
    return false;
  }

  Serial.println("SUCCESS: LoRa initialized!");
  return true;
}

// ==========================================
// CAPTURE AUDIO
// ==========================================

bool captureAudio() {
  int captured = 0;
  unsigned long start_ms = millis();

  while (captured < TOTAL_SAMPLES) {
    int requested = min(
        BLOCK_SAMPLES,
        TOTAL_SAMPLES - captured
    );

    size_t bytes_read = I2S.readBytes(
        (char*)raw_samples,
        requested * sizeof(int32_t)
    );

    if (bytes_read == 0 ||
        bytes_read % sizeof(int32_t) != 0) {
      Serial.println("ERROR: Microphone read failed!");
      return false;
    }

    int count = bytes_read / sizeof(int32_t);

    if (count > requested) return false;

    for (int i = 0; i < count; i++) {
      audio_buffer[captured + i] =
          (float)raw_samples[i] / 2147483648.0f;
    }

    captured += count;
  }

  Serial.printf(
      "Captured %d samples in %lu ms\n",
      captured,
      millis() - start_ms
  );

  return true;
}

// ==========================================
// MICROPHONE HEALTH CHECK
// ==========================================

bool checkMicrophone() {
  double sum_sq = 0.0;
  float peak = 0.0f;
  int nonzero = 0;
  int clipped = 0;
  int zero_run = 0;
  int longest_zero_run = 0;
  int long_zero_runs = 0;

  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    float value = audio_buffer[i];

    if (!isfinite(value)) {
      Serial.println("MIC ERROR: Invalid sample");
      return false;
    }

    sum_sq += (double)value * value;

    float amplitude = fabsf(value);
    if (amplitude > peak) peak = amplitude;

    if (value != 0.0f) {
      nonzero++;
      zero_run = 0;
    } else {
      zero_run++;
      if (zero_run == MAX_ZERO_RUN) long_zero_runs++;
      if (zero_run > longest_zero_run) longest_zero_run = zero_run;
    }
    if (amplitude >= 0.99f) clipped++;
  }

  float rms = sqrt(sum_sq / TOTAL_SAMPLES);

  Serial.println();
  Serial.println("MICROPHONE STATUS");
  Serial.printf("RMS: %.6f\n", rms);
  Serial.printf("Peak: %.6f\n", peak);
  Serial.printf("Nonzero: %d / %d\n",
                nonzero, TOTAL_SAMPLES);
  Serial.printf("Near-full-scale: %d\n", clipped);
  Serial.printf("Longest zero run: %d samples (%d runs >= %d)\n",
                longest_zero_run, long_zero_runs, MAX_ZERO_RUN);

  if (nonzero == 0 || rms < 0.000001f) {
    Serial.println("NO AUDIO SIGNAL - skipping");
    return false;
  }

  // Blocks of exact zeros are capture dropouts, not sound. They create
  // sharp edges in the spectrogram that the model reads as impulses
  // (simulated 512-sample dropouts turned 79% of chainsaws into GUNSHOT).
  if (long_zero_runs > 0) {
    Serial.println("CAPTURE DROPOUT DETECTED - skipping");
    return false;
  }

  if (clipped > TOTAL_SAMPLES / 100) {
    Serial.println("POSSIBLE CLIPPING - skipping");
    return false;
  }

  Serial.println("AUDIO SIGNAL DETECTED");
  return true;
}

// ==========================================
// FFT POWER SPECTRUM
// ==========================================

void calculatePowerSpectrum(
    int n_fft,
    int frame,
    const float* window
) {
  int center = frame * HOP;
  int start = center - n_fft / 2;

  // Zero padding outside the recording, matching librosa 0.11
  // (center=True, pad_mode="constant") used to create the training features.
  for (int i = 0; i < n_fft; i++) {
    int index = start + i;

    float sample =
        (index >= 0 && index < TOTAL_SAMPLES)
            ? audio_buffer[index]
            : 0.0f;

    fft_real[i] = sample * window[i];
    fft_imag[i] = 0.0f;
  }

  ArduinoFFT<float> FFT(
      fft_real,
      fft_imag,
      n_fft,
      SAMPLE_RATE
  );

  FFT.compute(FFTDirection::Forward);

  for (int k = 0; k <= n_fft / 2; k++) {
    power_spectrum[k] =
        fft_real[k] * fft_real[k] +
        fft_imag[k] * fft_imag[k];
  }
}

// ==========================================
// GENERATE MEL CHANNEL
// ==========================================

void processChannel(
    int channel,
    int n_fft,
    const float* window,
    const MelFilter* filters,
    const float* coefficients
) {
  for (int frame = 0; frame < N_FRAMES; frame++) {
    calculatePowerSpectrum(n_fft, frame, window);

    for (int mel = 0; mel < N_MELS; mel++) {
      const MelFilter& filter = filters[mel];

      float energy = 0.0f;

      for (int j = 0; j < filter.length; j++) {
        energy +=
            power_spectrum[filter.start + j] *
            coefficients[filter.offset + j];
      }

      size_t index =
          ((size_t)mel * N_FRAMES + frame)
          * N_CHANNELS + channel;

      mel_tensor[index] = energy;
    }
  }
}

// ==========================================
// POWER TO DECIBELS
// ==========================================

void convertPowerToDB() {
  for (int channel = 0; channel < N_CHANNELS; channel++) {
    float maximum_db = -INFINITY;

    for (int mel = 0; mel < N_MELS; mel++) {
      for (int frame = 0; frame < N_FRAMES; frame++) {
        size_t index =
            ((size_t)mel * N_FRAMES + frame)
            * N_CHANNELS + channel;

        float db = 10.0f * log10f(
            fmaxf(mel_tensor[index], 1e-10f)
        );

        mel_tensor[index] = db;

        if (db > maximum_db) maximum_db = db;
      }
    }

    float floor_db = maximum_db - 80.0f;

    for (int mel = 0; mel < N_MELS; mel++) {
      for (int frame = 0; frame < N_FRAMES; frame++) {
        size_t index =
            ((size_t)mel * N_FRAMES + frame)
            * N_CHANNELS + channel;

        if (mel_tensor[index] < floor_db) {
          mel_tensor[index] = floor_db;
        }
      }
    }
  }
}

// ==========================================
// REMOVE DC OFFSET
// ==========================================
// The SPH0645 output carries a constant offset (~0.12 of full scale on
// these boards). Training audio has essentially none, so subtract the
// recording's mean. Runs AFTER checkMicrophone(), whose dropout check
// needs the raw exact-zero samples.

void removeDC() {
  double sum = 0.0;
  for (int i = 0; i < TOTAL_SAMPLES; i++) sum += audio_buffer[i];
  float mean = (float)(sum / TOTAL_SAMPLES);

  double sum_sq = 0.0;
  float peak = 0.0f;
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    audio_buffer[i] -= mean;
    sum_sq += (double)audio_buffer[i] * audio_buffer[i];
    float a = fabsf(audio_buffer[i]);
    if (a > peak) peak = a;
  }

  Serial.printf("DC offset removed: %+.6f\n", mean);
  Serial.printf("Sound level (AC RMS): %.6f  peak: %.6f\n",
                (float)sqrt(sum_sq / TOTAL_SAMPLES), peak);
}

// ==========================================
// GENERATE COMPLETE MEL TENSOR
// ==========================================

void generateMelTensor() {
  unsigned long start_ms = millis();

  processChannel(
      0, 2048, hann_2048,
      mel_filters_2048,
      mel_coefficients_2048
  );

  processChannel(
      1, 1024, hann_1024,
      mel_filters_1024,
      mel_coefficients_1024
  );

  processChannel(
      2, 512, hann_512,
      mel_filters_512,
      mel_coefficients_512
  );

  convertPowerToDB();

  Serial.printf(
      "Mel processing: %lu ms\n",
      millis() - start_ms
  );
}

// ==========================================
// QUANTIZE MODEL INPUT
// ==========================================

void quantizeMelTensor() {
  float scale = input_tensor->params.scale;
  int zero = input_tensor->params.zero_point;

  for (size_t i = 0; i < FEATURE_COUNT; i++) {
    int q = (int)roundf(
        mel_tensor[i] / scale + zero
    );

    if (q < -128) q = -128;
    if (q > 127) q = 127;

    input_tensor->data.int8[i] = (int8_t)q;
  }
}

// ==========================================
// SEND LORA TEST DETECTION
// ==========================================

void sendLoRaAlert(
    const char* sound,
    float confidence
) {
  if (!lora_ready) {
    Serial.println("LoRa unavailable - not sent");
    return;
  }

  unsigned long now = millis();

  if (alert_sent_before &&
      now - last_alert_time < ALERT_COOLDOWN_MS) {
    Serial.println("LoRa cooldown - not sent");
    return;
  }

  packet_number++;

  String message =
      "TEST|NODE_01|" +
      String(sound) + "|" +
      String(confidence * 100.0f, 1) +
      "|" + String(packet_number);

  Serial.println();
  Serial.print("LORA SENDING: ");
  Serial.println(message);

  int state = radio.transmit(message);

  if (state == RADIOLIB_ERR_NONE) {
    Serial.println("LORA: Transmission completed");
    last_alert_time = millis();
    alert_sent_before = true;
  } else {
    Serial.printf(
        "LORA ERROR: %d\n",
        state
    );
  }
}

// ==========================================
// RUN MOBILENETV3
// ==========================================

void runInference() {
  quantizeMelTensor();

  unsigned long start_ms = millis();

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return;
  }

  Serial.printf(
      "Inference time: %lu ms\n",
      millis() - start_ms
  );

  float probabilities[3] = {0, 0, 0};
  int predicted = 0;

  Serial.println();
  Serial.println("CLASSIFICATION RESULTS");
  Serial.println("======================");

  for (int i = 0; i < 3; i++) {
    probabilities[i] =
        (output_tensor->data.int8[i] -
         output_tensor->params.zero_point)
        * output_tensor->params.scale;

    Serial.printf(
        "%s: %.2f%%\n",
        class_names[i],
        probabilities[i] * 100.0f
    );

    if (probabilities[i] >
        probabilities[predicted]) {
      predicted = i;
    }
  }

  Serial.printf(
      "\nDETECTED: %s\n",
      class_names[predicted]
  );

  if (predicted != 0 &&
      probabilities[predicted] >= ALERT_THRESHOLD) {
    sendLoRaAlert(
        class_names[predicted],
        probabilities[predicted]
    );
  } else {
    Serial.println("No LoRa test alert triggered");
  }
}

// ==========================================
// SETUP
// ==========================================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("RAINFOREST ML + LORA DETECTOR");
  Serial.println("============================");

  if (!psramFound()) {
    Serial.println("ERROR: PSRAM missing!");
    return;
  }

  audio_buffer = (float*)heap_caps_malloc(
      TOTAL_SAMPLES * sizeof(float),
      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
  );

  mel_tensor = (float*)heap_caps_malloc(
      FEATURE_COUNT * sizeof(float),
      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
  );

  tensor_arena = (uint8_t*)heap_caps_malloc(
      ARENA_SIZE,
      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
  );

  if (!audio_buffer || !mel_tensor || !tensor_arena) {
    Serial.println("ERROR: Memory allocation failed!");
    return;
  }

  Serial.println("PSRAM allocated");

  makeHann(hann_2048, 2048);
  makeHann(hann_1024, 1024);
  makeHann(hann_512, 512);

  const tflite::Model* model =
      tflite::GetModel(rainforest_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch!");
    return;
  }

  if (
      resolver.AddAdd() != kTfLiteOk ||
      resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddHardSwish() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddMul() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk
  ) {
    Serial.println("ERROR: Operator registration failed");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      ARENA_SIZE
  );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Tensor allocation failed");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  if (input_tensor->type != kTfLiteInt8 ||
      output_tensor->type != kTfLiteInt8 ||
      input_tensor->bytes != FEATURE_COUNT ||
      output_tensor->bytes != 3) {
    Serial.println("ERROR: Tensor format mismatch");
    return;
  }

  Serial.println("MobileNetV3 initialized");

  // Initialize LoRa
  lora_ready = initializeLoRa();

  // Initialize microphone
  I2S.setPins(
      MIC_BCLK, MIC_WS, -1, MIC_DIN
  );

  if (!I2S.begin(
      I2S_MODE_STD,
      SAMPLE_RATE,
      I2S_DATA_BIT_WIDTH_32BIT,
      I2S_SLOT_MODE_MONO
  )) {
    Serial.println("ERROR: Microphone initialization failed");
    return;
  }

  system_ready = true;

  Serial.println("Microphone initialized");
  Serial.println("System ready!");

  if (!lora_ready) {
    Serial.println(
        "WARNING: Classification will run without LoRa"
    );
  }
}

// ==========================================
// MAIN LOOP
// ==========================================

void loop() {
  if (!system_ready) {
    delay(1000);
    return;
  }

  Serial.println();
  Serial.println("Listening for 5 seconds...");

  if (captureAudio()) {
    if (checkMicrophone()) {
      removeDC();
      generateMelTensor();
      runInference();
    }
  }

  delay(500);
}





