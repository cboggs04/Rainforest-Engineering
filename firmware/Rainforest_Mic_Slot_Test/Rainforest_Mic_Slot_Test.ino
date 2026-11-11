// Diagnostic: read BOTH I2S slots from the SPH0645 and report which one carries audio.
// Same pins / rate / bit width as Rainforest_Detection_ESP32, but STEREO instead of MONO.
#include <Arduino.h>
#include <ESP_I2S.h>
#include "driver/gpio.h"

#define MIC_BCLK D0
#define MIC_WS   D1
#define MIC_DIN  D2

I2SClass I2S;
static int32_t buf[2 * 512];   // interleaved frames: [slot0, slot1]

struct Stats { double sum = 0, sum_sq = 0; float peak = 0; int nonzero = 0, n = 0; int32_t first[3] = {0, 0, 0}; };

void add(Stats& s, int32_t raw) {
  if (s.n < 3) s.first[s.n] = raw;
  float v = raw / 2147483648.0f;
  s.sum += v; s.sum_sq += (double)v * v; s.n++;
  if (raw != 0) s.nonzero++;
  if (fabsf(v) > s.peak) s.peak = fabsf(v);
}

void report(const char* name, Stats& s) {
  double mean = s.sum / s.n;
  double ac = sqrt(fmax(s.sum_sq / s.n - mean * mean, 0.0));
  Serial.printf("  %s: nonzero %5d/%d  mean(DC) %+.5f  AC-RMS %.6f  peak %.4f  raw %08lx %08lx %08lx\n",
                name, s.nonzero, s.n, mean, ac, s.peak,
                (unsigned long)s.first[0], (unsigned long)s.first[1], (unsigned long)s.first[2]);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nMIC SLOT TEST (stereo read)");
  I2S.setPins(MIC_BCLK, MIC_WS, -1, MIC_DIN);
  if (!I2S.begin(I2S_MODE_STD, 20000, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("ERROR: I2S begin failed");
    while (true) delay(1000);
  }
  // Diagnostic: weak pull-up on the data pin. Undriven line -> reads all ones; driven low -> zeros.
  gpio_pullup_en((gpio_num_t)MIC_DIN);
  Serial.printf("I2S started (data pin GPIO%d, pull-up ON). Reporting each slot once per second.\n", (int)MIC_DIN);
}

void loop() {
  Stats a, b;
  while (a.n < 20000) {                       // 1 s of frames
    size_t got = I2S.readBytes((char*)buf, sizeof(buf));
    int frames = got / (2 * sizeof(int32_t));
    for (int i = 0; i < frames && a.n < 20000; i++) { add(a, buf[2 * i]); add(b, buf[2 * i + 1]); }
    if (got == 0) { Serial.println("ERROR: read returned 0 bytes"); break; }
  }
  Serial.printf("[%lus]\n", millis() / 1000);
  report("slot0 (first) ", a);
  report("slot1 (second)", b);
}
