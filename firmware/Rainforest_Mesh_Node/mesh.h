// ==========================================================================
// RAINFOREST MESH  -  flooding LoRa mesh for detector nodes
//
// Every node: detects locally, relays other nodes' packets once (TTL-limited,
// de-duplicated), and retries its own detections until a gateway ACKs them.
// A node with gateway mode ON prints received detections as JSON lines
// ("@JSON {...}") on USB serial for gateway_bridge.py on the Mac.
//
// Runs as a FreeRTOS task on core 0 so the radio keeps receiving while
// core 1 records audio and runs the model.
//
// Serial commands (115200 baud, newline-terminated):
//   t        fake a CHAINSAW detection (test, also: press BOOT button)
//   g        fake a GUNSHOT detection (test)
//   s        print status
//   id N     set this node's ID (1-65534), saved in flash
//   gw 1|0   gateway mode on/off, saved in flash
//
// Gateway phone notifications (WiFi + ntfy.sh), all saved in flash:
//   ssid NAME      WiFi network to join (2.4 GHz; phone hotspot is fine)
//   pass PASSWORD  WiFi password
//   ntfy TOPIC     ntfy topic (a random private one is generated on first boot)
//   notify         send a test notification
//   wifi off       forget WiFi settings
// ==========================================================================
#pragma once

#include <Arduino.h>
#include <SPI.h>
#include <RadioLib.h>
#include <Preferences.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "esp_mac.h"

// ---------- Wio-SX1262 on the XIAO ESP32S3 board-to-board connector ----------
#define LORA_NSS    41
#define LORA_DIO1   39
#define LORA_RST    42
#define LORA_BUSY   40
#define LORA_RF_SW  38
#define LORA_SCK     7
#define LORA_MISO    8
#define LORA_MOSI    9
#define BOOT_BUTTON  0

// ---------- radio settings (all nodes must match) ----------
constexpr float    LORA_FREQ     = 915.0f;
constexpr float    LORA_BW       = 125.0f;
constexpr uint8_t  LORA_SF       = 9;
constexpr uint8_t  LORA_CR       = 7;
constexpr uint8_t  LORA_SYNC     = 0x12;
constexpr int8_t   LORA_POWER    = 10;
constexpr uint16_t LORA_PREAMBLE = 8;
constexpr float    LORA_TCXO     = 1.8f;

// ---------- mesh settings ----------
constexpr uint8_t  MESH_MAGIC0       = 'R';
constexpr uint8_t  MESH_MAGIC1       = 'F';
constexpr uint8_t  MESH_VERSION      = 1;
constexpr uint8_t  MESH_NET_ID       = 0x5A;     // change to separate networks
constexpr uint8_t  MESH_DEFAULT_TTL  = 5;        // max hops
constexpr uint8_t  MESH_MAX_ATTEMPTS = 4;        // 1 send + 3 retries
constexpr uint32_t MESH_ACK_TIMEOUT  = 6000;     // ms, grows with attempt
constexpr uint32_t HEARTBEAT_MS      = 120000;   // +/- jitter
constexpr uint32_t DETECT_COOLDOWN   = 30000;    // per class, real detections

enum : uint8_t { PKT_DETECTION = 1, PKT_ACK = 2, PKT_HEARTBEAT = 3 };
constexpr uint8_t CLS_TEST_FLAG = 0x80;          // high bit of cls = test trigger

struct __attribute__((packed)) MeshPacket {
  uint8_t  magic0, magic1;
  uint8_t  version;
  uint8_t  net_id;
  uint8_t  type;
  uint16_t origin;        // node that created the message
  uint16_t msg_id;        // origin's message counter
  uint8_t  attempt;       // 1..MESH_MAX_ATTEMPTS (retries of the same msg_id)
  uint8_t  ttl;           // hops left
  uint8_t  hops;          // hops taken so far
  uint16_t sender;        // node that transmitted this copy
  uint8_t  cls;           // 1 CHAINSAW, 2 GUNSHOT (| CLS_TEST_FLAG)
  uint8_t  conf;          // confidence * 255
  uint32_t origin_uptime_s;
  uint16_t ref_origin;    // ACK: which detection is acknowledged
  uint16_t ref_msg;
};
static_assert(sizeof(MeshPacket) == 24, "MeshPacket must stay 24 bytes");

static const char* MESH_CLASS[] = {"BACKGROUND", "CHAINSAW", "GUNSHOT"};
static const char* clsName(uint8_t c) { c &= 0x7F; return c < 3 ? MESH_CLASS[c] : "?"; }

// ---------- state ----------
static SX1262 radio = new Module(LORA_NSS, LORA_DIO1, LORA_RST, LORA_BUSY);
static Preferences mesh_prefs;
static uint16_t mesh_node_id = 1;
static bool     mesh_gateway = false;
static bool     mesh_radio_ok = false;
static uint16_t mesh_next_msg = 0;
static volatile bool radio_irq = false;
static QueueHandle_t det_queue = nullptr;
static uint32_t last_detect_ms[3] = {0, 0, 0};
static bool     detected_before[3] = {false, false, false};

struct DetEvent { uint8_t cls; uint8_t conf; };

struct Pending { bool used; MeshPacket pkt; uint32_t next_ms; };
static Pending pending[4];

struct SeenKey { uint8_t type; uint16_t origin; uint16_t msg; uint8_t attempt; };
static SeenKey seen[64]; static uint8_t seen_pos = 0; static uint8_t seen_n = 0;

struct Reported { uint16_t origin; uint16_t msg; };
static Reported reported[32]; static uint8_t rep_pos = 0; static uint8_t rep_n = 0;

static struct { uint32_t rx_ok, rx_crc, rx_foreign, rx_dup, relayed, sent, acked, failed; } stats = {};

static void IRAM_ATTR onDio1() { radio_irq = true; }

// ---------- gateway -> phone notifications (WiFi + ntfy.sh) ----------
static String wifi_ssid, wifi_pass, ntfy_topic;
static volatile bool wifi_changed = false;
static volatile bool wifi_scan_requested = false;
static QueueHandle_t notify_queue = nullptr;
struct NotifyMsg { char title[48]; char body[160]; char tags[40]; uint8_t priority; };

static void queueNotify(const char* title, const char* body, uint8_t priority, const char* tags) {
  if (!notify_queue || wifi_ssid.length() == 0) return;
  NotifyMsg m;
  strlcpy(m.title, title, sizeof(m.title)); strlcpy(m.body, body, sizeof(m.body));
  strlcpy(m.tags, tags, sizeof(m.tags)); m.priority = priority;
  if (xQueueSend(notify_queue, &m, 0) != pdTRUE) Serial.println("[wifi] notification queue full - dropped");
}

// Phone notification for a detection that reached this gateway.
static void notifyDetection(const MeshPacket& p, float rssi, float snr, bool local) {
  bool test = p.cls & CLS_TEST_FLAG; uint8_t c = p.cls & 0x7F;
  char title[48], body[160];
  snprintf(title, sizeof(title), "%s%s detected", test ? "[TEST] " : "", clsName(c));
  if (local) snprintf(body, sizeof(body), "%s %.0f%% heard by gateway node %u's own microphone.", clsName(c), p.conf / 2.55f, mesh_node_id);
  else snprintf(body, sizeof(body), "%s %.0f%% from node %u (%u hop%s, RSSI %.0f dBm, SNR %.1f dB).",
                clsName(c), p.conf / 2.55f, p.origin, p.hops, p.hops == 1 ? "" : "s", rssi, snr);
  uint8_t prio = test ? 3 : (c == 2 ? 5 : 4);
  queueNotify(title, body, prio, test ? "test_tube" : (c == 2 ? "rotating_light" : "evergreen_tree,warning"));
}

static bool ntfyPost(const NotifyMsg& m) {
  WiFiClientSecure client;
  client.setInsecure();                       // prototype: no certificate pinning
  HTTPClient http;
  if (!http.begin(client, "https://ntfy.sh/" + ntfy_topic)) return false;
  http.setTimeout(8000);
  http.addHeader("Title", m.title);
  http.addHeader("Priority", String(m.priority));
  http.addHeader("Tags", m.tags);
  int code = http.POST((uint8_t*)m.body, strlen(m.body));
  http.end();
  if (code < 200 || code >= 300) Serial.printf("[wifi] ntfy HTTP %d\n", code);
  return code >= 200 && code < 300;
}

// Separate task so slow WiFi/HTTPS never blocks the LoRa radio task.
static void notifyTask(void*) {
  uint32_t last_attempt = 0; bool was_connected = false, announced = false;
  for (;;) {
    if (wifi_scan_requested) {
      wifi_scan_requested = false;
      WiFi.mode(WIFI_STA); WiFi.disconnect(); vTaskDelay(pdMS_TO_TICKS(200));
      Serial.println("[wifi] scanning (2.4 GHz only)...");
      int n = WiFi.scanNetworks();
      for (int i = 0; i < n; i++)
        Serial.printf("[wifi]   \"%s\"  RSSI %d dBm  channel %d\n", WiFi.SSID(i).c_str(), WiFi.RSSI(i), WiFi.channel(i));
      Serial.printf("[wifi] scan done: %d network(s)\n", n);
      WiFi.scanDelete(); was_connected = false; last_attempt = 0;
    }
    if (wifi_changed) { wifi_changed = false; WiFi.disconnect(); was_connected = false; last_attempt = 0; }
    if (!mesh_gateway || wifi_ssid.length() == 0) { vTaskDelay(pdMS_TO_TICKS(300)); continue; }
    if (WiFi.status() != WL_CONNECTED) {
      if (was_connected) { Serial.println("[wifi] connection lost"); was_connected = false; }
      if (last_attempt == 0 || millis() - last_attempt > 20000) {
        last_attempt = millis();
        Serial.printf("[wifi] connecting to \"%s\"...\n", wifi_ssid.c_str());
        WiFi.mode(WIFI_STA); WiFi.disconnect(); WiFi.begin(wifi_ssid.c_str(), wifi_pass.c_str());
      }
      vTaskDelay(pdMS_TO_TICKS(500)); continue;   // queued alerts wait until WiFi is back
    }
    if (!was_connected) {
      was_connected = true;
      Serial.printf("[wifi] connected to \"%s\", IP %s, RSSI %d dBm\n", wifi_ssid.c_str(), WiFi.localIP().toString().c_str(), WiFi.RSSI());
      if (!announced) {
        announced = true; char body[96];
        snprintf(body, sizeof(body), "Gateway node %u is online and forwarding mesh alerts.", mesh_node_id);
        queueNotify("Rainforest gateway online", body, 2, "satellite");
      }
    }
    NotifyMsg m;
    if (xQueueReceive(notify_queue, &m, pdMS_TO_TICKS(500)) == pdTRUE) {
      bool ok = false;
      for (int i = 0; i < 3 && !ok; i++) { ok = ntfyPost(m); if (!ok) vTaskDelay(pdMS_TO_TICKS(2000)); }
      Serial.printf("[wifi] phone notification %s: %s\n", ok ? "sent" : "FAILED", m.title);
    }
  }
}

// Returns true if already seen; otherwise records it.
static bool seenBefore(uint8_t type, uint16_t origin, uint16_t msg, uint8_t attempt) {
  for (int i = 0; i < seen_n; i++)
    if (seen[i].type == type && seen[i].origin == origin && seen[i].msg == msg && seen[i].attempt == attempt) return true;
  seen[seen_pos] = {type, origin, msg, attempt};
  seen_pos = (seen_pos + 1) % 64; if (seen_n < 64) seen_n++;
  return false;
}

static bool reportedBefore(uint16_t origin, uint16_t msg) {
  for (int i = 0; i < rep_n; i++) if (reported[i].origin == origin && reported[i].msg == msg) return true;
  reported[rep_pos] = {origin, msg}; rep_pos = (rep_pos + 1) % 32; if (rep_n < 32) rep_n++;
  return false;
}

static MeshPacket newPacket(uint8_t type) {
  MeshPacket p = {};
  p.magic0 = MESH_MAGIC0; p.magic1 = MESH_MAGIC1; p.version = MESH_VERSION; p.net_id = MESH_NET_ID;
  p.type = type; p.origin = mesh_node_id; p.msg_id = mesh_next_msg++; p.attempt = 1;
  p.ttl = MESH_DEFAULT_TTL; p.hops = 0; p.sender = mesh_node_id;
  p.origin_uptime_s = millis() / 1000;
  return p;
}

// Listen-before-talk, transmit, return to receive.
static void txPacket(const MeshPacket& p) {
  if (!mesh_radio_ok) return;
  for (int i = 0; i < 5; i++) {
    int16_t cad = radio.scanChannel();
    if (cad == RADIOLIB_CHANNEL_FREE) break;
    delay(random(60, 250));
  }
  int16_t st = radio.transmit((uint8_t*)&p, sizeof(p));
  if (st != RADIOLIB_ERR_NONE) Serial.printf("[mesh] TX error %d\n", st);
  radio_irq = false;
  radio.startReceive();
}

static void gatewayJson(const MeshPacket& p, float rssi, float snr, bool local) {
  const char* type = p.type == PKT_DETECTION ? "detection" : "heartbeat";
  Serial.printf("@JSON {\"type\":\"%s\",\"origin\":%u,\"msg\":%u,\"attempt\":%u,\"class\":\"%s\","
                "\"confidence\":%.3f,\"test\":%s,\"hops\":%u,\"via\":%u,\"rssi\":%.1f,\"snr\":%.1f,"
                "\"origin_uptime_s\":%lu,\"gateway\":%u,\"local\":%s}\n",
                type, p.origin, p.msg_id, p.attempt, p.type == PKT_DETECTION ? clsName(p.cls) : "",
                p.conf / 255.0f, (p.cls & CLS_TEST_FLAG) ? "true" : "false", p.hops, p.sender,
                local ? 0.0f : rssi, local ? 0.0f : snr, (unsigned long)p.origin_uptime_s,
                mesh_node_id, local ? "true" : "false");
}

static void relay(MeshPacket p) {
  if (p.ttl <= 1) return;
  p.ttl--; p.hops++; p.sender = mesh_node_id;
  delay(random(50, 400));                 // de-synchronise relays
  txPacket(p); stats.relayed++;
  Serial.printf("[mesh] relayed type %u from node %u msg %u (ttl now %u)\n", p.type, p.origin, p.msg_id, p.ttl);
}

static void handleRx() {
  uint8_t buf[64];
  size_t len = radio.getPacketLength();
  if (len > sizeof(buf)) len = sizeof(buf);
  int16_t st = radio.readData(buf, len);
  float rssi = radio.getRSSI(), snr = radio.getSNR();
  radio.startReceive();
  if (st == RADIOLIB_ERR_CRC_MISMATCH) { stats.rx_crc++; return; }
  if (st != RADIOLIB_ERR_NONE) return;

  MeshPacket p;
  if (len != sizeof(p) || buf[0] != MESH_MAGIC0 || buf[1] != MESH_MAGIC1 || buf[2] != MESH_VERSION || buf[3] != MESH_NET_ID) {
    stats.rx_foreign++;
    Serial.printf("[mesh] ignored foreign packet (%u bytes, RSSI %.0f)\n", (unsigned)len, rssi);
    return;
  }
  memcpy(&p, buf, sizeof(p));
  if (p.origin == mesh_node_id) return;                     // our own message echoed back
  if (seenBefore(p.type, p.origin, p.msg_id, p.attempt)) { stats.rx_dup++; return; }
  stats.rx_ok++;

  if (p.type == PKT_DETECTION) {
    Serial.printf("[mesh] RX DETECTION %s %.0f%% from node %u (msg %u, attempt %u, %u hops, via %u, RSSI %.0f, SNR %.1f)\n",
                  clsName(p.cls), p.conf / 2.55f, p.origin, p.msg_id, p.attempt, p.hops + 1, p.sender, rssi, snr);
    if (mesh_gateway) {
      MeshPacket shown = p; shown.hops = p.hops + 1;
      if (!reportedBefore(p.origin, p.msg_id)) { gatewayJson(shown, rssi, snr, false); notifyDetection(shown, rssi, snr, false); }
      MeshPacket ack = newPacket(PKT_ACK);
      ack.ref_origin = p.origin; ack.ref_msg = p.msg_id;
      seenBefore(ack.type, ack.origin, ack.msg_id, ack.attempt);
      delay(random(20, 150));
      txPacket(ack);
      Serial.printf("[mesh] ACK sent to node %u for msg %u\n", p.origin, p.msg_id);
    } else {
      relay(p);
    }
  } else if (p.type == PKT_ACK) {
    if (p.ref_origin == mesh_node_id) {
      for (auto& q : pending) if (q.used && q.pkt.msg_id == p.ref_msg) {
        q.used = false; stats.acked++;
        Serial.printf("[mesh] ACK received: gateway %u confirmed msg %u (%s, attempt %u, %u hops)\n",
                      p.origin, p.ref_msg, clsName(q.pkt.cls), q.pkt.attempt, p.hops + 1);
      }
    } else if (!mesh_gateway) {
      relay(p);
    }
  } else if (p.type == PKT_HEARTBEAT) {
    if (mesh_gateway) { MeshPacket shown = p; shown.hops = p.hops + 1; gatewayJson(shown, rssi, snr, false); }
    else relay(p);
  }
}

static void sendDetection(uint8_t cls, uint8_t conf) {
  MeshPacket p = newPacket(PKT_DETECTION);
  p.cls = cls; p.conf = conf;
  seenBefore(p.type, p.origin, p.msg_id, p.attempt);
  if (mesh_gateway) {                        // gateway's own detections go straight to the Mac
    gatewayJson(p, 0, 0, true);
    notifyDetection(p, 0, 0, true);
    return;
  }
  Serial.printf("[mesh] SENDING DETECTION %s %.0f%% (msg %u)\n", clsName(cls), conf / 2.55f, p.msg_id);
  txPacket(p); stats.sent++;
  for (auto& q : pending) if (!q.used) { q.used = true; q.pkt = p; q.next_ms = millis() + MESH_ACK_TIMEOUT; return; }
  Serial.println("[mesh] pending list full - this detection will not be retried");
}

static void checkRetries() {
  uint32_t now = millis();
  for (auto& q : pending) {
    if (!q.used || (int32_t)(now - q.next_ms) < 0) continue;
    if (q.pkt.attempt >= MESH_MAX_ATTEMPTS) {
      q.used = false; stats.failed++;
      Serial.printf("[mesh] NO ACK for msg %u after %u attempts - no gateway reachable?\n", q.pkt.msg_id, q.pkt.attempt);
      continue;
    }
    q.pkt.attempt++;
    seenBefore(q.pkt.type, q.pkt.origin, q.pkt.msg_id, q.pkt.attempt);
    Serial.printf("[mesh] retry msg %u (attempt %u)\n", q.pkt.msg_id, q.pkt.attempt);
    txPacket(q.pkt);
    q.next_ms = now + MESH_ACK_TIMEOUT * q.pkt.attempt + random(0, 1500);
  }
}

static void printStatus() {
  Serial.printf("[mesh] node %u | gateway %s | radio %s | rx ok %lu dup %lu crc %lu foreign %lu | sent %lu acked %lu failed %lu relayed %lu\n",
                mesh_node_id, mesh_gateway ? "ON" : "off", mesh_radio_ok ? "ok" : "FAILED",
                (unsigned long)stats.rx_ok, (unsigned long)stats.rx_dup, (unsigned long)stats.rx_crc, (unsigned long)stats.rx_foreign,
                (unsigned long)stats.sent, (unsigned long)stats.acked, (unsigned long)stats.failed, (unsigned long)stats.relayed);
  if (mesh_gateway) {
    if (wifi_ssid.length() == 0) Serial.printf("[wifi] not configured - use: ssid NAME, then pass PASSWORD | ntfy topic: %s\n", ntfy_topic.c_str());
    else Serial.printf("[wifi] \"%s\" %s | ntfy topic: %s\n", wifi_ssid.c_str(),
                       WiFi.status() == WL_CONNECTED ? "CONNECTED" : "not connected", ntfy_topic.c_str());
  }
}

static void handleCommand(String cmd) {
  cmd.trim();
  if (cmd == "t") { DetEvent e = {(uint8_t)(1 | CLS_TEST_FLAG), 230}; xQueueSend(det_queue, &e, 0); }
  else if (cmd == "g") { DetEvent e = {(uint8_t)(2 | CLS_TEST_FLAG), 230}; xQueueSend(det_queue, &e, 0); }
  else if (cmd == "s") printStatus();
  else if (cmd.startsWith("id ")) {
    long v = cmd.substring(3).toInt();
    if (v >= 1 && v <= 65534) { mesh_node_id = v; mesh_prefs.putUShort("id", mesh_node_id); Serial.printf("[mesh] node id set to %u (saved)\n", mesh_node_id); }
    else Serial.println("[mesh] id must be 1-65534");
  } else if (cmd.startsWith("gw ")) {
    mesh_gateway = cmd.substring(3).toInt() != 0; mesh_prefs.putBool("gw", mesh_gateway);
    Serial.printf("[mesh] gateway mode %s (saved)\n", mesh_gateway ? "ON" : "off");
  } else if (cmd.startsWith("ssid ")) {
    wifi_ssid = cmd.substring(5); wifi_ssid.trim();
    // ssid "Name with spaces " keeps everything between the quotes exactly (incl. trailing spaces)
    if (wifi_ssid.length() >= 2 && wifi_ssid.startsWith("\"") && wifi_ssid.endsWith("\"")) wifi_ssid = wifi_ssid.substring(1, wifi_ssid.length() - 1);
    mesh_prefs.putString("ssid", wifi_ssid); wifi_changed = true;
    Serial.printf("[wifi] network set to \"%s\" (saved)%s\n", wifi_ssid.c_str(), wifi_pass.length() ? "" : " - now set: pass PASSWORD");
  } else if (cmd.startsWith("pass ")) {
    wifi_pass = cmd.substring(5); mesh_prefs.putString("pass", wifi_pass); wifi_changed = true;
    Serial.printf("[wifi] password saved (%u characters)\n", wifi_pass.length());
  } else if (cmd.startsWith("ntfy ")) {
    ntfy_topic = cmd.substring(5); ntfy_topic.trim(); mesh_prefs.putString("topic", ntfy_topic);
    Serial.printf("[wifi] ntfy topic set to %s (saved)\n", ntfy_topic.c_str());
  } else if (cmd == "notify") {
    if (!mesh_gateway) Serial.println("[wifi] only the gateway sends notifications (gw 1)");
    else if (wifi_ssid.length() == 0) Serial.println("[wifi] set WiFi first: ssid NAME, pass PASSWORD");
    else { queueNotify("Rainforest test notification", "If you can read this, phone alerts are working.", 3, "white_check_mark"); Serial.println("[wifi] test notification queued"); }
  } else if (cmd == "scan") {
    wifi_scan_requested = true; Serial.println("[wifi] scan requested");
  } else if (cmd == "wifi off") {
    wifi_ssid = ""; wifi_pass = ""; mesh_prefs.remove("ssid"); mesh_prefs.remove("pass"); wifi_changed = true;
    Serial.println("[wifi] WiFi settings cleared");
  } else if (cmd.length()) Serial.println("[mesh] commands: t | g | s | id N | gw 1|0 | ssid NAME | pass PW | ntfy TOPIC | notify | wifi off");
}

static void meshTask(void*) {
  String line; bool btn_prev = true; uint32_t btn_ms = 0;
  uint32_t next_hb = millis() + 15000 + random(0, 10000);
  for (;;) {
    if (radio_irq) { radio_irq = false; handleRx(); }

    DetEvent e;
    while (xQueueReceive(det_queue, &e, 0) == pdTRUE) sendDetection(e.cls, e.conf);

    checkRetries();

    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\n' || c == '\r') { handleCommand(line); line = ""; } else if (line.length() < 120) line += c;
    }

    bool btn = digitalRead(BOOT_BUTTON);
    if (!btn && btn_prev && millis() - btn_ms > 500) {
      btn_ms = millis(); Serial.println("[mesh] BOOT button: test CHAINSAW detection");
      DetEvent t = {(uint8_t)(1 | CLS_TEST_FLAG), 230}; xQueueSend(det_queue, &t, 0);
    }
    btn_prev = btn;

    if ((int32_t)(millis() - next_hb) >= 0) {
      next_hb = millis() + HEARTBEAT_MS + random(0, 20000);
      MeshPacket hb = newPacket(PKT_HEARTBEAT);
      seenBefore(hb.type, hb.origin, hb.msg_id, hb.attempt);
      if (mesh_gateway) gatewayJson(hb, 0, 0, true); else txPacket(hb);
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// ---------------- public API ----------------

// Call once from setup(). Starts the mesh task on core 0.
static bool meshBegin() {
  mesh_prefs.begin("mesh", false);
  uint8_t mac[6]; esp_read_mac(mac, ESP_MAC_WIFI_STA);
  uint16_t mac_id = ((uint16_t)mac[4] << 8) | mac[5];
  if (mac_id == 0 || mac_id == 0xFFFF) mac_id = 1;
  mesh_node_id = mesh_prefs.getUShort("id", mac_id);
  mesh_gateway = mesh_prefs.getBool("gw", false);
  randomSeed(esp_random());
  mesh_next_msg = esp_random() & 0xFFFF;     // avoids reusing IDs relays may still remember

  pinMode(BOOT_BUTTON, INPUT_PULLUP);
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);
  pinMode(LORA_RF_SW, OUTPUT); digitalWrite(LORA_RF_SW, HIGH);
  int st = radio.begin(LORA_FREQ, LORA_BW, LORA_SF, LORA_CR, LORA_SYNC, LORA_POWER, LORA_PREAMBLE, LORA_TCXO);
  mesh_radio_ok = (st == RADIOLIB_ERR_NONE);
  if (mesh_radio_ok) { radio.setDio1Action(onDio1); radio.startReceive(); }
  else Serial.printf("LORA ERROR: init code %d - detection continues without mesh\n", st);

  // WiFi / ntfy settings (used only in gateway mode)
  wifi_ssid = mesh_prefs.getString("ssid", "");
  wifi_pass = mesh_prefs.getString("pass", "");
  ntfy_topic = mesh_prefs.getString("topic", "");
  if (ntfy_topic.length() == 0) {           // private, hard-to-guess default topic
    char t[32]; snprintf(t, sizeof(t), "rainforest-%08lx%04x", (unsigned long)esp_random(), (unsigned)(esp_random() & 0xFFFF));
    ntfy_topic = t; mesh_prefs.putString("topic", ntfy_topic);
  }

  det_queue = xQueueCreate(8, sizeof(DetEvent));
  notify_queue = xQueueCreate(8, sizeof(NotifyMsg));
  xTaskCreatePinnedToCore(meshTask, "mesh", 8192, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(notifyTask, "notify", 12288, nullptr, 1, nullptr, 0);
  Serial.printf("MESH node %u | gateway %s | radio %s | %.1f MHz SF%u net 0x%02X\n",
                mesh_node_id, mesh_gateway ? "ON" : "off", mesh_radio_ok ? "ok" : "FAILED",
                LORA_FREQ, LORA_SF, MESH_NET_ID);
  if (mesh_gateway) Serial.printf("GATEWAY phone alerts: WiFi %s | ntfy topic %s\n",
                                  wifi_ssid.length() ? wifi_ssid.c_str() : "(not set)", ntfy_topic.c_str());
  Serial.println("MESH commands: t | g | s | id N | gw 1|0 | ssid NAME | pass PW | ntfy TOPIC | notify | wifi off");
  return mesh_radio_ok;
}

// Called by the detector (core 1) when the model fires. Non-blocking.
// cls: 1 CHAINSAW, 2 GUNSHOT.  Applies a per-class cooldown.
static void meshReportDetection(uint8_t cls, float confidence) {
  if (cls < 1 || cls > 2 || !det_queue) return;
  uint32_t now = millis();
  if (detected_before[cls] && now - last_detect_ms[cls] < DETECT_COOLDOWN) {
    Serial.printf("[mesh] %s cooldown - not sent\n", clsName(cls));
    return;
  }
  detected_before[cls] = true; last_detect_ms[cls] = now;
  DetEvent e = {cls, (uint8_t)constrain(lroundf(confidence * 255.0f), 0, 255)};
  xQueueSend(det_queue, &e, 0);
}
