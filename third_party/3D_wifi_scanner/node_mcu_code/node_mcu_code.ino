#include <WiFi.h>
#include <WiFiClient.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// this code is running on a NODEMCU32 with esp32 chip
// the NODEMCU is the moving probe
// requesting /rssi in the NODEMCUs ip will return the current signal strength

const char* ssid = "NETWOKNAME"; // change to network ssid
const char* password = "PASSWORD"; // change to networks pw

WebServer server(80);

const int led = 2;

void handleRoot() {
  digitalWrite(led, HIGH);
  float rssi = WiFi.RSSI();
  delay(1);
  rssi += WiFi.RSSI();
  delay(1);
  rssi += WiFi.RSSI();
  delay(1);
  rssi += WiFi.RSSI();
  delay(1);
  rssi += WiFi.RSSI();
  rssi = rssi / 5.0;
  server.send(200, "application/json", "{\"signal\":" + String(rssi,2) + "}");
  digitalWrite(led, LOW);
}

void setup(void) {
  Serial.begin(9600);
  pinMode(led, OUTPUT);
  digitalWrite(led, 0);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  // Serial.println("");

  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    // Serial.print(".");
  }
  /*
    Serial.println("");
    Serial.print("Connected to ");
    Serial.println(ssid);
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  */
  if (MDNS.begin("esp32")) {
    // Serial.println("MDNS responder started");
  }

  server.on("/rssi", handleRoot);


  server.begin();
  // Serial.println("HTTP server started");
}

void loop(void) {
  server.handleClient();
  delay(5);
  // float rssi = WiFi.RSSI();
  // Serial.println(rssi);
}
