#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>
#include "audio.h"
#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include "audio.h"
#include "driver/i2s.h"
#include "soc/i2s_reg.h"

#define SAMPLE_RATE (44100)
#define BITS_PER_SAMPLE (I2S_BITS_PER_SAMPLE_16BIT)

// Mic I2S constants
#define MIC_I2S_PORT (I2S_NUM_0)
#define MIC_I2S_BCLK_PIN (GPIO_NUM_32)
#define MIC_I2S_WS_PIN (GPIO_NUM_25)
#define MIC_I2S_DATA_PIN (GPIO_NUM_33)

// Speaker I2S constants
#define SPEAKER_I2S_PORT (I2S_NUM_1)
#define SPEAKER_I2S_BCLK_PIN (GPIO_NUM_4)
#define SPEAKER_I2S_WS_PIN (GPIO_NUM_5)
#define SPEAKER_I2S_DATA_PIN (GPIO_NUM_18)

#define READ_BUF_SIZE_BYTES (250)

static uint8_t mic_read_buf[READ_BUF_SIZE_BYTES];

static uint8_t audio_output_buf[250];


const char* ssid = "Robot";
const char* password = "123456789";

const char* websocket_server_host = "192.168.11.199";
const uint16_t websocket_server_port = 8877;  // <WEBSOCKET_SERVER_PORT>

using namespace websockets;
WebsocketsClient client;
bool isWebSocketConnected;

void onEventsCallback(WebsocketsEvent event, String data) {
  if (event == WebsocketsEvent::ConnectionOpened) {
    Serial.println("Connnection Opened");
    isWebSocketConnected = true;
  } else if (event == WebsocketsEvent::ConnectionClosed) {
    Serial.println("Connnection Closed");
    isWebSocketConnected = false;
  } else if (event == WebsocketsEvent::GotPing) {
    Serial.println("Got a Ping!");
  } else if (event == WebsocketsEvent::GotPong) {
    Serial.println("Got a Pong!");
  }
}



static void init_mic() {

    i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = BITS_PER_SAMPLE,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0,
    };

    esp_err_t err = i2s_driver_install(MIC_I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        printf("Error initializing I2S Mic\n");
    }

    i2s_pin_config_t pin_config = {
        .mck_io_num = I2S_PIN_NO_CHANGE,
        .bck_io_num = MIC_I2S_BCLK_PIN,
        .ws_io_num = MIC_I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = MIC_I2S_DATA_PIN,
    };

    // Fixes for SPH0645 Mic
    // REG_SET_BIT(I2S_TIMING_REG(MIC_I2S_PORT), BIT(9));
    // REG_SET_BIT(I2S_CONF_REG(MIC_I2S_PORT), I2S_RX_MSB_SHIFT);

    err = i2s_set_pin(MIC_I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        printf("Error setting I2S mic pins\n");
    }

    err = i2s_set_clk(MIC_I2S_PORT, SAMPLE_RATE, BITS_PER_SAMPLE, I2S_CHANNEL_MONO);
    if (err != ESP_OK) {
        printf("Error setting I2S mic clock\n");
    }
}


static void audio_capture_task(void* task_param) {

    init_mic();

    size_t bytes_read = 0;
    TickType_t ticks_to_wait = 100;

    while (true) {
        esp_err_t res = i2s_read(MIC_I2S_PORT, (char*)mic_read_buf, READ_BUF_SIZE_BYTES, &bytes_read, ticks_to_wait);

        if (res == ESP_OK && isWebSocketConnected) {
          client.sendBinary((char*)mic_read_buf, bytes_read);
        }
    }
}

void init_audio() {
    xTaskCreate(audio_capture_task, "audio_capture_task", 4096,NULL, 10, NULL);
}


void setup(){

    Serial.begin(115200);
    connectWiFi();
    connectWSServer();
    init_audio();

    while (true) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}

void connectWiFi() {
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
}

void connectWSServer() {
  client.onEvent(onEventsCallback);
  while (!client.connect(websocket_server_host, websocket_server_port, "/")) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Websocket Connected!");
}


void loop(){

}