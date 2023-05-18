#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>
#include <WiFi.h>
#include "AsyncUDP.h"
//#include <ArduinoWebsockets.h>
//#include "audio.h"
#include "driver/i2s.h"
#include "soc/i2s_reg.h"

//Set youy WiFi network name and password:
//const char* ssid = "master-2.4G-2233";
const char* ssid = "Robot";
const char* pswd = "123456789";

IPAddress udpAddress(192, 168, 11, 199);
const int udpPort = 12345; //UDP Listener Port:
boolean connected = false; //UDP State:
AsyncUDP udp; //Class UDP:


#define SAMPLE_RATE (44100)
#define BITS_PER_SAMPLE (I2S_BITS_PER_SAMPLE_16BIT)
// Mic I2S constants
#define MIC_I2S_PORT (I2S_NUM_0)
#define MIC_I2S_BCLK_PIN (GPIO_NUM_32)
#define MIC_I2S_WS_PIN (GPIO_NUM_25)
#define MIC_I2S_DATA_PIN (GPIO_NUM_33)
/* Speaker I2S constants
#define SPEAKER_I2S_PORT (I2S_NUM_1)
#define SPEAKER_I2S_BCLK_PIN (GPIO_NUM_4)
#define SPEAKER_I2S_WS_PIN (GPIO_NUM_5)
#define SPEAKER_I2S_DATA_PIN (GPIO_NUM_18)*/
#define READ_BUF_SIZE_BYTES (128)
static uint8_t mic_read_buf[READ_BUF_SIZE_BYTES];
//static uint8_t audio_output_buf[250];

static void init_mic() {

    i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 44100,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 128,
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

int32_t buffer[512];    // Buffer
volatile uint16_t rpt = 0; // Pointer

void readMic(){
    //size_t bytes_read = 0;
    //TickType_t ticks_to_wait = 100;
    size_t num_bytes_read = 0;
    i2s_read(MIC_I2S_PORT, (char*)buffer + rpt, READ_BUF_SIZE_BYTES, &num_bytes_read, portMAX_DELAY);

    rpt = rpt + num_bytes_read;
    if (rpt > 2043) rpt = 0;
}



void setup(){

    Serial.begin(115200);
    
    Serial.println("Configuring WiFi");
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, pswd);

    if (WiFi.waitForConnectResult() != WL_CONNECTED) {
        Serial.println("WiFi Failed");
        while (1) {
            delay(1000);
        }
    }

    init_mic();


}


void loop(){

      static uint8_t state = 0; 

 

      readMic();

      if (!connected) {
          if (udp.connect(udpAddress, udpPort)) {
              connected = true;
              Serial.println("Connected to UDP Listener");
              Serial.println("Under Linux for listener use: netcat -u -p 16500 -l | play -t s16 -r 48000 -c 2 -");
              Serial.println("Under Linux for recorder use: netcat -u -p 12345 -l | rec -t s16 -r 48000 -c 2 - file.mp3");

          }
      } else {
          switch (state) {
              case 0: // wait for index to pass halfway
                  if (rpt > 1023) {
                  state = 1;
                  }
                  break;
              case 1: // send the first half of the buffer
                  state = 2;
                  udp.write( (uint8_t *)buffer, 1024);
                  break;
              case 2: // wait for index to wrap
                  if (rpt < 1023) {
                      state = 3;
                  }
                  break;
              case 3: // send second half of the buffer
                  state = 0;
                  udp.write((uint8_t*)buffer+1024, 1024);
                  break;
          }
      }   

}





