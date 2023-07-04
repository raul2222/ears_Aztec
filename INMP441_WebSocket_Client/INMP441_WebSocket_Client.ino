
#include <driver/i2s.h>
#include <WiFi.h>


#define I2S_SD 32
#define I2S_WS 25
#define I2S_SCK 33

#define I2S_PORT I2S_NUM_0

#define bufferCnt 4
#define bufferLen 512
int16_t sBuffer[bufferLen];

// Estructura del filtro
struct HighPassFilter {
  float x1;
  float y1;
  float alpha;
};

// Función para inicializar el filtro
void highPassFilterInit(HighPassFilter *filter, float cutoffFrequency, float sampleRate) {
  filter->x1 = 0.0;
  filter->y1 = 0.0;
  filter->alpha = 1.0 / (2.0 * M_PI * cutoffFrequency / sampleRate + 1.0);
}

// Función para aplicar el filtro
int16_t highPassFilterApply(HighPassFilter *filter, int16_t sample) {
  float y = filter->alpha * (filter->y1 + sample - filter->x1);
  filter->x1 = sample;
  filter->y1 = y;
  return (int16_t)y;
}

// Inicializa el filtro en la tarea micTask
HighPassFilter hpFilter;

const char* ssid = "Robot";
const char* password = "123456789";

const char* tcpAddress = "192.168.0.197";
const uint16_t tcpPort = 9999;  // <WEBSOCKET_SERVER_PORT>

boolean connected = false;
WiFiClient client; 

#define COMPRESSOR_THRESHOLD 10000 // Puedes ajustar este valor
#define COMPRESSOR_RATIO 4.0 // Puedes ajustar este valor

#define ACCUMULATED_SAMPLES 1
int16_t accumulatedBuffer[bufferLen * ACCUMULATED_SAMPLES];
int accumulatedIndex = 0;

#include <ESP32Servo.h>

// create four servo objects 
Servo servo1;
Servo servo2;

// Published values for SG90 servos; adjust if needed
int minUs = 500;
int maxUs = 2500;

// These are all GPIO pins on the ESP32
// Recommended pins include 2,4,12-19,21-23,25-27,32-33
// for the ESP32-S2 the GPIO pins are 1-21,26,33-42
// for the ESP32-S3 the GPIO pins are 1-21,35-45,47-48
// for the ESP32-C3 the GPIO pins are 1-10,18-21

int servo1Pin = 26;
int servo2Pin = 27;

int mytime = 39;

int pos = 0;      // position in degrees
ESP32PWM pwm;


void micTask(void* parameter) {
  highPassFilterInit(&hpFilter, 400.0, 32000.0);
  i2s_install();
  i2s_setpin();
  i2s_start(I2S_PORT);
  double amplificationFactor = 32; 
  size_t bytesIn = 0;
  while (1) {
    esp_err_t result = i2s_read(I2S_PORT, &sBuffer, bufferLen, &bytesIn, portMAX_DELAY);
    if (result == ESP_OK && connected) {
      
      for (int i = 0; i < bytesIn / 2; i++) {
          int16_t sample = ((int16_t*)sBuffer)[i];
          sample *= amplificationFactor;
          sample = (sample > INT16_MAX) ? INT16_MAX : (sample < -INT16_MAX) ? -INT16_MAX : sample;

          sample = highPassFilterApply(&hpFilter, sample);
          if (abs(sample) > COMPRESSOR_THRESHOLD) {
              int sign = (sample > 0) ? 1 : -1;
              sample = sign * COMPRESSOR_THRESHOLD + (abs(sample) - COMPRESSOR_THRESHOLD) / COMPRESSOR_RATIO;
          }
          accumulatedBuffer[accumulatedIndex] = sample;
          accumulatedIndex++;

          if (accumulatedIndex >= bufferLen * ACCUMULATED_SAMPLES) {  // buffer is full, send data
            if (client.connected()) {

              client.write((const uint8_t*)accumulatedBuffer, bufferLen * ACCUMULATED_SAMPLES * sizeof(int16_t));
              accumulatedIndex = 0;  // reset index
            } else {
              Serial.println("TCP connection lost!");
              connected = false;

              break;
            }
          }
      }
    }
  }
}



void i2s_install() {
  // Set up I2S Processor configuration
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 32000,
    //.sample_rate = 16000,
    .bits_per_sample = i2s_bits_per_sample_t(16),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S),
    .intr_alloc_flags = 0,
    .dma_buf_count = bufferCnt,
    .dma_buf_len = bufferLen,
    .use_apll = false
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin() {
  // Set I2S pin configuration
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_set_pin(I2S_PORT, &pin_config);
}

void normal_pose(){
  servo1.write(100);
  delay(1000);   
  servo2.write(72);
  delay(1000); 
}


void setup() {
  Serial.begin(115200);
	// Allow allocation of all timers
	//ESP32PWM::allocateTimer(0);
	//ESP32PWM::allocateTimer(1);
	//ESP32PWM::allocateTimer(2);
	ESP32PWM::allocateTimer(3);
	
	servo1.setPeriodHertz(250);      // Standard 50hz servo
	servo2.setPeriodHertz(250);      // Standard 50hz servo
  servo1.attach(servo1Pin, minUs, maxUs);
	servo2.attach(servo2Pin, minUs, maxUs);
  normal_pose();



  connectToWiFi(ssid, password);



  xTaskCreatePinnedToCore(micTask, "micTask", 10000, NULL, 1, NULL, 1);
  Serial.println("start");
 
}

void loop() {

      if(WiFi.status() != WL_CONNECTED){
        Serial.println("WiFi connection lost. Trying to reconnect...");
        WiFi.begin(ssid, password);
        
        while (WiFi.status() != WL_CONNECTED) {
            delay(1000);
            Serial.println("Reconnecting to WiFi...");
        }
 
        Serial.println("Reconnected to WiFi");
    }

      if(connected==false){
        if (!client.connect(tcpAddress, tcpPort)) {
          Serial.println("Connection failed!");
          connected = false;
        } else {
          connected = true;
          esp_restart();
        }
      }
      delay(1000);
}


void connectToWiFi(const char* ssid, const char* pwd) {
  Serial.println("Connecting to WiFi network: " + String(ssid));

  WiFi.disconnect(true);
  WiFi.onEvent(WiFiEvent);
  WiFi.begin(ssid, pwd);

  Serial.println("Waiting for WIFI connection...");
}




void WiFiEvent(WiFiEvent_t event) {
  switch (event) {
    case SYSTEM_EVENT_STA_GOT_IP:
      Serial.print("WiFi connected! IP address: ");
      Serial.println(WiFi.localIP());
      if(connected==false){
        if (!client.connect(tcpAddress, tcpPort)) {
          Serial.println("Connection failed!");
          connected = false;
        } else {
          connected = true;
        }
      }
      connected = true;
      break;
    case SYSTEM_EVENT_STA_DISCONNECTED:
      Serial.println("WiFi lost connection");
      connected = false;
      break;
  }
}

