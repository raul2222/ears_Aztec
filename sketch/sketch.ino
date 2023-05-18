#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "I2SSampler.h"
#include <Wire.h>
#include <SparkFun_WM8960_Arduino_Library.h> 
// Click here to get the library: http://librarymanager/All#SparkFun_WM8960
WM8960 codec;

// replace the ip address with your machine's ip address
#define I2S_SERVER_URL "http://192.168.11.199:5003/i2s_samples"
#define PING_URL "http://192.168.11.199:5003/ping"
// replace the SSID and PASSWORD with your WiFi settings
#define SSID "Robot"
#define PASSWORD "123456789"
// i2s config - this is set up to read fro the left channel
/*
i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 44100,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),//I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
};
*/
i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 44100,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format =  (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0,
    .mclk_multiple = i2s_mclk_multiple_t(I2S_MCLK_MULTIPLE_DEFAULT),
    .bits_per_chan = i2s_bits_per_chan_t(I2S_BITS_PER_CHAN_DEFAULT)
};


// i2s pins
//codec
i2s_pin_config_t i2s_pins = {
    .bck_io_num = 17,
    .ws_io_num = 16,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = 4};


// i2s pins
/*
i2s_pin_config_t i2s_pins = {
    .bck_io_num = 14,
    .ws_io_num = 33,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = 26};
*/
I2SSampler *i2s_sampler = NULL;

// Task to write samples to our server
void i2sWriterTask(void *param)
{
  I2SSampler *sampler = (I2SSampler *)param;
  WiFiClient *wifi_client = new WiFiClient();
  HTTPClient *http_client = new HTTPClient();
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true)
  {
    // wait for some samples to save
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      Serial.println("Sending data");
      http_client->begin(*wifi_client, I2S_SERVER_URL);
      http_client->addHeader("content-type", "application/octet-stream");
      http_client->POST((uint8_t *)sampler->getCapturedAudioBuffer(), sampler->getBufferSizeInBytes());
      http_client->end();
      Serial.println("Sent Data");
    }
  }
}

// Ping task - pings the server every 5 seconds so we know the connection is good
void pingTask(void *param)
{
  WiFiClient *wifi_client_ping = new WiFiClient();
  HTTPClient *http_client_ping = new HTTPClient();
  ;
  const TickType_t wait_time = pdMS_TO_TICKS(5000);
  while (true)
  {
    Serial.println("Ping Server Start");
    http_client_ping->begin(*wifi_client_ping, PING_URL);
    http_client_ping->GET();
    http_client_ping->end();
    Serial.println("Ping Server Done");
    vTaskDelay(wait_time);
          pinMode(2,OUTPUT);
  digitalWrite(2,1);
  }
}

void setup()
{

  Serial.begin(115200);

   Wire.begin();

  if (codec.begin() == false) //Begin communication over I2C
  {
    Serial.println("The device did not respond. Please check wiring.");
      pinMode(2,OUTPUT);
  digitalWrite(2,1);
    
    while (1); // Freeze
  }

  codec_setup();

  
  // launch WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);
  if (WiFi.waitForConnectResult() != WL_CONNECTED)
  {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }
  Serial.println("Started up");

  // set up i2s to read from our microphone
  i2s_sampler = new I2SSampler();

  // set up the i2s sample writer task
  TaskHandle_t writer_task_handle;
  xTaskCreate(i2sWriterTask, "I2S Writer Task", 4096, i2s_sampler, 1, &writer_task_handle);

  // start sampling from i2s device
  i2s_sampler->start(I2S_NUM_1, i2s_pins, i2s_config, 32768, writer_task_handle);

  // set up the ping task
  TaskHandle_t ping_task_handle;
  xTaskCreate(pingTask, "Ping Task", 4096, nullptr, 1, &ping_task_handle);
}


void codec_setup()
{
  // General setup needed
  codec.enableVREF();
  codec.enableVMID();

  // Setup signal flow to the ADC

  codec.enableLMIC();
  codec.enableRMIC();

  // Connect from INPUT1 to "n" (aka inverting) inputs of PGAs.
  codec.connectLMN1();
  codec.connectRMN1();

  // Disable mutes on PGA inputs (aka INTPUT1)
  codec.disableLINMUTE();
  codec.disableRINMUTE();

  // Set pga volumes
  codec.setLINVOLDB(6.00); // Valid options are -17.25dB to +30dB (0.75dB steps)
  codec.setRINVOLDB(6.00); // Valid options are -17.25dB to +30dB (0.75dB steps)

  // Set input boosts to get inputs 1 to the boost mixers
  codec.setLMICBOOST(WM8960_MIC_BOOST_GAIN_13DB);
  codec.setRMICBOOST(WM8960_MIC_BOOST_GAIN_13DB);

  // Connect from MIC inputs (aka pga output) to boost mixers
  codec.connectLMIC2B();
  codec.connectRMIC2B();

  // Enable boost mixers
  codec.enableAINL();
  codec.enableAINR();

  // Connect LB2LO (booster to output mixer (analog bypass)
  codec.enableLB2LO();
  codec.enableRB2RO();

  // Disconnect from DAC outputs to output mixer
  codec.disableLD2LO();
  codec.disableRD2RO();

  // Set gainstage between booster mixer and output mixer
  codec.setLB2LOVOL(WM8960_OUTPUT_MIXER_GAIN_NEG_6DB); 
  codec.setRB2ROVOL(WM8960_OUTPUT_MIXER_GAIN_NEG_6DB); 

  // Enable output mixers
  codec.enableLOMIX();
  codec.enableROMIX();

  // CLOCK STUFF, These settings will get you 44.1KHz sample rate, and class-d 
  // freq at 705.6kHz
  codec.enablePLL(); // Needed for class-d amp clock
  codec.setPLLPRESCALE(WM8960_PLLPRESCALE_DIV_2);
  codec.setSMD(WM8960_PLL_MODE_FRACTIONAL);
  codec.setCLKSEL(WM8960_CLKSEL_PLL);
  codec.setSYSCLKDIV(WM8960_SYSCLK_DIV_BY_2);
  codec.setBCLKDIV(4);
  codec.setDCLKDIV(WM8960_DCLKDIV_16);
  codec.setPLLN(7);
  codec.setPLLK(0x86, 0xC2, 0x26); // PLLK=86C226h
  //codec.setADCDIV(0); // Default is 000 (what we need for 44.1KHz)
  //codec.setDACDIV(0); // Default is 000 (what we need for 44.1KHz)
  codec.setWL(WM8960_WL_16BIT);

  codec.enablePeripheralMode();
  //codec.enableMasterMode();
  //codec.setALRCGPIO(); // Note, should not be changed while ADC is enabled.

  // Enable ADCs, and disable DACs
  codec.enableAdcLeft();
  codec.enableAdcRight();
  codec.disableDacLeft();
  codec.disableDacRight();
  codec.disableDacMute();

  //codec.enableLoopBack(); // Loopback sends ADC data directly into DAC
  codec.disableLoopBack();

  // Default is "soft mute" on, so we must disable mute to make channels active
  codec.enableDacMute(); 

  codec.enableHeadphones();
  codec.enableOUT3MIX(); // Provides VMID as buffer for headphone ground

  codec.setHeadphoneVolumeDB(0.00);
}

void loop()
{
  // nothing to do here - it's all driven by the i2s peripheral reading samples
}