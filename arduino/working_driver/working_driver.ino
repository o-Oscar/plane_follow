#include <SoftwareSerial.h>

// #define DEBUG 1

SoftwareSerial rs232 =  SoftwareSerial(13, 22, true);

# define MAX_RESP_WAIT_DELAY 10
# define MAX_RESP_SIZE 16
uint8_t resp_buffer [MAX_RESP_SIZE];
uint8_t resp_len = 0;

bool wait_rs232_response () {
  resp_len = 0;
  for (int i = 0; i < MAX_RESP_WAIT_DELAY; i++) {
    delay(1);
    while (rs232.available()) {
      resp_buffer[resp_len] = rs232.read();
      #ifdef DEBUG
      Serial.println(resp_buffer[resp_len], HEX);
      #endif
      resp_len = (resp_len + 1) % MAX_RESP_SIZE;
      if (resp_buffer[resp_len-1] == 0xFF) {
        return true;
      }
    }
  }
  return false;
}

bool adress_set () {
  #ifdef DEBUG
  Serial.println("adress_set");
  #endif

  rs232.write(0x88);
  rs232.write(0x30);
  rs232.write(0x01);
  rs232.write(0xFF);
  
  return wait_rs232_response();
}

bool if_clear () {
  #ifdef DEBUG
  Serial.println("if_clear");
  #endif

  rs232.write(0x81);
  rs232.write(0x01);
  rs232.write((uint8_t) 0x00);
  rs232.write(0x01);
  rs232.write(0xFF);
  
  return wait_rs232_response();
}

bool home () {
  #ifdef DEBUG
  Serial.println("home");
  #endif

  rs232.write(0x81);
  rs232.write(0x01);
  rs232.write(0x06);
  rs232.write(0x04);
  rs232.write(0xFF);
  
  return wait_rs232_response();
}

bool move_right () {
  #ifdef DEBUG
  Serial.println("move_right");
  #endif

  rs232.write(0x81);
  rs232.write(0x01);
  rs232.write(0x06);
  rs232.write(0x01);
  rs232.write(0x08);
  rs232.write(0x08);
  rs232.write(0x02);
  rs232.write(0x03);
  rs232.write(0xFF);
  
  return wait_rs232_response();
}

bool move_left () {
  #ifdef DEBUG
  Serial.println("move_left");
  #endif

  rs232.write(0x81);
  rs232.write(0x01);
  rs232.write(0x06);
  rs232.write(0x01);
  rs232.write(0x08);
  rs232.write(0x08);
  rs232.write(0x01);
  rs232.write(0x03);
  rs232.write(0xFF);
  
  return wait_rs232_response();
}

bool stop () {
  #ifdef DEBUG
  Serial.println("stop");
  #endif

  rs232.write(0x81);
  rs232.write(0x01);
  rs232.write(0x06);
  rs232.write(0x01);
  rs232.write(0x08);
  rs232.write(0x08);
  rs232.write(0x03);
  rs232.write(0x03);
  rs232.write(0xFF);
  
  return wait_rs232_response();
}

# define BUF_SIZE 1000 
bool buffer [BUF_SIZE];

void setup() {
  Serial.begin(38400);
  rs232.begin(38400);
  
  delay(10);
  Serial.println();
  Serial.println("-----------");

  // necessary start cmds
  adress_set();
  if_clear();

  // debug movement
  move_right();
  delay(333);
  stop();
  delay(333);
  home();
  delay(1000);
  while (rs232.available()) { rs232.read(); }
}

void loop() {
  // replicate everything we see on 
  if (Serial.available()) {
    rs232.write(Serial.read());
  }
  if (rs232.available()) {
    Serial.write(rs232.read());
  }
}

