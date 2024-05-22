#include <SoftwareSerial.h>

// #define DEBUG 1

SoftwareSerial rs232 =  SoftwareSerial(13, 22, true);

# define MAX_RESP_WAIT_DELAY 10
# define MAX_RESP_SIZE 32

uint8_t resp_buffer [MAX_RESP_SIZE];
uint8_t last_pos [8];
uint8_t last_zoom [4];
bool found_any = false;
bool found_pos = false;
bool found_zoom = false;
bool found_40 = false;
bool found_50 = false;
int b_id = 0;
long wait_start;
uint8_t b;


void wait_any_response () {
  wait_start = millis();
  while  (millis() - wait_start < MAX_RESP_WAIT_DELAY) {
    if (rs232.available()) {
      b = rs232.read();
      resp_buffer[b_id] = b;
      // #ifdef DEBUG
        // Serial.print(b, HEX);
        // Serial.write(" ");
      // #endif
      b_id++;
      if (b == 0xFF) {
        // Serial.println();
        found_any = true;
        b_id = 0;
      } 
    }
  }
}

void wait_for_cmd_end () {
  found_40 = false;
  found_50 = false;
  wait_start = millis();
  while  (millis() - wait_start < MAX_RESP_WAIT_DELAY && (!found_40 || !found_50)) {
    if (rs232.available()) {
      b = rs232.read();
      resp_buffer[b_id] = b;
      b_id++;
      if (b == 0xFF) {
        if ((resp_buffer[b_id-2] == 0x41 || resp_buffer[b_id-2] == 0x42) && resp_buffer[b_id-3] == 0x90) {
          found_40 = true;
        }
        if ((resp_buffer[b_id-2] == 0x51 || resp_buffer[b_id-2] == 0x52) && resp_buffer[b_id-3] == 0x90) {
          found_50 = true;
        }
        b_id = 0;
      } 
    }
  }
}

void wait_for_pos_response () {
  found_pos = false;
  wait_start = millis();
  while  (millis() - wait_start < MAX_RESP_WAIT_DELAY && !found_pos) {
    if (rs232.available()) {
      b = rs232.read();
      resp_buffer[b_id] = b;
      b_id++;
      if (b == 0xFF) {
        if (resp_buffer[b_id-10] == 0x50 && resp_buffer[b_id-11] == 0x90) {
          found_pos = true;
          for (int i = 0; i < 8; i++) {
            last_pos[i] = resp_buffer[b_id-9+i];
          } 
        }
        b_id = 0;
      } 
    }
  }
}

void wait_for_zoom_response () {
  found_zoom = false;
  wait_start = millis();
  while  (millis() - wait_start < MAX_RESP_WAIT_DELAY && !found_zoom) {
    if (rs232.available()) {
      b = rs232.read();
      resp_buffer[b_id] = b;
      b_id++;
      if (b == 0xFF) {
        if (resp_buffer[b_id-6] == 0x50 && resp_buffer[b_id-7] == 0x90) {
          found_zoom = true;
          for (int i = 0; i < 4; i++) {
            last_zoom[i] = resp_buffer[b_id-5+i];
          }
        }
        b_id = 0;
      } 
    }
  }
}

# define MAX_CMD_SIZE 32
uint8_t pc_cmd_buffer [MAX_CMD_SIZE];
int cmd_b_id = 0;
int pc_cmd_len = 0;

void update_pc_cmd () {
  pc_cmd_len = 0;
  
  if (Serial.available()) {
    b = Serial.read();
    pc_cmd_buffer[cmd_b_id] = b;
    cmd_b_id++;
    if (b == 0xFF) {
      pc_cmd_len = cmd_b_id;
      cmd_b_id = 0;
    } 
  }
}


void adress_set () {
  #ifdef DEBUG
  Serial.println("adress_set");
  #endif

  rs232.write(0x88);
  rs232.write(0x30);
  rs232.write(0x01);
  rs232.write(0xFF);
  
  wait_any_response();
}

void if_clear () {
  #ifdef DEBUG
  Serial.println("if_clear");
  #endif

  rs232.write(0x81);
  rs232.write(0x01);
  rs232.write((uint8_t) 0x00);
  rs232.write(0x01);
  rs232.write(0xFF);
  
  wait_any_response();
}

void home () {
  #ifdef DEBUG
  Serial.println("home");
  #endif

  rs232.write(0x81);
  rs232.write(0x01);
  rs232.write(0x06);
  rs232.write(0x04);
  rs232.write(0xFF);
  
  wait_any_response();
}

void move_right () {
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
  
  wait_any_response();
}

void move_left () {
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
  
  wait_any_response();
}

void stop () {
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
  
  wait_any_response();
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
  delay(10);
  if_clear();
  delay(10);

  // debug movement
  move_right();
  delay(333);
  stop();
  delay(333);
  home();
  delay(1000);
  while (rs232.available()) { rs232.read(); }
}

int loop_id = 0;
void send_optimized_cmds () {
  // send_zoom_read cmd (waiting for 90 50 04 00 00 00 FF)
  rs232.write(0x81);
  rs232.write(0x09);
  rs232.write(0x04);
  rs232.write(0x47);
  rs232.write(0xFF);
  
  wait_for_zoom_response();

  // send pos_read_cmd (waiting for 90 50 00 00 00 00 00 00 00 00 FF)
  rs232.write(0x81);
  rs232.write(0x09);
  rs232.write(0x06);
  rs232.write(0x12);
  rs232.write(0xFF);
  
  wait_for_pos_response();

  if ((loop_id%2) == 0) { 
    //send zoom cmd
    rs232.write(0x81);
    rs232.write(0x01);
    rs232.write(0x04);
    rs232.write(0x47);
    // if ((millis() /1000)%2 == 0) {rs232.write(0x02); } else {rs232.write(0x06); } 
    // rs232.write(0x04);
    // rs232.write((uint8_t) 0x00);
    // rs232.write((uint8_t) 0x00);
    // rs232.write((uint8_t) 0x00);
    rs232.write(pc_cmd_buffer[5]);
    rs232.write(pc_cmd_buffer[6]);
    rs232.write(pc_cmd_buffer[7]);
    rs232.write(pc_cmd_buffer[8]);
    rs232.write(0xFF);
    wait_for_cmd_end ();
  } else {
    // send speed_cmd
    rs232.write(0x81);
    rs232.write(0x01);
    rs232.write(0x06);
    rs232.write(0x01);
    // rs232.write(0x04);
    // rs232.write(0x04);
    // if ((millis() /1000)%2 == 0) {rs232.write(0x01); } else {rs232.write(0x02); } 
    // rs232.write(0x03);
    // rs232.write(0x03);
    rs232.write(pc_cmd_buffer[1]);
    rs232.write(pc_cmd_buffer[2]);
    rs232.write(pc_cmd_buffer[3]);
    rs232.write(pc_cmd_buffer[4]);
    rs232.write(0xFF);

    wait_for_cmd_end();
  }

  loop_id = loop_id+1;

  // write the position
  Serial.write(0x90);
  Serial.write(0x50);
  for (int i = 0; i < 8; i++) {
    Serial.write(last_pos[i]);
  }
  Serial.write(0xFF);

  // write the zoom
  Serial.write(0x90);
  Serial.write(0x50);
  for (int i = 0; i < 4; i++) {
    Serial.write(last_zoom[i]);
  }
  Serial.write(0xFF);

}

void loop() {
  update_pc_cmd();

  if (pc_cmd_len == 10 && pc_cmd_buffer[0] == 0xAA) {
    send_optimized_cmds();
  }else if (pc_cmd_len > 0) {
    for (int i = 0; i < pc_cmd_len; i++) {
      rs232.write(pc_cmd_buffer[i]);
    }
  }else {
    if (rs232.available()) {
      Serial.write(rs232.read());
    }
  }
}

