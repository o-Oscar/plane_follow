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

uint8_t buf [MAX_RESP_SIZE];
bool found = false;
bool found_pos = false;
bool found_zoom = false;
bool found_40 = false;
bool found_50 = false;
int b_id = 0;

void wait_for_cmd_end () {
  found_40 = false;
  found_50 = false;
  for  (int t =0; t < 10 && (!found_40 || !found_50);) {
    if (rs232.available()) {
      uint8_t b = rs232.read();
      buf[b_id] = b;
      b_id++;
      if (b == 0xFF) {
        if ((buf[b_id-2] == 0x41 || buf[b_id-2] == 0x42) && buf[b_id-3] == 0x90) {
          found_40 = true;
        }
        if ((buf[b_id-2] == 0x51 || buf[b_id-2] == 0x52) && buf[b_id-3] == 0x90) {
          found_50 = true;
        }
        b_id = 0;
      } 
    }else {delay(1); t++;}
  }
}


void wait_for_pos_response () {
  found_pos = false;
  for  (int t =0; t < 10 && !found_pos;) {
    if (rs232.available()) {
      uint8_t b = rs232.read();
      buf[b_id] = b;
      b_id++;
      if (b == 0xFF) {
        if (buf[b_id-10] == 0x50 && buf[b_id-11] == 0x90) {
          found_pos = true;
        }
        b_id = 0;
      } 
    }else {delay(1); t++;}
  }
}

void wait_for_zoom_response () {
  found_zoom = false;
  for  (int t =0; t < 10 && !found_zoom;) {
    if (rs232.available()) {
      uint8_t b = rs232.read();
      buf[b_id] = b;
      b_id++;
      if (b == 0xFF) {
        if (buf[b_id-6] == 0x50 && buf[b_id-7] == 0x90) {
          found_zoom = true;
        }
        b_id = 0;
      } 
    }else {delay(1); t++;}
  }
}

int loop_id = 0;

void loop() {
  // replicate everything we see on 
  // if (Serial.available()) {
  //   rs232.write(Serial.read());
  // }
  // if (rs232.available()) {
  //   Serial.write(rs232.read());
  // }

  long start_time = millis();


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
    rs232.write(0x04);
    rs232.write((uint8_t) 0x00);
    rs232.write((uint8_t) 0x00);
    rs232.write((uint8_t) 0x00);
    rs232.write(0xFF);
    wait_for_cmd_end ();
  } else {
    // send speed_cmd
    rs232.write(0x81);
    rs232.write(0x01);
    rs232.write(0x06);
    rs232.write(0x01);
    rs232.write(0x04);
    rs232.write(0x04);
    // if ((millis() /1000)%2 == 0) {rs232.write(0x01); } else {rs232.write(0x02); } 
    rs232.write(0x03);
    rs232.write(0x03);
    rs232.write(0xFF);

    wait_for_cmd_end();
  }

  loop_id = loop_id+1;





  // for  (int t =0; t < 100;) {
  //   if (rs232.available()) {
  //     uint8_t b = rs232.read();
  //     buf[b_id] = b;
  //     Serial.print(b, HEX);
  //     Serial.print(" ");
  //     b_id++;
  //     if (b == 0xFF) {
  //       Serial.println();
  //       // if (buf[b_id-10] == 0x50 && buf[b_id-11] == 0x90) {
  //       //   found = true;
  //       //   Serial.println("found");
  //       // }
  //       b_id = 0;
  //     } 
  //   }else {delay(1); t++;}
  // }

  // Serial.println();


  // found = false;
  // b_id = 0;
  // for  (int t =0; t < 30 && !found;) {
  //   if (rs232.available()) {
  //     uint8_t b = rs232.read();
  //     buf[b_id] = b;
  //     Serial.print(b, HEX);
  //     Serial.print(" ");
  //     b_id++;
  //     if (b == 0xFF) {
  //       Serial.println();
  //       if (buf[b_id-6] == 0x50 && buf[b_id-7] == 0x90) {
  //         found = true;
  //         Serial.println("found");
  //       }
  //       b_id = 0;
  //     } 
  //   }else {delay(1); t++;}
  // }

  // Serial.println();

  // Serial.println();

  // print total time

  found_pos = false;
  found_zoom = false;


  // for  (int t =0; millis()-start_time < 33;) {
  //   if (rs232.available()) {
  //     uint8_t b = rs232.read();
  //     buf[b_id] = b;
  //     // Serial.print(b, HEX);
  //     // Serial.print(" ");
  //     b_id++;
  //     if (b == 0xFF) {
  //       // Serial.println();
  //       if (buf[b_id-10] == 0x50 && buf[b_id-11] == 0x90) {
  //         found_pos = true;
  //         Serial.println("found pos");
  //       }
  //       if (buf[b_id-6] == 0x50 && buf[b_id-7] == 0x90) {
  //         found_zoom = true;
  //         Serial.println("found zoom");
  //       }
  //       // if (buf[b_id-2] == 0x41 && buf[b_id-3] == 0x90) {
  //       //   found_pos = true;
  //       //   Serial.println("found pos");
  //       // }
  //       // if (buf[b_id-2] == 0x41 && buf[b_id-3] == 0x90) {
  //       //   found_zoom = true;
  //       //   Serial.println("found zoom");
  //       // }
  //       b_id = 0;
  //     } 
  //   }else {delay(1); t++;}
  // }

  // delay(1000);
  Serial.print("time : ");
  while (millis() -start_time < 38) {}
  Serial.println(millis() - start_time);

}

