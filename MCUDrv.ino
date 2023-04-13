#include<Servo.h>

Servo myservos[6];
char cmdbuf[128];
unsigned int pos[6];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  print("MCU start up");
  for(int i=0;i<6;i++){
    myservos[i].attach(i+2);
  }
}

void loop() {
    scanf("%d|%d|%d|%d|%d|%d",cmdbuf,&pos[0],&pos[1],&pos[2],&pos[3],&pos[4],&pos[5]);
    for(int i=0;i<6;i++){
      myservos[i].write(pos[i]);
      delay(5);
    }
}
