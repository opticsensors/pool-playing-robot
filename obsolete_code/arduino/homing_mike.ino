/* End stop calibration by Mike Cook
 *  two limit switched wired to micro switch
 *  with common wired to ground and normally closed contact to pin
*/


#define UPPER_LIMIT 8
#define LOWER_LIMIT 9
#define EN 10
#define RST 11
#define STEP 12
#define DIR 13
#define INTERFACE_TYPE 1

#include <AccelStepper.h>
byte controlPins[] = {EN, RST, STEP, DIR}; 
long maxTravel = 200000; // max distance you could be away from zero switch
long maxBackup = 200; // max distance to correct limit switch overshoot

// Create a new instance of the AccelStepper class:
AccelStepper stepper = AccelStepper(INTERFACE_TYPE, STEP, DIR);

void setup(){
  // Set limit switch inputs
  pinMode(UPPER_LIMIT, INPUT_PULLUP);
  pinMode(LOWER_LIMIT, INPUT_PULLUP);
  for(int i = 0; i<4; i++){
    pinMode(controlPins[i], OUTPUT);
    digitalWrite(controlPins[i], LOW);
  }
  delay(200); // hold reset until we can see it on the scope
  digitalWrite(RST, HIGH); // take out of reset
  delay(2); // allow charge pump on driver to start
  
  Serial.begin(9600);
  Serial.println("Stepper calibration");
  // Set the maximum speed and acceleration for calibration:
  stepper.setMaxSpeed(700); // note 1000 is the fastest speed for this library
  stepper.setAcceleration(500);
  findEnd(LOWER_LIMIT, 0); // hit the zero limit switch
  backup(LOWER_LIMIT, 1);  // nudge back so switch is not triggered
  // Set the maximum speed and acceleration for use:
  stepper.setCurrentPosition(0); // set this point as zero
  stepper.setMaxSpeed(700); // note 1000 is the fastest speed for this library
  stepper.setAcceleration(500);
  digitalWrite(EN, HIGH); // turn off motor drive, write a LOW to turn back on
}

void findEnd(byte limitSwitch, byte moveDirection){ 
  if(moveDirection == 0) maxTravel = maxTravel * -1; // make a negative value
  //Serial.print("moving to "); Serial.println(maxTravel); // uncomment for debug 
  stepper.moveTo(maxTravel);
  while(digitalRead(limitSwitch) == LOW) {
    stepper.run();
  }
    stepper.stop(); // Stop as fast as possible 
   // Serial.println("motor at low end with limit switch touched");  // uncomment for debug 
  }

void backup(byte limitSwitch, byte moveDirection){ // move off the end spot
  if(moveDirection != 0) maxBackup = maxBackup * -1; // make a negative value
  // Serial.print("Backup moving to "); Serial.println(maxBackup);  // uncomment for debug 
  stepper.moveTo(maxBackup);
  while(digitalRead(limitSwitch) == HIGH) {
    stepper.run();
  }
    stepper.stop(); // Stop as fast as possible 
    // Serial.println("motor at low end with limit switch not touched"); // uncomment for debug 
  }  

// move to a position with the limit switches monitored
boolean moveSafe(long moveToHere){ // returns true is limit switch tripped otherwise returns false
  boolean tripped = false;
  stepper.moveTo(moveToHere);
  while(stepper.distanceToGo() != 0 && (digitalRead(UPPER_LIMIT) == LOW) && (digitalRead(LOWER_LIMIT) == LOW)) {
    stepper.run();
  }
  if( (digitalRead(UPPER_LIMIT) == HIGH) || (digitalRead(LOWER_LIMIT) == HIGH)){
    stepper.stop();
    tripped = true;
  }
  return tripped;
}

void recover(){ // recover when a limit switch is triggered
  if(digitalRead(LOWER_LIMIT) == HIGH) backup(LOWER_LIMIT, 1);
  if(digitalRead(UPPER_LIMIT) == HIGH) backup(UPPER_LIMIT, 0);
}

/*
 // example of how to use the moveSafe function
 if(moveSafe(300L)) recover();
 */

void loop(){
 // digitalWrite(EN, LOW); // turn on motor drive,
 // your code here 
}