/* Arduino Multiple Stepper Control Using The Serial Monitor
 
Created by Yvan / https://Brainy-Bits.com

This code is in the public domain...

You can: copy it, use it, modify it, share it or just plain ignore it!
Thx!

*/
#define topSwitch 10
#define bottomSwitch 12
#define leftSwitch 11
#define rightSwitch 13

#include "AccelStepper.h" 
// Library created by Mike McCauley at http://www.airspayce.com/mikem/arduino/AccelStepper/

// AccelStepper Setup
AccelStepper stepperX(1, 5, 2);   // 1 = Easy Driver interface
                                  // UNO Pin 2 connected to STEP pin of Easy Driver
                                  // UNO Pin 3 connected to DIR pin of Easy Driver
                                  
AccelStepper stepperZ(1, 6, 3);   // 1 = Easy Driver interface
                                  // UNO Pin 5 connected to STEP pin of Easy Driver
                                  // UNO Pin 6 connected to DIR pin of Easy Driver

// Stepper Travel Variables
long TravelX;  // Used to store the X value entered in the Serial Monitor
long TravelZ;  // Used to store the Z value entered in the Serial Monitor

int move_finished=1;  // Used to check if move is completed


void setup() {
  
  Serial.begin(9600);  // Start the Serial monitor with speed of 9600 Bauds    
  pinMode(topSwitch, INPUT_PULLUP);
  pinMode(bottomSwitch, INPUT_PULLUP);
  pinMode(leftSwitch, INPUT_PULLUP);
  pinMode(rightSwitch, INPUT_PULLUP);

// Print out Instructions on the Serial Monitor at Start
  Serial.println("Enter Travel distance seperated by a comma: X,Z ");
  Serial.print("Enter Move Values Now: ");

//  Set Max Speed and Acceleration of each Steppers
  stepperX.setMaxSpeed(100.0);      // Set Max Speed of X axis
  stepperX.setAcceleration(150.0);  // Acceleration of X axis

  stepperZ.setMaxSpeed(100.0);      // Set Max Speed of Y axis slower for rotation
  stepperZ.setAcceleration(150.0);  // Acceleration of Y axis

}


void loop() {

while (Serial.available()>0)  { // Check if values are available in the Serial Buffer

  move_finished=0;  // Set variable for checking move of the Steppers
  
  TravelX= Serial.parseInt();  // Put First numeric value from buffer in TravelX variable
  Serial.print(TravelX);
  Serial.print(" X Travel , ");
  
  TravelZ= Serial.parseInt();  // Put Second numeric value from buffer in TravelZ variable
  Serial.print(TravelZ);  
  Serial.println(" Y Travel ");
  
  stepperX.moveTo(TravelX);  // Set new move position for X Stepper
  stepperZ.moveTo(TravelZ);  // Set new move position for Z Stepper
  
  delay(1000);  // Wait 1 seconds before moving the Steppers
  Serial.print("Moving Steppers into position...");
  }

// Check if the Steppers have reached desired position
  if ((stepperX.distanceToGo() != 0) || (stepperZ.distanceToGo() !=0)) {
    if (digitalRead(topSwitch) && digitalRead(bottomSwitch) && digitalRead(leftSwitch) && digitalRead(rightSwitch)){  
      stepperX.run();  // Move Stepper X into position
      stepperZ.run();  // Move Stepper Z into position
    }
    
  }

// If move is completed display message on Serial Monitor
  if ((move_finished == 0) && (stepperX.distanceToGo() == 0) && (stepperZ.distanceToGo() == 0)) {
    Serial.println("COMPLETED!");
    Serial.println("");
    Serial.println("Enter Next Move Values (0,0 for reset): ");  // Get ready for new Serial monitor values
    move_finished=1;  // Reset move variable
  }
}
