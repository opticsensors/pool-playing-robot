// This is very similar to Example 3 - Receive with start- and end-markers
//    in Serial Input Basics   http://forum.arduino.cc/index.php?topic=396450.0

#include <AccelStepper.h>

//stepper control outputs
#define mot1StepPin 6
#define mot1DirPin 3
#define mot2StepPin 5
#define mot2DirPin 2
#define topSwitch 10
#define bottomSwitch 12
#define leftSwitch 11
#define rightSwitch 13

// Define the stepper motor and the pins that is connected to
AccelStepper stepper1(1, mot1StepPin, mot1DirPin); // (Typeof driver: with 2 pins, STEP, DIR)
AccelStepper stepper2(1, mot2StepPin, mot2DirPin);

int initial_homing1 = -1;
int initial_homing2 = -1;

int pos1 = 1500;
int pos2 = 1500;
int time = 0;

int maxTravel1=5000;
int maxTravel2=5000;

//===============

void setup() {

    Serial.begin(9600);

    pinMode(topSwitch, INPUT_PULLUP);
    pinMode(bottomSwitch, INPUT_PULLUP);
    pinMode(leftSwitch, INPUT_PULLUP);
    pinMode(rightSwitch, INPUT_PULLUP);

    delay(200);
    Serial.println("<Arduino is ready>");
    
    //homing();

}

//===============

void loop() {
    homing_v2();
    // every time new data comes, we set the current position of the motors to 0 steps
    // this is because we will move the motors with visual servoing (error vector)
    stepper1.setCurrentPosition(0); 
    stepper2.setCurrentPosition(0);

    stepper1.moveTo(pos1);
    stepper2.moveTo(pos2);
    // when we achieved the desired position, we exit the while loop
    while (stepper1.currentPosition() != pos1 || stepper2.currentPosition() != pos2 ) {
        // before running we make sure that switches are not pressed
        if (digitalRead(topSwitch) && digitalRead(bottomSwitch) && digitalRead(leftSwitch) && digitalRead(rightSwitch)){
            stepper1.run();  // Move or step the motor implementing accelerations and decelerations to achieve the target position. Non-blocking function
            stepper2.run();
        }

    }   

    pos1=-pos1;
    //pos2=-pos2;
    delay(3000);

}

//===============

void homing() {
    stepper1.setMaxSpeed(400); // Set maximum speed value for the stepper
    stepper1.setAcceleration(700); // Set acceleration value for the stepper
    stepper2.setMaxSpeed(400);
    stepper2.setAcceleration(700);

    while (digitalRead(leftSwitch)) {  // Make the Stepper move CCW until the switch is activated   
        stepper1.moveTo(initial_homing1);  // Set the position to move to
        stepper2.moveTo(initial_homing2);  // Set the position to move to
        initial_homing1--;  // Decrease by 1 for next move if needed
        initial_homing2--;  // Decrease by 1 for next move if needed
        stepper1.run();  // Start moving the stepper
        stepper2.run();  // Start moving the stepper
        delay(time);
    }
    Serial.println("left reached");
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    initial_homing1=1;
    initial_homing2=1;
    delay(200);

    //while (!digitalRead(leftSwitch)) { // Make the Stepper move CW until the switch is deactivated
    //    stepper1.moveTo(initial_homing1);  // Set the position to move to
    //    stepper2.moveTo(initial_homing2);  // Set the position to move to
    //    stepper1.run();
    //    stepper2.run();
    //    initial_homing1++;
    //    initial_homing2++;
        //delay(5);
    //}
    for (int i = 0; i <= 200; i++) {
        stepper1.moveTo(initial_homing1);  // Set the position to move to
        stepper2.moveTo(initial_homing2);  // Set the position to move to
        initial_homing1++;  // Decrease by 1 for next move if needed
        initial_homing2++;  // Decrease by 1 for next move if needed
        stepper1.run();  // Start moving the stepper
        stepper2.run();  // Start moving the stepper
        //Serial.println(i);
        delay(time);
    }
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    initial_homing1=1;
    initial_homing2=-1;
    delay(200);

    while (digitalRead(topSwitch)) {  // Make the Stepper move CCW until the switch is activated   
        stepper1.moveTo(initial_homing1);  // Set the position to move to
        stepper2.moveTo(initial_homing2);  // Set the position to move to
        initial_homing1++;  // Decrease by 1 for next move if needed
        initial_homing2--;  // Decrease by 1 for next move if needed
        stepper1.run();  // Start moving the stepper
        stepper2.run();  // Start moving the stepper
        delay(time);
    }
    Serial.println("top reached");
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    initial_homing1=-1;
    initial_homing2=1;
    delay(200);

    //while (!digitalRead(topSwitch)) { // Make the Stepper move CW until the switch is deactivated
    //    stepper1.moveTo(initial_homing1);  // Set the position to move to
    //    stepper2.moveTo(initial_homing2);  // Set the position to move to
    //    initial_homing1--;
    //    initial_homing2++;
    //    stepper1.run();
    //    stepper2.run();
        //delay(5);
    //}
    for (int i = 0; i <= 200; i++) {
        stepper1.moveTo(initial_homing1);  // Set the position to move to
        stepper2.moveTo(initial_homing2);  // Set the position to move to
        initial_homing1--;  // Decrease by 1 for next move if needed
        initial_homing2++;  // Decrease by 1 for next move if needed
        stepper1.run();  // Start moving the stepper
        stepper2.run();  // Start moving the stepper
        //Serial.println(i);
        delay(time);

    }
    stepper1.setMaxSpeed(600); // Set maximum speed value for the stepper
    stepper1.setAcceleration(800); // Set acceleration value for the stepper
    stepper2.setMaxSpeed(600);
    stepper2.setAcceleration(800);
    delay(200);
}

void homing_v2() {
    stepper1.setMaxSpeed(400); // Set maximum speed value for the stepper
    stepper1.setAcceleration(700); // Set acceleration value for the stepper
    stepper2.setMaxSpeed(400);
    stepper2.setAcceleration(700);

    stepper1.moveTo(-maxTravel1);  // Set the position to move to
    stepper2.moveTo(-maxTravel2);  // Set the position to move to
    while (digitalRead(leftSwitch)) {  // Make the Stepper move CCW until the switch is activated   
        stepper1.run();  // Start moving the stepper
        stepper2.run();  // Start moving the stepper
        //delay(time);
    }
    stepper1.stop(); // Stop as fast as possible
    stepper2.stop(); // Stop as fast as possible  
    Serial.println("left reached");
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    delay(200);

    stepper1.moveTo(200);  // Set the position to move to
    stepper2.moveTo(200);  // Set the position to move to
    while (stepper1.currentPosition() != 200 || stepper2.currentPosition() != 200) { // Make the Stepper move CW until the switch is deactivated
        stepper1.run();
        stepper2.run();
        //delay(5);
    }

    Serial.println("left backup");
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    delay(200);

    stepper1.moveTo(maxTravel1);  // Set the position to move to
    stepper2.moveTo(-maxTravel2);  // Set the position to move to
    while (digitalRead(topSwitch)) {  // Make the Stepper move CCW until the switch is activated   
        stepper1.run();  // Start moving the stepper
        stepper2.run();  // Start moving the stepper
        //delay(time);
    }
    stepper1.stop(); // Stop as fast as possible
    stepper2.stop(); // Stop as fast as possible  
    Serial.println("top reached");
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    delay(200);

    stepper1.moveTo(-200);  // Set the position to move to
    stepper2.moveTo(200);  // Set the position to move to
    while (stepper1.currentPosition() != -200 || stepper2.currentPosition() != 200) { // Make the Stepper move CW until the switch is deactivated
        stepper1.run();
        stepper2.run();
        //delay(5);
    }
 
    Serial.println("top backup");
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    delay(200);

    stepper1.setMaxSpeed(600); // Set maximum speed value for the stepper
    stepper1.setAcceleration(800); // Set acceleration value for the stepper
    stepper2.setMaxSpeed(600);
    stepper2.setAcceleration(800);
    delay(200);
}

