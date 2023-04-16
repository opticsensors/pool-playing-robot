// This is very similar to Example 3 - Receive with start- and end-markers
//    in Serial Input Basics   http://forum.arduino.cc/index.php?topic=396450.0

#include <AccelStepper.h>

//stepper control outputs
#define mot1StepPin 6
#define mot1DirPin 3
#define mot2StepPin 5
#define mot2DirPin 2
#define motMS1Pin 7
#define motMS2Pin 8
#define motMS3Pin 9
#define topSwitch 10
#define bottomSwitch 12
#define leftSwitch 11
#define rightSwitch 13

// Define the stepper motor and the pins that is connected to
AccelStepper stepper1(1, mot1StepPin, mot1DirPin); // (Typeof driver: with 2 pins, STEP, DIR)
AccelStepper stepper2(1, mot2StepPin, mot2DirPin);

// Define velocities and accelerations
int calibrationVelocity = 500;
int calibrationAcceleration = 250;
int workingVelocity = 300;
int workingAcceleration = 150;

// variables to store calibration positions
int reach_position1;
int reach_position2;

const byte numChars = 64;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing
int sign1 = 1;
int sign2 = 1;
int backup_position = 200;
int infinity = 32766;
int untilTheEnd1 = 0;
int untilTheEnd2 = 0;

boolean newData = false;

// variables to hold the parsed data
int absolute_position_stepper1 = 0;
int absolute_position_stepper2 = 0;
int mode = -1;
byte microSteps[6] = {
    B000, // full step
    B100, // half step
    B010, // 1/4 step
    B110, // 1/8 step
    B001, // 1/16 step
    B101, // 1/32 step
};

//===============

void setup() {

    Serial.begin(9600);

    // configure initial motor settings
    pinMode(motMS1Pin, OUTPUT);
    pinMode(motMS2Pin, OUTPUT);
    pinMode(motMS3Pin, OUTPUT);

    pinMode(topSwitch, INPUT_PULLUP);
    pinMode(bottomSwitch, INPUT_PULLUP);
    pinMode(leftSwitch, INPUT_PULLUP);
    pinMode(rightSwitch, INPUT_PULLUP);

    digitalWrite(motMS1Pin, LOW);
    digitalWrite(motMS2Pin, LOW);
    digitalWrite(motMS3Pin, LOW);

    delay(200);
    setSpeedAccel(workingVelocity, workingAcceleration);
    Serial.println("<Arduino is ready>");
   

}

//===============

void loop() {
    recvWithStartEndMarkers();
    
    // every time new data comes, we set the current position of the motors to 0 steps
    // this is because we will move the motors with visual servoing (error vector)
    // stepper1.setCurrentPosition(0); 
    // stepper2.setCurrentPosition(0);

    if (newData == true) {
        strcpy(tempChars, receivedChars);
            // this temporary copy is necessary to protect the original data
            //   because strtok() used in parseData() replaces the commas with \0
        parseData();
        
        // step mode (includes microstepping option)
        if (mode >= 0) {
            digitalWrite(motMS1Pin, bitRead(microSteps[mode], 2));
            digitalWrite(motMS2Pin, bitRead(microSteps[mode], 1));
            digitalWrite(motMS3Pin, bitRead(microSteps[mode], 0));
            moveSafe();
        }

        // homing
        else if (mode == -1) {
            delay(5000);
            digitalWrite(motMS1Pin, bitRead(microSteps[1], 2));
            digitalWrite(motMS2Pin, bitRead(microSteps[1], 1));
            digitalWrite(motMS3Pin, bitRead(microSteps[1], 0));
            setSpeedAccel(calibrationVelocity, calibrationAcceleration);
            homing();
            setSpeedAccel(workingVelocity, workingAcceleration);
        }

        // calibration mode (find corners)
        else if (mode == -2) {
            delay(5000);
            setSpeedAccel(calibrationVelocity, calibrationAcceleration);
            homing();
            findMaxPositions();
            delay(5000);
            homing();
            setSpeedAccel(workingVelocity, workingAcceleration);
        }

    replyToPython();

    }
}

//===============

void setSpeedAccel(int speed, int accel){
    stepper1.setMaxSpeed(speed); // Set maximum speed value for the stepper
    stepper1.setAcceleration(accel); // Set acceleration value for the stepper
    stepper2.setMaxSpeed(speed);
    stepper2.setAcceleration(accel);
}

void positionSigns(int pin){

    // when we touch a limit switch, we need to back off a little bit until 
    // it stops being pressed. For this reason, negative values on limit  
    // switched pins are considered  

    if (pin==leftSwitch || pin==-rightSwitch) {
        sign1=-1;  
        sign2=-1;  
    }
    else if (pin==rightSwitch || pin==-leftSwitch) {
        sign1=1;  
        sign2=1;
    }
    else if (pin==topSwitch || pin==-bottomSwitch) {
        sign1=1;  
        sign2=-1;
    }
    else if (pin==bottomSwitch || pin==-topSwitch) {
        sign1=-1;  
        sign2=1;
    }
}

void reachEnd(int pin) {
    
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    positionSigns( pin );
    stepper1.moveTo(sign1*infinity);  // Set the position to move to
    stepper2.moveTo(sign2*infinity);  // Set the position to move to
    while (digitalRead(pin)) {  // Make the Stepper move CCW until the switch is activated   
        stepper1.run();  // Start moving the stepper
        stepper2.run();  // Start moving the stepper
        untilTheEnd1=untilTheEnd1+sign1;
        untilTheEnd2=untilTheEnd2+sign2;

    }
    stepper1.stop(); // Stop as fast as possible
    stepper2.stop(); // Stop as fast as possible  

    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    positionSigns( -pin );
    delay(200);

    stepper1.moveTo(sign1*backup_position);  // Set the position to move to
    stepper2.moveTo(sign2*backup_position);  // Set the position to move to
    while (stepper1.currentPosition() != sign1*backup_position || stepper2.currentPosition() != sign2*backup_position) { // Make the Stepper move CW until the switch is deactivated
        stepper1.run();
        stepper2.run();
        untilTheEnd1=untilTheEnd1+sign1;
        untilTheEnd2=untilTheEnd2+sign2;
    }
    
    stepper1.setCurrentPosition(0);  // Set the current position as zero for now
    stepper2.setCurrentPosition(0);  // Set the current position as zero for now
    untilTheEnd1=0;
    untilTheEnd2=0;
    delay(200);

}

void homing() {

    reachEnd(leftSwitch);
    reachEnd(topSwitch);

    absolute_position_stepper1 = 0;
    absolute_position_stepper2 = 0;

}

void findMaxPositions() {

    homing();

    reachEnd(rightSwitch);
    absolute_position_stepper1=absolute_position_stepper1+untilTheEnd1;
    absolute_position_stepper2=absolute_position_stepper2+untilTheEnd2;

    reachEnd(bottomSwitch);
    absolute_position_stepper1=absolute_position_stepper1+untilTheEnd1;
    absolute_position_stepper2=absolute_position_stepper2+untilTheEnd2;
}

void moveSafe() {
    stepper1.moveTo(absolute_position_stepper1);
    stepper2.moveTo(absolute_position_stepper2);
    // when we achieved the desired position, we exit the while loop
    while (stepper1.currentPosition() != absolute_position_stepper1 || stepper2.currentPosition() != absolute_position_stepper2 ) {
        // before running we make sure that switches are not pressed
        if (digitalRead(topSwitch) && digitalRead(bottomSwitch) && digitalRead(leftSwitch) && digitalRead(rightSwitch)){
            stepper1.run();  // Move or step the motor implementing accelerations and decelerations to achieve the target position. Non-blocking function
            stepper2.run();
        }

        else{
            stepper1.stop(); // Stop as fast as possible
            stepper2.stop(); // Stop as fast as possible  
            mode = -1;
            break;
        }
    }
}

void parseData() {      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index
    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    mode = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    absolute_position_stepper1 = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    absolute_position_stepper2 = atoi(strtokIndx);     // convert this part to an integer
}

void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

//===============

void replyToPython() {

    Serial.print("<");
    Serial.print(mode);
    Serial.print(",");
    Serial.print(absolute_position_stepper1);
    Serial.print(",");
    Serial.print(absolute_position_stepper2);
    Serial.print('>');

        // change the state of the data bool everytime a reply is sent
    newData = false;
}

//===============
