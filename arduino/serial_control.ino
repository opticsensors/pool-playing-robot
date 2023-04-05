// This is very similar to Example 3 - Receive with start- and end-markers
//    in Serial Input Basics   http://forum.arduino.cc/index.php?topic=396450.0

#include <AccelStepper.h>

//stepper control outputs
#define mot1StepPin 5
#define mot1DirPin 2
#define mot2StepPin 6
#define mot2DirPin 3
#define motMS1Pin 7
#define motMS2Pin 8
#define motMS3Pin 9

// Define the stepper motor and the pins that is connected to
AccelStepper stepper1(1, mot1StepPin, mot1DirPin); // (Typeof driver: with 2 pins, STEP, DIR)
AccelStepper stepper2(1, mot2StepPin, mot2DirPin);

const byte numChars = 64;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing

boolean newData = false;

byte ledPin = 13;   // the onboard LED

// variables to hold the parsed data
int relative_position_stepper1 = 0;
int relative_position_stepper2 = 0;
int absolute_position_stepper1 = 0;
int absolute_position_stepper2 = 0;
int mode;
byte microSteps[6] = {
    B000, // full step
    B100, // half step
    B010, // 1/4 step
    B110, // 1/8 step
    B001, // 1/16 step
    B101, // 1/32 step
}

//===============

void setup() {
    stepper1.setMaxSpeed(500); // Set maximum speed value for the stepper
    stepper1.setAcceleration(250); // Set acceleration value for the stepper
    stepper2.setMaxSpeed(500);
    stepper2.setAcceleration(250);

    Serial.begin(9600);

    pinMode(ledPin, OUTPUT);
    digitalWrite(ledPin, HIGH);
    delay(200);
    digitalWrite(ledPin, LOW);
    delay(200);
    digitalWrite(ledPin, HIGH);

    Serial.println("<Arduino is ready>");
}

//===============

void loop() {
    recvWithStartEndMarkers();
    
    // every time new data comes, we set the current position of the motors to 0 steps
    // this is because we will move the motors with visual servoing (error vector)
    stepper1.setCurrentPosition(0); 
    stepper2.setCurrentPosition(0);

    if (newData == true) {
        strcpy(tempChars, receivedChars);
            // this temporary copy is necessary to protect the original data
            //   because strtok() used in parseData() replaces the commas with \0
        parseData();
        
        // step mode (includes microstepping option)
        if (mode != -1) {
            digitalWrite(motMS1Pin, bitRead(microSteps[mode]), 2)
            digitalWrite(motMS2Pin, bitRead(microSteps[mode]), 1)
            digitalWrite(motMS3Pin, bitRead(microSteps[mode]), 0)

            stepper1.moveTo(relative_position_stepper1);
            stepper2.moveTo(relative_position_stepper2);
                // when we achieved the desired position, we exit the while loop
            while (stepper1.currentPosition() != relative_position_stepper1 || stepper2.currentPosition() != relative_position_stepper2 ) {
                stepper1.run();  // Move or step the motor implementing accelerations and decelerations to achieve the target position. Non-blocking function
                stepper2.run();
            }
        }

        // calibration mode (find corners)
        else {
            
        }

    replyToPython();

    }
}

//===============

void parseData() {      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index
    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    mode = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    relative_position_stepper1 = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    relative_position_stepper2 = atoi(strtokIndx);     // convert this part to an integer

    // update absolute position
    absolute_position_stepper1=absolute_position_stepper1+relative_position_stepper1;
    absolute_position_stepper2=absolute_position_stepper2+relative_position_stepper2;
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
    Serial.print(relative_position_stepper1);
    Serial.print(",");
    Serial.print(relative_position_stepper2);
    Serial.print(",");
    Serial.print(absolute_position_stepper1);
    Serial.print(",");
    Serial.print(absolute_position_stepper2);
    Serial.print('>');
        // change the state of the LED everytime a reply is sent
    digitalWrite(ledPin, ! digitalRead(ledPin));
    newData = false;
}

//===============
