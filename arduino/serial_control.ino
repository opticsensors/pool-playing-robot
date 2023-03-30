// This is very similar to Example 3 - Receive with start- and end-markers
//    in Serial Input Basics   http://forum.arduino.cc/index.php?topic=396450.0

#include <AccelStepper.h>

// Define the stepper motor and the pins that is connected to
AccelStepper stepper1(1, 5, 2); // (Typeof driver: with 2 pins, STEP, DIR)
AccelStepper stepper2(1, 6, 3);

const byte numChars = 64;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing


boolean newData = false;

byte ledPin = 13;   // the onboard LED

      // variables to hold the parsed data
int x = 0;
int y = 0;

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
    
    // every time new data comes, we sset the curretn position of the motors to 0 steps
    // this is because we will move the motors with visual servoing (error vector)
    stepper1.setCurrentPosition(0); 
    stepper2.setCurrentPosition(0);

    if (newData == true) {
        strcpy(tempChars, receivedChars);
            // this temporary copy is necessary to protect the original data
            //   because strtok() used in parseData() replaces the commas with \0
        parseData();
        replyToPython();
        
        stepper1.moveTo(x);
        stepper2.moveTo(y);
            // when we achieved the desired position, we exit the while loop
        while (stepper1.currentPosition() != x || stepper2.currentPosition() != y) {
            stepper1.run();  // Move or step the motor implementing accelerations and decelerations to achieve the target position. Non-blocking function
            stepper2.run();
    }
    }
}

//===============

void parseData() {      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index
    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    x = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    y = atoi(strtokIndx);     // convert this part to an integer
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
    Serial.print("<Motor stopped at: ");
    Serial.print("   ");
    Serial.print(x);
    Serial.print("   ");
    Serial.print(y);
    Serial.print("   ");
    Serial.print(millis());
    Serial.print('>');
        // change the state of the LED everytime a reply is sent
    digitalWrite(ledPin, ! digitalRead(ledPin));
    newData = false;
}

//===============
