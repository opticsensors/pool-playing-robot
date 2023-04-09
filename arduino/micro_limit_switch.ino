#define topSwitch 10
#define bottomSwitch 11
#define leftSwitch 12
#define rightSwitch 13

void setup() {

    Serial.begin(9600);

    pinMode(topSwitch, INPUT_PULLUP);
    pinMode(bottomSwitch, INPUT_PULLUP);
    pinMode(leftSwitch, INPUT_PULLUP);
    pinMode(rightSwitch, INPUT_PULLUP);
}

void loop() {

    while (digitalRead(leftSwitch)) {  // print until the switch is activated   
        Serial.print("switch is not pressed");
        delay(5);
    }

    while (!digitalRead(leftSwitch)) { // print until the switch is deactivated
        Serial.print("switch is pressed");
        delay(5);
    }
}