int solenoidPin = 4;                    //This is the output pin on the Arduino

void setup() 
{
  pinMode(solenoidPin, OUTPUT);          //Sets that pin as an output
}

void loop() 
{
  digitalWrite(solenoidPin, HIGH);      //Switch Solenoid ON
  delay(250);                          //Wait 1 Second
  digitalWrite(solenoidPin, LOW);       //Switch Solenoid OFF
  delay(10000);                          //Wait 1 Second
}