int solenoidPin = 4;                    //This is the output pin on the Arduino
int activate;

void setup() 
{
  
  pinMode(solenoidPin, OUTPUT);
  Serial.begin(9600);          
}

void loop() 
{


while (Serial.available()>0)  { // Check if values are available in the Serial Buffer
  
  activate= Serial.parseInt();  
  Serial.println(activate);
  
  if (activate==1){
    Serial.println("activating");
    digitalWrite(solenoidPin, HIGH);      //Switch Solenoid ON
    delay(300);                          //Wait 1 Second
    digitalWrite(solenoidPin, LOW);       //Switch Solenoid OFF
    delay(10000); 

  }
}

}