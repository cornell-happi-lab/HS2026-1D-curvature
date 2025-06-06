void setup() {
  // Start serial communication at 9600 baud
  Serial.begin(9600);
}

void loop() {
  // Read analog value from pin A1
  int value = analogRead(A0);
  
  // Print the value to Serial Monitor or Plotter
  Serial.println(value);
  delay(5);
}
