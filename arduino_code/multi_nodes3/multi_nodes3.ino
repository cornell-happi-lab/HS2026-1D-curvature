const int rowPins[4] = {5, 6, 7, 8};  // D5-D8
const int colPins[4] = {2, 3, 4, 9};  // D2-D4, D9

int sensor1[16];
int sensor2[16];
int sensor3[16];

void setup() {
  // Set row and column pins as outputs
  for (int i = 0; i < 4; i++) {
    pinMode(rowPins[i], OUTPUT);
    pinMode(colPins[i], OUTPUT);
  }
  Serial.begin(115200);
}

void loop() {
  // Loop over rows
  for (int row = 0; row < 4; row++) {
    // Set one row HIGH, others LOW
    for (int i = 0; i < 4; i++) {
      digitalWrite(rowPins[i], (i == row) ? HIGH : LOW);
    }

    // Loop over 16 column configurations
    for (int col = 0; col < 12; col++) {
      // Set column pins according to binary of `col`
      for (int j = 0; j < 4; j++) {
        digitalWrite(colPins[j], (col >> j) & 0x01); // takes the jth value of the col
      }
      // Read sensor value from A0
      int value = analogRead(A0);

      // Store in appropriate sensor array
      if (col < 4) {
        sensor1[row * 4 + col] = value;
      } else if (col < 8) {
        sensor2[row * 4 + (col - 4)] = value;
      } else if (col < 12) {
        sensor3[row * 4 + (col - 8)] = value;
      }
      // Ignore col >= 12 if only 3 arrays are used
    }
  }

  // Optional: print sensor values
//  Serial.println("Sensor1:");
  for (int i = 0; i < 16; i++) Serial.print(sensor1[i]), Serial.print(",");
//  Serial.println("\nSensor2:");
//  for (int i = 0; i < 16; i++) Serial.print(sensor2[i]), Serial.print(",");
//  Serial.println("\nSensor3:");
//  for (int i = 0; i < 16; i++) Serial.print(sensor3[i]), Serial.print(",");
  Serial.println();

  delay(10); // Small delay before next reading
}
