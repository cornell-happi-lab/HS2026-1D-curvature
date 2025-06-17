const int rowpins[] = {2, 3, 4, 5}; // D2-D5
const int colpins[] = {A0, A1, A2, A3}; // A0-A3

int readings[4][4];

void setup(){
  for (int i = 0; i < 4; i++)
  {
    pinMode(rowpins[i], OUTPUT);
  }
  Serial.begin(115200);
}

void loop(){
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      if (j != i)
        digitalWrite(rowpins[j], LOW);
    }
    digitalWrite(rowpins[i], HIGH);
    delay(2);  // allow signal to stabilize
    for (int k = 0; k < 4; k++)
    {
      readings[i][k] = analogRead(colpins[k]);
    }
  }

  // Sending Data
   for (int i = 0; i < 4; i++)
   {
    for (int j = 0; j < 4; j++)
    {
      Serial.print(readings[i][j]);
      if (i == 3 && j == 3)
        Serial.println(); // end of line
      else
        Serial.print(','); // tab-separated
    }
   }
//  Serial.println(readings[3][3]);
}

//// test code
//int reading;
//void setup(){
//  pinMode(2, OUTPUT);
//  pinMode(3, OUTPUT);
//  pinMode(4, OUTPUT);
//  pinMode(5, OUTPUT);
//
//  Serial.begin(9600);
//}
//
//void loop(){
//  digitalWrite(2, HIGH);
//  digitalWrite(3, LOW);
//  digitalWrite(4, LOW);
//  digitalWrite(5, LOW);
//
//  reading = analogRead(A0);
//  Serial.println(reading);
//  delay(1);
//}
