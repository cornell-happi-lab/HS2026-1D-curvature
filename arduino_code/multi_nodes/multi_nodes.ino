const int colpins[] = {4, 5, 6, 7}; //D4-D7
const int rowpins[] = {2, 3}; //D2-D3

int readings[4][4]; //4x4 matrix to store analog readings

void setup(){
  for (int i = 0; i < 4; i++)
  {
    pinMode(colpins[i], OUTPUT);
  }
  for (int i = 0; i < 2; i++)
  {
    pinMode(rowpins[i], OUTPUT);
  }
  Serial.begin(9600);
}

void loop(){
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      if (j != i)
        digitalWrite(colpins[j], HIGH);
    }
    digitalWrite(colpins[i], LOW);

    for (int combo = 0; combo < 4; combo++)
    {
      digitalWrite(rowpins[0], combo & 0x01);
      digitalWrite(rowpins[1], (combo >> 1) & 0x01);
      delay(5); // 1ms
      readings[i][combo] = analogRead(A0);
    }
  }

//  Serial.print(readings[0][0]);
//  Serial.println();

  //print matrix
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
}
