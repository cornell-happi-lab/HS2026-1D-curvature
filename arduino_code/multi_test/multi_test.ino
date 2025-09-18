const int rowPins[4] = {2, 3, 4, 5};
const int colPins[4] = {6, 7, 8, 9};

int sensor1[4][4];
int sensor2[4][4];
int sensor3[4][4];

int rowsum[4];
int colsum[4];

void setup() {
  for (int i = 0; i < 4; i++) {
    pinMode(rowPins[i], OUTPUT);
    pinMode(colPins[i], OUTPUT);
  }
  Serial.begin(115200);
}

void loop() {
  for (int row = 0; row < 4; row++){
    for (int i = 0; i < 4; i++){
      digitalWrite(rowPins[i], (i == row) ? HIGH : LOW);
    }

    // SENSOR 1
    digitalWrite(colPins[0], LOW);
    digitalWrite(colPins[1], LOW);
    digitalWrite(colPins[2], LOW);
    digitalWrite(colPins[3], LOW);
    sensor1[row][0] = analogRead(A0);

    digitalWrite(colPins[0], LOW);
    digitalWrite(colPins[1], LOW);
    digitalWrite(colPins[2], LOW);
    digitalWrite(colPins[3], HIGH);
    sensor1[row][1] = analogRead(A0);

    digitalWrite(colPins[0], LOW);
    digitalWrite(colPins[1], LOW);
    digitalWrite(colPins[2], HIGH);
    digitalWrite(colPins[3], LOW);
    sensor1[row][2] = analogRead(A0);

    digitalWrite(colPins[0], LOW);
    digitalWrite(colPins[1], LOW);
    digitalWrite(colPins[2], HIGH);
    digitalWrite(colPins[3], HIGH);
    sensor1[row][3] = analogRead(A0);

//    // SENSOR 2
//    digitalWrite(colPins[0], LOW);
//    digitalWrite(colPins[1], HIGH);
//    digitalWrite(colPins[2], LOW);
//    digitalWrite(colPins[3], LOW);
//    sensor2[row][0] = analogRead(A0);
//
//    digitalWrite(colPins[0], LOW);
//    digitalWrite(colPins[1], HIGH);
//    digitalWrite(colPins[2], LOW);
//    digitalWrite(colPins[3], HIGH);
//    sensor2[row][1] = analogRead(A0);
//
//    digitalWrite(colPins[0], LOW);
//    digitalWrite(colPins[1], HIGH);
//    digitalWrite(colPins[2], HIGH);
//    digitalWrite(colPins[3], LOW);
//    sensor2[row][2] = analogRead(A0);
//
//    digitalWrite(colPins[0], LOW);
//    digitalWrite(colPins[1], HIGH);
//    digitalWrite(colPins[2], HIGH);
//    digitalWrite(colPins[3], HIGH);
//    sensor2[row][3] = analogRead(A0);

//    //SENSOR 3
//    digitalWrite(colPins[0], HIGH);
//    digitalWrite(colPins[1], LOW);
//    digitalWrite(colPins[2], LOW);
//    digitalWrite(colPins[3], LOW);
//    sensor3[row][0] = analogRead(A0);
//
//    digitalWrite(colPins[0], HIGH);
//    digitalWrite(colPins[1], LOW);
//    digitalWrite(colPins[2], LOW);
//    digitalWrite(colPins[3], HIGH);
//    sensor3[row][1] = analogRead(A0);
//
//    digitalWrite(colPins[0], HIGH);
//    digitalWrite(colPins[1], LOW);
//    digitalWrite(colPins[2], HIGH);
//    digitalWrite(colPins[3], LOW);
//    sensor3[row][2] = analogRead(A0);
//
//    digitalWrite(colPins[0], HIGH);
//    digitalWrite(colPins[1], LOW);
//    digitalWrite(colPins[2], HIGH);
//    digitalWrite(colPins[3], HIGH);
//    sensor3[row][3] = analogRead(A0);
    }


//    // Calculate row sumation and column sumation
//    for (int i = 0; i < 4; i ++){
//      rowsum[i] = 0;
//      colsum[i] = 0;
//      for (int j = 0; j < 4; j++)
//      {
//        rowsum[i] = rowsum[i] + sensor1[i][j];
//        colsum[i] = colsum[i] + sensor1[j][i];
//      }
//    }
    
    for (int i = 0; i < 4; i++){
      for (int j = 0; j < 4; j++){
        Serial.print(sensor1[i][j]);
        if (i == 3 && j == 3)
          Serial.println();
        else
          Serial.print(',');
      }
    }

//    for (int i = 0; i < 4; i++){
//      Serial.print(rowsum[i]);
//      if (i == 3)
//        Serial.println();
//      else
//        Serial.print(',');
//    }

//    for (int i = 0; i < 4; i++){
//      Serial.print(colsum[i]);
//      if (i == 3)
//        Serial.println();
//      else
//        Serial.print(',');
//    }
}
