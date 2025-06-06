import serial
import time

arduino = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # Give the Arduino time to reset

print("Reading data from Arduino...")

while True:
    if arduino.in_waiting > 0:
        try:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            print("Analog Value:", line)
        except Exception as e:
            print("Error:", e)
