## 1. Hardware
The tactile sensor is fabricated following a resistive-based design commonly used in tactile sensing research. All required materials — including sensor fabrication supplies, PCB components, cables, and connectors — are listed in [`orders`](orders.xlsx). 

We used KiCad for both the schematic design and PCB layout. The complete board files are available at [`/PCB/main_board`](/PCB/main_board/). The board was manufactured through JLCPCB, and the Gerber files uploaded for fabrication can be found at [`/PCB/gerbers`](/PCB/gerbers/gerber.zip).

An Arduino code [`/arduino_code/multi_test/multi_test.ino`](/arduino_code/multi_test/multi_test.ino) is used to read the sensor data, while a Python script receives the readings through the serial port and performs post-processing.

## 2. Curvature Dataset and Model Training
The sensor can estimate the mean curvature of the surface it is attached to. This is achieved by collecting baseline readings (no force applied) on cylindrical surfaces with different curvatures and training a residual MLP to map these inputs to a single curvature value.

For general 3D surfaces, the model predicts the mean curvature:
$$
\kappa = \tfrac{1}{2}(k_1 + k_2)
$$
For cylindrical surfaces, this simplifies to:
$$
\kappa = \tfrac{1}{R} \quad \text{since } k_2 = 0
$$

### 2.1 3D Print Fixtures
3D print the cylindrical fixtures with mean curvatures of 0, 10, 20, 30, 40, 50, 60, 70, and 80 m⁻¹. The STEP files are located in [`/3D_print/MLPdataset`](/3D_print/MLPdataset). The radii for each fixture are calculated in [`MLPdataset.xlsx`](/3D_print/MLPdataset/MLPdataset.xlsx).

### 2.2 Collect Baseline Data
After the parts have been printed, attach the sensor onto a fixture and navigate to [`/MLPtraining`](/MLPtraining). Run the following script to collect baseline readings:

```bash
python baseline.py
```
This script collects 100 baseline readings from the sensor and averages them. Save the averaged data to [`/MLPtraining/sensor_data.csv`](/MLPtraining/sensor_data.csv). Then rotate and/or flip the sensor, and repeat this process 10 times for each curvature fixture.

### 2.3 Train the Model
Once all data are collected, run:
```bash
python  train_strong.py
```
This script trains the residual MLP using the collected data and generates a .pt file.

### 2.4 Validate Curvature Prediction
To validate the trained model and test real-time curvature prediction, run
```bash
python realtime_prediction_strong_save.py
```

### 2.5 Pre-Collected Dataset and Model
The dataset used in our paper is available at [`/MLPtraining/sensor_data.csv`](/MLPtraining/sensor_data.csv), and the trained model can be found at [`/MLPtraining/runs_strong/best.pt`](/MLPtraining/runs_strong/best.pt)

## 3. Force Calibration
### 3.1 Hardware Setup
An experimental setup was built for force calibration. The stepper motor and motor controller used in the setup are listed in [`orders.xlsx`](/orders.xlsx). Fixtures with different curvatures were 3D printed using resin, and all corresponding `.stl` files can be found in [`/3D_print/fixture_resin`](/3D_print/fixture_resin/). An [**ATI Nano 17**](https://www.ati-ia.com/products/ft/ft_models.aspx?id=Nano17) force/torque sensor was used as the ground truth reference.  
The 3D-printed mounting case for the FT sensor is available at [`/3D_print/sensor_case`](/3D_print/sensor_case/).

### 3.2 Data Collection
During data collection, the sensor was attached to fixtures with different curvatures. The tip mounted on the ATI Nano 17 force/torque sensor pressed the four central sensing nodes (S5, S6, S9, S10) together as a block, since variations within each block were small and stable. A stepper motor moved incrementally, and at each step, the sensor readings were recorded using the Python script:
```bash
python calibration_static_multinodes.py
```
After data collection, run:
```bash
python plot_static.py
```
to visualize all recorded data under differnt curvatures.

Finally, execute:
```bash
python model_predictNN.py
```
to generate the curvature-aware calibration model. The model is constrained to ensure that all predicted forces are non-negative.

## 4. Validation
We embedded the sensor into five daily objects and developed a GUI to collect static sensing data while a person grasped each object.

To run the GUI, execute the code:
```bash
python mygui_sensor.py
```
During testing, the script automatically records data during periods when the human grasping force matches the preset reference force. We tested four reference forces: 2 N, 4 N, 6 N, and 8 N. All collected data are stored in [`/validation/daily_objects`](/validation/daily_objects/).

To validate sensor's reading under human's natural holding force, run:
```bash
python mygui_sensor_natural.py
```
After collecting all data, calculate the average and standard deviation using:
```bash
python cal_avg_stddev.py
```
All our processed results are stored in ['/validation/validation_data.xlsx'](/validation/validation_data.xlsx).