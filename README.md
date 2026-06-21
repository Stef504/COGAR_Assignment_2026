# Subgroup C1: Tactile Data Analysis

## Assignment 6: Multi-Modal Tactile Dataset Creation (REAL-SENSOR)

What to do: Systematically collect and analyze data from Daimon sensor's 4 modalities
1) Design experiments to collect depth, image, deformation, shear data
2) Test sensor on 20+ materials/objects
3) Analyze correlation between different sensor modalities
4) Create comprehensive tactile material database
Software needed: ROS2, Python data analysis (pandas, numpy, matplotlib)
Research needed: Tactile sensing papers, material property analysis
Deliverables: Multi-modal tactile dataset, sensor characterization report

# Interface for Daimon tactile sensor (DM-Tac WS) Vision-based Tactile Sensor using the `original_main.py` script

# How to use

## Work with Python 3.8/3.9/3.10/3.11. Make sure you have cuda toolkit 12.x installed, otherwise you might need to modify setup.py

## Install the package
    pip install .

## Plug in the sensor
    Note that the sensor works best with Linux.

## Run
    orginal_main.py

# Baxter:
Fork and use -> https://github.com/giangalv/baxter_rosbridge_adapter, follow the README. 


## How to execute Baxter, IK, Daimon Sensor and Position Kinematics 

# Executing Baxter:
- To set up the experiment and utilize Baxter with the attached sensor, these commands and scripts need to be executed.
- Ensure Baxter is turned on and plugged into the PC

This allows us to visualise Baxter in RVIZ and enables the connection:
```bash
ros2 launch baxter_rosbridge_adapter baxter_visualization.launch.py 
```

This allows us to control the joint commands:
```bash
ros2 run baxter_rosbridge_adapter baxter_cli
```
To view what Baxter is cable of and what information we can obtain it was useful to view the `baxter_common_ros2` package. 

Prior to any motion it is advised to calibrate the arm of choice:
```bash
ros2 run baxter_rosbridge_adapter calibrate_arm -l {left/right}
```


## Useful Observations to consider when running Baxter:
- Baxters joints have elastic actuators; therefore, accuracy in its motion cannot be guaranteed and needs to be considered. 
- Due to its joint mechanism when commencing the y-axis motion, the arm fluctuated in the beginning and end of each repetition, and the motion was not conducted along a straight line. This curved motion resulted in varying pressure applied, which proved useful for the experiments.
- Due to the socket connection, the readings taken from `Position Kinematics` vary, so trial and error needs to be conducted when setting the initial position with respect to the tested material. 
- To obtain better sensor results, it is advised to start with the sensor either hovering or placed on the object. This resulted in a smoother sliding action.
- Adjustments found useful for the z-axis from position kinematics were to adjust with a +/-0.0025 m.
- `LEFT_METAL_POSITION` was added to `baxter_cli.py "to assist in a desired arm configuration that resulted in a better y-axis motion.

# Position Kinematics:
- This allows the user to find the Cartesian co-ordinates of the end-effector with respect to the base. We feed this value into the inverse kinematics node to move the robot along the x-direction for a slide. 
- This uses Baxter's built-in forward kinematics toolkit.
- The user can choose which limb to analyse. 
 
# Code: 
```bash
ros2 run baxter_ik position_kinematics -l {right/left}
```

# Inverse Kinematics:
- Using Baxter's built-in commands and computations, we can solve for inverse kinematics and perform the desired motion.
- This node allows us to choose the desired limb to move.
- Moves along the y-axis by a user-chosen amount for a user-chosen duration, and it breaks in between each swipe. This break stops and starts the local Daimon sensor reading. 
- These local Daimon sensor readings are required for the neural network (NN).
- The duration chosen by the user allows for a slower/faster swiping motion.
- For this experiment a duration of 2 seconds was chosen.
  
# Code: 
```bash
ros2 run baxter_ik ik_baxter -l {right.left}
```

## Daimon Sensor
- With each swipe the sensor is activated and records the respective arrays of data (depth, shear, shear velocity (derivative of shear), and normal velocity (derivative of depth)). These sets are then fed into the NN. 
- For accurate representation of the active contact zones, the depth and shear maps/graphs were isolated to focus only on the active area to avoid any readings of noise.
- Bridge of used data from the daimon sensor:
  - getDepth-- (240, 320) represents the normal deformation, elastic deformation of the elastomer 
  - getShear-- (240, 320, 2) tangential deformation information of each pixel in mm (parallel to contact surface)
- The analysis of these graphs and their time derivatives allows us to distinguish between each material, which mainly focuses on identifying the differences between surfaces roughness and material hardness. 
- A helper function was created `Utilities` to stack into a multivariate time-series matrix (shear, depth, shear velocity, and normal velocity) for the neural network.
- After an entire experiment of repetitions has been completed, a graph is generated with the respective data. This allows us to visualise whether the data obtained is ideal. 
- Data gathered from each iteration is saved in the respective material name, and orientation.
- The experiment should be run before the `ik_baxter` script to properly record the data.

# Code: 
```bash
ros2 run baxter_ik daimon_sensor
```

# Neural Network:
- Uses the time series transformer neural networking technique to learn from the variables (shear, depth, shear velocity, and depth velocity)
- It assigns weights to the data and receives penalties (for its incorrect assumptions, once compared with the actual tensor).
- The scale taken from the datasets was normalized as to account for the varying scale.
- This is executed using the recorded data for each iterations  
- A confusion matrix and classification report is produced. The confusion matrix maps the true labels against the predicted labels, this way we can anaylse which catergories are being confsued with other catergories, this would be a result of similar patterns
- Classification report leads to the stastical anaylses . the statics are groups by precision, recall and F1-score. Precision: how well or poorly the model was we naming a material out of all the materials. High precision resulst in a low false-positive. Recall : Out of all the samples of a specific material present in the test , how succeful was the model per material. High recall indicates a low false-negative rate. F1-Score: 2* (Precision * Recall)/(precision+recall). balance metrix representing performance for that specific class. So how trustworthy is the systen and how thorough
- System is based on multi-feature Vision Based Tactile System 
- A classification report, confusion matrix and the trained model is saved in `Saved_Models`

# Live Classification:
- 

# File Location:
```bash
|
+-- baxter_ros2_ws
|   +-- baxter_common_ros2/       
|
|   +-- baxter_dataflow/      
|
|   +-- baxter_ik/
|        +--ik_baxter.py                inverse kinematics
|        +--position_kinematics.py      forward kinematics
|        +-- daimon_sensor.py           daimon sensor        
|        +--utilities.py                assiting function
|        
|    +-- baxter_interface/
|    
|    +-- baxter_rosbridge_adapter/
|        +--baxter_cli.py
|        +--calibrate_arm.py
|        +--baxter_grippers_cli.py
|        +--joint_state_bridge.py
|
+-- Dataset
|   +--Metal
|       +--solid_metal_part
|          +--down
|           +--left
|           +--right
|           +--up
|   +--Plastic
|       +--3D-printed-part
|           +--down
|           +--left
|           +--right
|           +--up
|   +--Rubber
|       +--rubber-ball
|           +--down
|           +--left
|           +--right
|           +--up
|
+--dmrobotics
|
+--dmrobotics.egg-info
|
+-- plots               The generated graphs
|
+-- Saved_Models        The saved trained models (.pth), confusion matrix, classification report
|
+-- DM-Tac W- English.pdf
|
+-- DM-Tac.pdf
|
+-- graphs.py           Display graphs
|
+-- live_classifier.py
|
+-- main.py             Hand testing- prior to any ros2 development
|
+-- neural_network.py   Neural Network script
|
+-- original_main.py    Original Daimon Sensor script, without any modifications
|
+-- Report and Presentation.md
|
+-- requirements.txt    Packages required
|
+-- setup.py            Used to install packges
|
+-- utilities.py        Helper Function
```


