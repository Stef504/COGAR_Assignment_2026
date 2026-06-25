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

Due to limited time and available data in the understanding of the capabilities of the Daimon sensor, only two modalities were used. A new project layout is formed with the following changes:
1) Design experiments to collect depth and shear data.
2) Test sensor on 3 materials/objects


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


# How to execute Baxter, IK, Daimon Sensor and Position Kinematics 

## Executing Baxter:
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

## Position Kinematics:
- This allows the user to find the Cartesian co-ordinates of the end-effector with respect to the base. We feed this value into the inverse kinematics node to move the robot along the x-direction for a slide. 
- This uses Baxter's built-in forward kinematics toolkit.
- The user can choose which limb to analyse. 
 
### Execution: 
```bash
ros2 run baxter_ik position_kinematics -l {right/left}
```

## Inverse Kinematics:
- Using Baxter's built-in commands and computations, we can solve for inverse kinematics and perform the desired motion.
- This node allows us to choose the desired limb to move.
- Moves along the y-axis by a user-chosen amount for a user-chosen duration, and it breaks in between each swipe. This break stops and starts the local Daimon sensor reading. 
- These local Daimon sensor readings are required for the neural network (NN).
- The duration chosen by the user allows for a slower/faster swiping motion.
- For this experiment a duration of 2 seconds was chosen.
  
### Execution: 
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

### Execution: 
```bash
ros2 run baxter_ik daimon_sensor
```

## Neural Network:
- Employs the time series transformer neural network methodology to analyse the variables (shear, depth, shear velocity, and depth velocity)
- It allocates weights to the data and incurs penalties for incorrect presumptions when contrasted to the real tensor.
- The scale derived from the datasets was normalised to accommodate the differing magnitudes.
- The process is performed utilising the documented data for each cycle.
- A confusion matrix and a classification report are generated. The confusion matrix correlates true labels with predicted labels, enabling the analysis of category misclassifications.
- The classification report results in the statistical analyses. The statistics are categorised by precision, recall, and F1-score. 
  - Precision: the accuracy with which the model identified a material among all available materials. High precision yields a minimal false positive rate. 
  - Recall: verfies the model's success rate for each material among all the samples of a certain material in the test. A high recall signifies a low rate of false negatives. 
  - F1-Score: 2 (Precision + Recall) / (Precision + Recall). The balance metric indicates the system's performance for that particular class. This indicates the reliability and comprehensiveness of the training. 
- The model is based on a multi-feature vision-based tactile system.
- A classification report, confusion matrix, and trained model have been stored in `Saved_Models`.

## Live Classification:
- Builds the same architecture that was used in the neural network
- Imports the .pth file from the workspace to attain the knowledge from the neural network
- Lets the user start and stop the recording motion. Once stopped, the data collected from the sensor is transferred into matrices. The data are then normalised to adjust the scale of the input data.
- Computes the comparative analysis to determine which material is currently being tested. This knowledge is based off the trained data.

### Utilities:
- Calculates the shear and depth velocities
- Compresses the data to accept only 500 rows of data. This ensures standardisation when analysing the data.
- Creates a multivariate time-series matrix which is used by the neural network


# File Description:

This project requires the Baxter ROS2 bridge adapter to function properly. 
Please ensure you have cloned the customized fork into your workspace:

* **Baxter ROSBridge Adapter (Custom Fork):** [https://github.com/Stef504/baxter_rosbridge_adapter.git](https://github.com/Stef504/baxter_rosbridge_adapter.git)
* **Original Upstream Repository:** [https://github.com/maylinnkaa/baxter_rosbridge_adapter.git](https://github.com/maylinnkaa/baxter_rosbridge_adapter.git)
```bash
    |
    +-- baxter_rosbridge_adapter
    |   +-- baxter_common_ros2/       
    |
    |   +-- baxter_dataflow/      
    |
    |   +-- baxter_ik/
    |        +--ik_baxter.py                inverse kinematics
    |        +--position_kinematics.py      forward kinematics
    |        +--daimon_sensor.py            daimon sensor script     
    |        +--utilities.py                assiting function
    |        
    |    +-- baxter_interface/
    |    
    |    +-- baxter_rosbridge_adapter/
    |        +--baxter_cli.py
    |        +--calibrate_arm.py
    |        +--baxter_grippers_cli.py
    |        +--joint_state_bridge.py


+-- baxter_ik/
|        +--ik_baxter.py                inverse kinematics
|        +--position_kinematics.py      forward kinematics
|        +--daimon_sensor.py            daimon sensor script       
|        +--utilities.py                assiting function
|
|
+-- Dataset
|   +-- Metal
|       +-- solid_metal_part
|          +-- down
|           +-- left
|           +-- right
|           +-- up
|   +-- Plastic
|       +-- 3D-printed-part
|           +-- down
|           +-- left
|           +-- right
|           +-- up
|   +-- Rubber
|       +-- rubber-ball
|           +-- down
|           +-- left
|           +-- right
|           +-- up
|
+-- dmrobotics
|
+-- dmrobotics.egg-info
|
+-- plots               The generated graphs
|
+-- Saved_Models        The saved trained models (.pth), confusion matrix, classification report
|
+-- DM-Tac W- English.pdf
|
+-- DM-Tac.pdf
|
+--graphs.py           Display graphs
|
+--live_classifier.py
|
+--main.py             Hand testing- prior to any ros2 development
|
+--neural_network.py   Neural Network script
|
+--original_main.py    Original Daimon Sensor script, without any modifications
|
+--Report and Presentation.md
|
+--requirements.txt    Packages required
|
+--setup.py            Used to install packges
|
+--utilities.py        Helper Function
```


# Cognitive Approach:

## Sensor Device/Interface
- The sensor gathered sensory data (depth, shear). 
- The processed that data to yield shear velocity and depth velocity invoke the computational step. 
- These raw data, measured in pixels per mm, were then converted into graphical representations enabling users to comprehend the material's "fingerprint" that the sensor was collecting. This gave an understanding of if the data gathered had any differentiation between materials.
- The use of the sensor's raw image also provided diagnostic information to identify any wear-and-tear concerns. 


## It additionally integrated "Publish and Subscribe":
- The Daimon sensor script had to commence prior to the inverse kinematic scripts because the publisher does not care whether any node is "listening".
- The sensor is the subscriber, and the published command starts/stops the local recordings.
- The command `"START"` initiated the recording of the local recordings, specifically capturing the 2-second swiping motion. 
- `"STOP"` terminated the local recording. Together these commands allowed each swipe to be saved individually. This data was then fed into the neural network for training. 
- `"EXPERIMENT_COMPLETE"` terminated the entire recording and produced the global data for the graph. 
- This gave a synchronous and autonomous flow of data acquisition during the swiping motion. 


## Hybrid reactive-deliberative architecture:
- First is the analysis of the sense-plan-act architecture:
- The data sourced from the VBTS (`sense`) informs the `reasoning` process. - The `sense` layer is based on how the raw data and calculated data from the sensor are compared to the trained neural network, and it provides the ontology with predicates such as the material classification.
- The `plan/reasoning` then dictates the appropriate grasping force based on the material identification.
- Finally, the execution (`act`) phase entails the actual grasping of the object and task completion.
- This operates under closed-world assumptions, so linking to the bio-inspired architecture will render it a hybrid reactive-deliberative architecture.
-`Bio-inspired` denotes the approach of data acquisition and its association with receptors in human skin for the purpose of material identification.
- The deliberative layer consists of `conscious type 2 `reasoning, characterised by the `sense-plan-act` framework previously outlined.
- The reactive layer represents `reflexive type 1 logic`, integrating low-level bio-inspired behaviour. Behaviour is categorised into perceptual and motor schemas. 
- Bio-inspired behaviours are triggered by a `releaser`. The releaser is activated when the shear velocity graphs indicate a slippage.
- The `perceptual` schema processes the loss of static friction , which leads to the `motor` schema engaging and bypassing the planner. This prompts a firmer grip on the material.
