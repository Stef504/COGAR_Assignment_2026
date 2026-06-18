## How to execute Baxter, IK, Daimon Sensor and Position Kinematics 

# Position Kinematics:
- This allows us to find the artesian co-ordinates of the wrist with respect to the base. We feed this value into the inverse kinematics node to move the robot along the x-direction for a slide. 
- This uses Baxter's built-in FK toolkit.
- The user can choose which limb to move. 
# Code: 
```bash
ros2 run baxter_ik position_kinematics -l {right/left}
```

# Inverse Kinematics:
- To make the tests as uniform as possible, it was decided to use Baxter. This allows repeatability and a continuous force (Baxter uses a spring for its joint – need to check how accurate and continuous the force is???).
- Using Baxter's built-in commands and computations, we can solve for inverse kinematics and perform the desired motion.
- This node allows us to choose the desired limb to move.
- Moves along the x-axis by a user-chosen amount, and it breaks in between each swipe. This break stops and starts the Daimon sensor reading. 
- Individual readings are required for the neural network (NN).
- 
- 
# Code: 
```bash
ros2 run baxter_ik ik_baxter -l {right.left}
```

## Daimon Sensor
- This is connected to the sensor on the robot's fingertips.
- With each swipe the sensor is activated and records the respective arrays of data (depth, shear, shear velocity (derivative of shear), and normal velocity (derivative of depth)). These sets are then fed into the NN. 
- For accurate representation of the active contact zones, the depth and shear maps/graphs were isolated to focus only on the active area so as to avoid any readings of noise.
- Bridge of data from the daimon sensor:
- Raw Image-- (240,320)
- getDeformation2D-- (240, 320, 2) represents two-dimensional deformation of each pixel in mm.
- getDepth-- (240, 320) represents the normal deformation, elastic deformation of the elastomer 
- getShear-- (240,320,2) tangential deformation information of each pixel in mm (parallel to contact surface)
- The shear and deformation maps are taken from the raw image and depth maps. The analysis of these graphs and their time derivatives allows us to distinguish between each material, which mainly focuses on the squishableness and roughness of the material. 
- A helper function was created `Utilities`to stack the arrays (shear, depth, shear velocity, and normal velocity).
- After an entire experiment of repetitions has been completed, a graph is generated with the respective data. This allows us to visualise whether the data obtained is ideal. (Prior hand testing was done with the chosen variables to validate if a clear difference is noticeable.)

# Code: 
```bash
ros2 run baxter_ik daimon_sensor
```

# Neural Network and Classifier:
- Uses the time series transformer neural networking technique to learn from the variables (shear, depth, shear velocity, and depth velocity)
- It assigns weights to the data and receives penalties (for its incorrect assumptions based (once compared with the actual tensor)).
- This is executed after taking a certain amount of trails for each material in different directions.
- A confusion matrix and classification report is produced. The confusion matrix maps the true labels against the predicted labels, this way we can anaylse which catergories are being confsued with other catergories, this would be a result of similar patterns
- Classification report leads to the stastical anaylses . the statics are groups by precision, recall and F1-score. Precision: how well or poorly the model was we naming a material out of all the materials. High precision resulst in a low false-positive. Recall : Out of all the samples of a specific material present in the test , how succeful was the model per material. High recall indicates a low false-negative rate. F1-Score: 2* (Precision * Recall)/(precision+recall). balance metrix representing performance for that specific class. So how trustworthy is the systen and how thorough
- System is based on multi-feature Vision Based Tactile System 
- 

# Executing Baxter:
- To execute Baxter, the following commands can be run: 

This allows us to visualise Baxter in RVIZ (ensure Baxter is plugged in):
```bash
ros2 launch baxter_rosbridge_adapter baxter_visualization.launch.py 
```

This allows us to control the joint commands:
```bash
ros2 run baxter_rosbridge_adapter baxter_cli
ros2 run baxter_rosbridge_adapter calibrate_arm -l left
```
To view what Baxter is cable of and what information we can obtain it was useful to view the `baxter_common_ros2` package. 

## Useful Observations to consider when running Baxter:
- Baxters joints are springs
- The longer the during of the movement along x the better "straight" line touch we will get instead of a dip touch. This could be due to stability. 
- Also when in the initial position, adjust the z axis to be 10cm high because Baxter lowers itself. For example the posistion_kinematics reported its current position in a range in the z-axis from -0.1454 to -0.1468. I adjusted the z-axis in the IK prompt to be -0.1300 and that resulted in smoother slide across the plastic component.
- Try to keep the downard motion constant across all materials so using trail and error to ensure that when placing an object at a certain location baxter does the same swiping movement. 
- Second readings for z-axis from position-Kinematics now the range of the most accurate IK is the following with a +/-0.0025
- x,y,z,qx,qy,qz,qw=[0.7631,0.2538,-0.1350,0.4857,0.5257,0.5141,0.4727]

lmetal_pos:
[-0.2784,0.5193,-0.8130,0.9906,0.7271,-1.1068, 2.1334]

new lmetal_pos
[-0.0245,0.1956,-0.2075,1.3081,0.8122,-1.3461,1.6713]


position from white section till lowest black line is 0.6cm + 0.02cm=0.62cm  (0.9mm)
3.6cm
0.2cm + 0.025cm  (5*0.05)
