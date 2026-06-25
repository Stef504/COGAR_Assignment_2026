import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dmrobotics import Sensor, put_arrows_on_image
from baxter_ik.utilities import preprocess_experiment_run
import roslibpy



# --- 1. CONFIGURATION & CALIBRATION ---
PIXEL_TO_MM = 20       
PIXEL_AREA = (16.0 * 12.0 / (320 * 240))   
GEL_THICKNESS = 20.1      
DEPTH_THRESHOLD = 0.015  
H_INITIAL = 10.0         



class DaimonROSLogger:
    def __init__(self, host='130.251.13.31', port=9090):

        # --- 1. INTERACTIVE CONFIGURATION ---
        print("\n" + "="*40)
        print(" DAIMON SENSOR DATA COLLECTION INITIALIZED")
        print("="*40)
        self.material_name = input("Enter Material Name (e.g., Plastic, Wood, Glass, Metal): ").strip()
        self.orientation = input("Enter Orientation (e.g., Up, Down,Diagonal_Right,Diagonal_Left,Left,Right): ").strip()
        print(f"\n[TARGET DIRECTORY] -> Dataset/{self.material_name}/{self.orientation}/")
        print("="*40 + "\n")

        # Hardware Setup
        dev_serial_id = "S2508080069" # N160MU2 Camera
        self.sensor = Sensor(dev_serial_id)
        print("Taring sensor baseline...")
        self.sensor.reset()
        time.sleep(1.0)
        
        # ROS Setup
        self.client = roslibpy.Ros(host=host, port=port)
        try:
            self.client.run()
        except Exception as exc:
            print(f"CRITICAL: Daimon node could not connect to Baxter at ws://{host}:{port}: {exc}\n"
                  "Check that the robot is on, rosbridge_server is running, and the host/port are correct.")
            raise SystemExit(1)
        if not self.client.is_connected:
            print("CRITICAL: Daimon node could not connect to Baxter rosbridge.")
            raise SystemExit(1)
            
        print("Daimon node connected! Listening for Baxter commands...")
        
        # State Machine Variables
        self.is_recording = False
        self.current_rep = 0
        self.current_direction = ""
        self.rep_start_time = 0.0
        self.experiment_start_time = time.time()
        self.contact_start_depth = 0.0
        
        # LOCAL MEMORY (For Neural Network .npy files - clears every swipe)
        self.local_time, self.local_shear, self.local_depth = [], [], []
        
        # GLOBAL MEMORY (For the final Matplotlib graph - never clears)
        self.global_time, self.global_shear, self.global_depth, self.global_area = [], [], [], []
        
        # Subscribe to Baxter
        self.status_sub = roslibpy.Topic(self.client, '/tactile_experiment/status', 'std_msgs/String')
        self.status_sub.subscribe(self.status_callback)

    def status_callback(self, msg):
        command = msg['data']
        print(f"[NETWORK COMMAND] -> {command}")
        
        if command.startswith("START"):
            parts = command.split("_")
            self.current_direction = parts[1].lower() 
            self.current_rep = parts[3]
            
            # Clear Local Memory ONLY
            self.local_time, self.local_shear, self.local_depth = [], [], []
            self.rep_start_time = time.time()
            self.is_recording = True
            
        elif command.startswith("STOP"):
            self.is_recording = False
            self.save_local_matrix()
            
        elif command == "EXPERIMENT_COMPLETE":
            print("\nAll loops finished. Generating Global Analysis Graph...")
            self.is_recording = False
            self.status_sub.unsubscribe()
            self.generate_global_plot() # Trigger the graph before shutting down
            
            # Safely close hardware
            self.client.terminate()
            self.sensor.disconnect()
            cv2.destroyAllWindows()
            exit()

    def save_local_matrix(self):
        """Processes Local Memory into a matrix and saves the .npy file."""
        if len(self.local_time) < 10:
            print("  [WARNING] Not enough data to save matrix.")
            return
        
        # 1. Gets: .../baxter_ros2_ws/src/baxter_ik/baxter_ik
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Goes up ONE level to: .../baxter_ros2_ws/src/baxter_ik
        PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "./../../../../../"))
        
        # 3. Safely build the Dataset path
        folder_path = os.path.join(PACKAGE_ROOT, "Dataset", self.material_name, self.orientation)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        fixed_sequence = preprocess_experiment_run(self.local_time, self.local_depth, self.local_shear)
        filename = os.path.join(folder_path, f"{self.material_name}_{self.orientation}_rep{self.current_rep}_{self.current_direction}.npy")
        
        np.save(filename, fixed_sequence)
        print(f"  [SAVED NN MATRIX] -> {filename}")

    def generate_global_plot(self):
        """Saves the raw global data so it can be viewed interactively offline."""
        
        # 1. Gets: .../baxter_ros2_ws/src/baxter_ik/baxter_ik
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Goes up ONE level to: .../baxter_ros2_ws/src/baxter_ik
        PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../../"))
        
        # 3. Safely build the plots path
        PLOT_DIR = os.path.join(PACKAGE_ROOT, "plots")
        
        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)
            
        data_filename = os.path.join(PLOT_DIR, f"{self.material_name}_{self.orientation}_GLOBAL_DATA.npz")
        
        np.savez(data_filename, 
                 time=np.array(self.global_time), 
                 depth=np.array(self.global_depth), 
                 shear=np.array(self.global_shear))
                 
        print(f"\n[SUCCESS] Global timeline data saved safely to: {data_filename}")

    def run_sensor_loop(self):
        while self.client.is_connected:
            img_raw = self.sensor.getRawImage()
            depth_map = self.sensor.getDepth()
            shear_map = self.sensor.getShear()
            black_img = np.zeros((240, 320, 3), dtype=np.uint8)

            depth_smooth = cv2.GaussianBlur(depth_map, (5, 5), 0)
            contact_mask = (depth_smooth > DEPTH_THRESHOLD).astype(np.uint8)
            contact_area_mm2 = np.sum(contact_mask) * PIXEL_AREA
            
            masked_shear = shear_map * contact_mask[:, :, np.newaxis]
            total_shear_force = np.sqrt(np.sum(masked_shear[:, :, 0])**2 + np.sum(masked_shear[:, :, 1])**2)
            
            # Record Global Initial Contact Depth
            max_depth = np.max(depth_map)
            if self.contact_start_depth == 0.0 and max_depth > DEPTH_THRESHOLD:
                self.contact_start_depth = max_depth

            # --- ALWAYS RECORD TO GLOBAL MEMORY ---
            # This captures the data continuously so the pauses show up as flatlines on the graph
            global_curr_time = time.time() - self.experiment_start_time
            self.global_time.append(global_curr_time)
            self.global_depth.append(max_depth)
            self.global_shear.append(total_shear_force)
            self.global_area.append(contact_area_mm2)

            # --- ONLY RECORD TO LOCAL MEMORY IF ACTIVE ---
            if self.is_recording:
                local_curr_time = time.time() - self.rep_start_time
                self.local_time.append(local_curr_time)
                self.local_depth.append(max_depth)
                self.local_shear.append(total_shear_force)
                cv2.putText(black_img, f"REC: REP {self.current_rep} {self.current_direction.upper()}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Visualization Updates
            vector_vis = put_arrows_on_image(black_img, masked_shear * 20)
            if len(img_raw.shape) == 3:
                gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_raw

            cv2.imshow('1. Raw Image', gray)
            cv2.imshow('2. Depth Heatmap', cv2.applyColorMap((depth_smooth*100).astype('uint8'), cv2.COLORMAP_HOT))        
            cv2.imshow('3. Tangential Shear Vectors', vector_vis)
            
            if cv2.waitKey(3) & 0xFF == ord('q'):
                print("Emergency exit triggered.")
                break

def main(args=None):
    logger = DaimonROSLogger()
    try:
        logger.run_sensor_loop()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()