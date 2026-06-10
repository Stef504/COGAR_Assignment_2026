import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dmrobotics import Sensor, put_arrows_on_image
from utilities import preprocess_experiment_run
import roslibpy

# --- 1. CONFIGURATION & CALIBRATION ---
PIXEL_TO_MM = 20       
PIXEL_AREA = (16.0 * 12.0 / (320 * 240))   
GEL_THICKNESS = 20.1      
DEPTH_THRESHOLD = 0.015  
H_INITIAL = 10.0         

# EXPERIMENT VARIABLES (Change these before running)
MATERIAL_NAME = "Plastic"       
ORIENTATION   = "Down"   

class DaimonROSLogger:
    def __init__(self, host='130.251.13.31', port=9090):
        # Hardware Setup
        dev_serial_id = "S2508080069" # N160MU2 Camera
        self.sensor = Sensor(dev_serial_id)
        print("Taring sensor baseline...")
        self.sensor.reset()
        time.sleep(1.0)
        
        # ROS Setup
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        
        if not self.client.is_connected:
            print("CRITICAL: Daimon node could not connect to ROS master.")
            exit()
            
        print("Daimon node connected! Listening for Baxter commands...")
        
        # State Machine Variables
        self.is_recording = False
        self.current_rep = 0
        self.current_direction = ""
        self.start_time = 0.0
        
        # Data Arrays
        self.time_hist, self.shear_hist, self.depth_hist = [], [], []
        
        # Subscribe to the Experiment Director
        self.status_sub = roslibpy.Topic(self.client, '/tactile_experiment/status', 'std_msgs/String')
        self.status_sub.subscribe(self.status_callback)

    def status_callback(self, msg):
        """Listens to Baxter and triggers recording states."""
        command = msg['data']
        print(f"[NETWORK COMMAND] -> {command}")
        
        if command.startswith("START"):
            # Example command: "START_FORWARD_REP_1"
            parts = command.split("_")
            self.current_direction = parts[1].lower() # 'forward' or 'backward'
            self.current_rep = parts[3]
            
            # Clear previous arrays and start fresh
            self.time_hist, self.shear_hist, self.depth_hist = [], [], []
            self.start_time = time.time()
            self.is_recording = True
            
        elif command.startswith("STOP"):
            # Stop recording and process the file
            self.is_recording = False
            self.save_data_matrix()
            
        elif command == "EXPERIMENT_COMPLETE":
            print("\nAll loops finished. Shutting down Daimon node safely.")
            self.status_sub.unsubscribe()
            self.client.terminate()
            self.sensor.disconnect()
            cv2.destroyAllWindows()
            exit()

    def save_data_matrix(self):
        """Processes the lists into a matrix and saves the .npy file."""
        if len(self.time_hist) < 10:
            print("Warning: Not enough data captured to save matrix.")
            return
            
        # Ensure directory exists
        folder_path = f"Thesis_Dataset/{MATERIAL_NAME}/{ORIENTATION}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Call your utility function to normalize to 500 steps
        fixed_sequence = preprocess_experiment_run(self.time_hist, self.depth_hist, self.shear_hist)
        
        # Build dynamic filename: e.g., Plastic_Down_rep1_forward.npy
        filename = f"{folder_path}/{MATERIAL_NAME}_{ORIENTATION}_rep{self.current_rep}_{self.current_direction}.npy"
        np.save(filename, fixed_sequence)
        print(f"[SAVED] Data matrix successfully written to: {filename}")

    def run_sensor_loop(self):
        """The main OpenCV loop. Runs continuously but only logs when instructed."""
        while self.client.is_connected:
            img_raw = self.sensor.getRawImage()
            depth_map = self.sensor.getDepth()
            shear_map = self.sensor.getShear()
            black_img = np.zeros((240, 320, 3), dtype=np.uint8)

            depth_smooth = cv2.GaussianBlur(depth_map, (5, 5), 0)
            contact_mask = (depth_smooth > DEPTH_THRESHOLD).astype(np.uint8)
            
            masked_shear = shear_map * contact_mask[:, :, np.newaxis]
            total_shear_force = np.sqrt(np.sum(masked_shear[:, :, 0])**2 + np.sum(masked_shear[:, :, 1])**2)
            
            # --- ONLY APPEND DATA IF BAXTER SAID 'START' ---
            if self.is_recording:
                curr_time = time.time() - self.start_time
                max_depth = np.max(depth_map)
                
                self.time_hist.append(curr_time)
                self.depth_hist.append(max_depth)
                self.shear_hist.append(total_shear_force)
                
                # Visual indicator that it is actively recording
                cv2.putText(black_img, f"RECORDING: {self.current_direction.upper()}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            vector_vis = put_arrows_on_image(black_img, masked_shear * 20)
            
            if len(img_raw.shape) == 3:
                gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_raw

            # Live Visualization
            cv2.imshow('1. Raw Image', gray)
            cv2.imshow('2. Depth Heatmap', cv2.applyColorMap((depth_smooth*100).astype('uint8'), cv2.COLORMAP_HOT))        
            cv2.imshow('3. Tangential Shear Vectors', vector_vis)
            
            # You can still press 'q' for an emergency manual override stop
            if cv2.waitKey(3) & 0xFF == ord('q'):
                print("Emergency manual exit triggered.")
                break

if __name__ == "__main__":
    logger = DaimonROSLogger()
    try:
        logger.run_sensor_loop()
    except KeyboardInterrupt:
        pass