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

# EXPERIMENT VARIABLES (Change these manually before running)
MATERIAL_NAME = "plastic_jug"       
ORIENTATION   = "Up"   

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
            return
            
        folder_path = f"Thesis_Dataset/{MATERIAL_NAME}/{ORIENTATION}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        fixed_sequence = preprocess_experiment_run(self.local_time, self.local_depth, self.local_shear)
        filename = f"{folder_path}/{MATERIAL_NAME}_{ORIENTATION}_rep{self.current_rep}_{self.current_direction}.npy"
        np.save(filename, fixed_sequence)
        print(f"  [SAVED NN MATRIX] -> {filename}")

    def generate_global_plot(self):
        """Uses Global Memory to generate the entire experiment timeline."""
        depth_arr = np.array(self.global_depth)
        shear_arr = np.array(self.global_shear)
        time_arr = np.array(self.global_time)
        
        # Calculate Derivatives
        velocity = np.diff(shear_arr) / np.diff(time_arr)
        normal_velocity = np.diff(depth_arr) / np.diff(time_arr)
        
        # Strain Calculation
        delta_depth = depth_arr[-1] - self.contact_start_depth
        h_final = H_INITIAL - delta_depth
        final_strain = (h_final - H_INITIAL) / H_INITIAL

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        
        # Subplot 1
        ax1.plot(time_arr, depth_arr, color='blue', linewidth=2, label="Depth (mm)")
        ax1.set_ylabel("Depth (mm)", color='blue', fontsize=12)
        ax1.set_title(f"Full Experiment Timeline ({MATERIAL_NAME})", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        ax1_right = ax1.twinx()
        ax1_right.plot(time_arr, shear_arr, color='red', linewidth=1.5, linestyle='--', label="Integrated Shear")
        ax1_right.set_ylabel("Shear (mm)", color='red', fontsize=12)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Subplot 2
        ax2.plot(time_arr, shear_arr, color='red', linewidth=2)
        ax2.set_ylabel("Shear Integral", color='red', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.text(0.02, 0.05, f"Final Strain: {delta_depth:.4f}", transform=ax2.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

        # Subplot 3
        ax3.plot(time_arr[1:], velocity, color='green')
        ax3.set_ylabel("Shear Vel (mm/s)", color='green', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Subplot 4
        ax4.plot(time_arr[1:], normal_velocity, color='crimson', linewidth=1.5)
        ax4.set_xlabel("Continuous Experiment Time (s)", fontsize=12)
        ax4.set_ylabel("Depth Vel (mm/s)", color='crimson', fontsize=11)
        ax4.grid(True, linestyle='--', alpha=0.5)

        if not os.path.exists("plots"):
            os.makedirs("plots")
        graph_filename = f"plots/{MATERIAL_NAME}_{ORIENTATION}_FULL_ANALYSIS.png"
        plt.savefig(graph_filename, dpi=300)
        print(f"\n[SUCCESS] Global timeline plot saved to: {graph_filename}")

        plt.tight_layout()
        plt.show() # This will keep the window open until you manually close it

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
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) if len(img_raw.shape) == 3 else img_raw

            cv2.imshow('1. Raw Image', gray)
            cv2.imshow('2. Depth Heatmap', cv2.applyColorMap((depth_smooth*100).astype('uint8'), cv2.COLORMAP_HOT))        
            cv2.imshow('3. Tangential Shear Vectors', vector_vis)
            
            if cv2.waitKey(3) & 0xFF == ord('q'):
                print("Emergency exit triggered.")
                break

if __name__ == "__main__":
    logger = DaimonROSLogger()
    try:
        logger.run_sensor_loop()
    except KeyboardInterrupt:
        pass