import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import roslibpy
from dmrobotics import Sensor, put_arrows_on_image
from baxter_ik.utilities import preprocess_experiment_run

# --- 1. IMPORT OR RECREATE YOUR MODEL ARCHITECTURE ---
# The architecture must exactly match the one used during training!
class TactileTransformerClassifier(nn.Module):
    def __init__(self, num_features=5, num_classes=4, seq_len=500, d_model=64, nhead=4, num_layers=2):
        super(TactileTransformerClassifier, self).__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x) + self.positional_embedding
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1) 
        output = self.fc_out(x)
        return output

# --- 2. LIVE CLASSIFICATION NODE ---
class LiveTactileClassifier:
    def __init__(self, weights_path, host='130.251.13.31', port=9090):
        # 1. Hardware Setup (N160MU2 Camera)
        self.sensor = Sensor("S2508080069")
        print("Taring sensor baseline...")
        self.sensor.reset()
        time.sleep(1.0)
        
        # 2. Neural Network Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TactileTransformerClassifier().to(self.device)
        
        # Load the saved intelligence and lock it into evaluation mode
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval() 
        print(f"Neural Network Loaded from: {weights_path}")
        
        # The reverse dictionary to translate the math ID back to English
        self.class_map = {0: "Glass", 1: "Plastic", 2: "Wood", 3: "Metal"}
        
        # 3. ROS Network Setup
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        if not self.client.is_connected:
            print("CRITICAL: Failed to connect to ROS.")
            exit()
            
        self.status_sub = roslibpy.Topic(self.client, '/tactile_experiment/status', 'std_msgs/String')
        self.status_sub.subscribe(self.status_callback)
        
        # Memory variables
        self.is_recording = False
        self.start_time = 0.0
        self.time_hist, self.shear_hist, self.depth_hist = [], [], []

    def status_callback(self, msg):
        command = msg['data']
        
        if command.startswith("START_FORWARD"):
            print("\n[LIVE INFERENCE] Recording swipe data...")
            self.time_hist, self.shear_hist, self.depth_hist = [], [], []
            self.start_time = time.time()
            self.is_recording = True
            
        elif command.startswith("STOP_FORWARD"):
            self.is_recording = False
            self.execute_live_inference()

    def execute_live_inference(self):
        if len(self.time_hist) < 10:
            return
            
        # 1. Preprocess the live data exactly how it was trained
        fixed_sequence = preprocess_experiment_run(self.time_hist, self.depth_hist, self.shear_hist)
        
        # 2. Convert to PyTorch Tensor. 
        # Crucial step: 'unsqueeze(0)' adds a dummy Batch dimension so shape becomes [1, 500, 5]
        tensor_x = torch.tensor(fixed_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Feed it to the brain
        with torch.no_grad(): # Tells PyTorch not to calculate gradients (saves memory/time)
            raw_prediction = self.model(tensor_x)
            
            # 4. Extract the highest probability
            predicted_class_id = torch.argmax(raw_prediction, dim=1).item()
            confidence = torch.softmax(raw_prediction, dim=1)[0][predicted_class_id].item() * 100
            
            material_name = self.class_map[predicted_class_id]
            print("==================================================")
            print(f" PREDICTION: This material is {material_name.upper()} ({confidence:.1f}% match)")
            print("==================================================")

    def run_sensor_loop(self):
        print("Ready. Awaiting Baxter's swipe command...")
        DEPTH_THRESHOLD = 0.015
        
        while self.client.is_connected:
            img_raw = self.sensor.getRawImage()
            depth_map = self.sensor.getDepth()
            shear_map = self.sensor.getShear()
            
            if self.is_recording:
                curr_time = time.time() - self.start_time
                max_depth = np.max(depth_map)
                
                depth_smooth = cv2.GaussianBlur(depth_map, (5, 5), 0)
                contact_mask = (depth_smooth > DEPTH_THRESHOLD).astype(np.uint8)
                masked_shear = shear_map * contact_mask[:, :, np.newaxis]
                total_shear_force = np.sqrt(np.sum(masked_shear[:, :, 0])**2 + np.sum(masked_shear[:, :, 1])**2)
                
                self.time_hist.append(curr_time)
                self.depth_hist.append(max_depth)
                self.shear_hist.append(total_shear_force)

            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) if len(img_raw.shape) == 3 else img_raw
            cv2.imshow('Live Inference Feed', gray)
            
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    # Point this to whichever .pth file you want to test!
    WEIGHTS_FILE = "Saved_Models/daimon_transformer_20260610_1230.pth" 
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: Cannot find model at {WEIGHTS_FILE}")
    else:
        classifier = LiveTactileClassifier(weights_path=WEIGHTS_FILE)
        try:
            classifier.run_sensor_loop()
        except KeyboardInterrupt:
            pass