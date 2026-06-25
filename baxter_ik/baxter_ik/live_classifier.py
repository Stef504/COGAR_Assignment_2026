import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from dmrobotics import Sensor, put_arrows_on_image
from utilities import preprocess_experiment_run

# --- 1. IMPORT YOUR MODEL ARCHITECTURE ---
# This MUST perfectly match the architecture used during training!
class TactileTransformerClassifier(nn.Module):
    def __init__(self, num_features=4, num_classes=3, seq_len=500, d_model=64, nhead=4, num_layers=2):
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

# --- 2. LIVE CLASSIFICATION NODE (MANUAL TESTING) ---
class LiveTactileClassifier:
    def __init__(self, weights_path):
        # 1. Hardware Setup
        self.sensor = Sensor("S2508080077")
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
        self.class_map = {0: "Rubber", 1: "Plastic", 2: "Metal"}
        
        # Memory variables
        self.is_recording = False
        self.start_time = 0.0
        self.contact_start_depth=0.0
        self.experiment_start_time= time.time()
        self.time_hist, self.shear_hist, self.depth_hist = [], [], []

    def execute_live_inference(self):
        if len(self.time_hist) < 10:
            print("Not enough data recorded! Try swiping longer.")
            return
            
        # If the maximum recorded depth is 0, the sensor was hovering in the air 
        # (or the human didn't press hard enough to cross the DEPTH_THRESHOLD).
        if max(self.depth_hist) <= 0.0:
            print("\n" + "="*50)
            print(" [WARNING] NO CONTACT DETECTED")
            print(" The sensor did not touch anything firmly enough during the recording.")
            print(" Prediction aborted to prevent false guesses.")
            print("="*50 + "\n")
            return
        
        print("\n[PROCESSING] Analyzing tactile physics...")
        
        # 1. Preprocess: Standardize length to 500 steps
        features_matrix = preprocess_experiment_run(self.time_hist, self.depth_hist, self.shear_hist)

        # 3. Z-Score Normalization (Mirroring the Training Script)
        for i in range(features_matrix.shape[1]):
            col_mean = np.mean(features_matrix[:, i])
            col_std = np.std(features_matrix[:, i])
            if col_std > 1e-6: 
                features_matrix[:, i] = (features_matrix[:, i] - col_mean) / col_std
            else:
                features_matrix[:, i] = features_matrix[:, i] - col_mean
        
        # 4. Convert to Tensor & add Batch dimension (Shape: [1, 500, 4])
        tensor_x = torch.tensor(features_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 5. Feed it to the brain
        with torch.no_grad():
            raw_prediction = self.model(tensor_x)
            
            predicted_class_id = torch.argmax(raw_prediction, dim=1).item()
            confidence = torch.softmax(raw_prediction, dim=1)[0][predicted_class_id].item() * 100
            
            material_name = self.class_map[predicted_class_id]
            print("\n" + "="*50)
            print(f" PREDICTION: This material is {material_name.upper()} ({confidence:.1f}% match)")
            print("="*50 + "\n")

    def run_sensor_loop(self):
        print("\nReady! Press the SPACEBAR to start recording a swipe.")
        print("Press SPACEBAR again to stop and classify.")
        print("Press 'q' to quit.")
        DEPTH_THRESHOLD = 0.015
        
        while True:
            img_raw = self.sensor.getRawImage()
            depth_map = self.sensor.getDepth()
            shear_map = self.sensor.getShear()
            black_img = np.zeros((240, 320, 3), dtype=np.uint8)
            
            # Real-time Symmetrical Physics Calculation
            depth_smooth = cv2.GaussianBlur(depth_map, (5, 5), 0)
            contact_mask = (depth_smooth > DEPTH_THRESHOLD).astype(np.uint8)
            
            masked_shear = shear_map * contact_mask[:, :, np.newaxis]
            masked_depth = depth_smooth * contact_mask
            
            cumulative_shear = np.sqrt(np.sum(masked_shear[:, :, 0])**2 + np.sum(masked_shear[:, :, 1])**2)
            cumulative_depth = np.sum(masked_depth)
            
            if self.contact_start_depth == 0.0 and cumulative_depth > 0.5:
                self.contact_start_depth = cumulative_depth
                print(f"Contact triggered at {time.time() - self.experiment_start_time:.2f}s")
            
            if self.is_recording:
                curr_time = time.time() - self.start_time
                self.time_hist.append(curr_time)
                self.depth_hist.append(cumulative_depth)
                self.shear_hist.append(cumulative_shear)
                
                # Visual indicator drawn on both diagnostic windows
                cv2.putText(black_img, "RECORDING SWIPE...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 150, 2)

            # --- Generate the arrow visual using the masked shear data ---
            vector_vis = put_arrows_on_image(black_img, masked_shear * 20)

            # Display all 3 perspectives
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) if len(img_raw.shape) == 3 else img_raw
            
            cv2.imshow('1. Raw Sensor', gray)
            cv2.imshow('2. Depth Heatmap', cv2.applyColorMap((depth_smooth*100).astype('uint8'), cv2.COLORMAP_HOT))        
            cv2.imshow('3. Tangential Shear Vectors', vector_vis)
            
            # Keyboard Controls
            key = cv2.waitKey(3) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '): # Spacebar pressed
                if not self.is_recording:
                    print("-> Recording started...")
                    self.time_hist, self.shear_hist, self.depth_hist = [], [], []
                    self.start_time = time.time()
                    self.is_recording = True
                else:
                    print("-> Recording stopped.")
                    self.is_recording = False
                    self.execute_live_inference()

if __name__ == "__main__":
    
    WEIGHTS_FILE = "Saved_Models/daimon_transformer_20260623_1459.pth" 
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"[ERROR] Cannot find model at: {WEIGHTS_FILE}")
        print("Please verify the filename in your Saved_Models folder!")
    else:
        classifier = LiveTactileClassifier(weights_path=WEIGHTS_FILE)
        try:
            classifier.run_sensor_loop()
        except KeyboardInterrupt:
            pass
        finally:
            classifier.sensor.disconnect()
            cv2.destroyAllWindows()