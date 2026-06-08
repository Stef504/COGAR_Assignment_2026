import torch
import torch.nn as nn
import numpy as np
from utilities import preprocess_experiment_run

# --- 1. THE TIME-SERIES TRANSFORMER ARCHITECTURE ---
class TactileTransformerClassifier(nn.Module):
    def __init__(self, num_features=4, num_classes=3, seq_len=500, d_model=64, nhead=4, num_layers=2):
        super(TactileTransformerClassifier, self).__init__()
        
        # Linear projection to map your 4 input channels to the Transformer's internal dimension (d_model)
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional Encoding: Tells the Transformer the chronological order of your time-series steps
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # The core Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1,
            batch_first=True  # Keeps shape as [Batch, Seq, Features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head: Maps the global temporal features to your final material labels
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Num_Features]
        
        # Project features and inject temporal position context
        x = self.input_projection(x) + self.positional_embedding
        
        # Pass the entire timeline through the Self-Attention layers simultaneously
        x = self.transformer_encoder(x)
        
        # Global Average Pooling across the time axis to get a single vector per sample
        x = torch.mean(x, dim=1)
        
        # Compute final classification probabilities
        output = self.fc_out(x)
        return output


# --- 2. PREPARING AND SHAPING YOUR EXPERIMENTAL DATA ---
def preprocess_experiment_run(time_hist, depth_hist, shear_hist, target_len=500):
    """
    Takes your raw lists from a single experiment run, aligns derivatives,
    and reshapes them into a fixed sequence length window for the neural network.
    """
    depth_arr = np.array(depth_hist)
    shear_arr = np.array(shear_hist)
    time_arr = np.array(time_hist)
    
    # Calculate aligned physical derivatives
    normal_velocity = np.diff(depth_arr) / np.diff(time_arr)
    shear_slip_velocity = np.diff(shear_arr) / np.diff(time_arr)
    
    # Align lengths by slicing away the first index of the raw metrics
    depth_aligned = depth_arr[1:]
    shear_aligned = shear_arr[1:]
    
    # Stack channels horizontally into a multivariate time-series matrix
    # Matrix shape: (Sequence Length, 4 Features)
    multivariate_matrix = np.column_stack((depth_aligned, shear_aligned, normal_velocity, shear_slip_velocity))
    
    # Interpolate/Resize to a fixed length (target_len) so the Transformer layers match
    current_len = multivariate_matrix.shape[0]
    indices = np.linspace(0, current_len - 1, target_len).astype(int)
    fixed_length_sequence = multivariate_matrix[indices]
    
    return fixed_length_sequence


# --- 3. PIPELINE PIPING & TRAINING LOOP ---
if __name__ == "__main__":
    # Hyperparameters
    SEQ_LEN = 500       # Every experimental run is normalized to 500 timesteps
    NUM_FEATURES = 4    # [Depth, Shear, Normal_Vel, Shear_Slip_Vel]
    NUM_CLASSES = 3     # Example: 0=PLA_Smooth, 1=PLA_Rough, 2=Rubber
    
    # Target execution on your dedicated PC GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline execution model on: {device}")
    
    # Initialize the model
    model = TactileTransformerClassifier(
        num_features=NUM_FEATURES, 
        num_classes=NUM_CLASSES, 
        seq_len=SEQ_LEN
    ).to(device)
    
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # --- SIMULATED DATA LOADING ---
    # In your real setup, you will save your preprocessed runs using np.save()
    # and load them here to train. Let's create dummy shapes for demonstration:
    num_samples = 60 # e.g., 10 repetitions * 3 materials * 2 orientations
    dummy_x = torch.randn(num_samples, SEQ_LEN, NUM_FEATURES).to(device)
    dummy_y = torch.randint(0, NUM_CLASSES, (num_samples,)).to(device)
    
    # Simple training loop layout
    model.train()
    print("\nBeginning Transformer Training...")
    for epoch in range(1, 21):
        optimizer.zero_grad()
        
        # Forward Pass
        predictions = model(dummy_x)
        loss = criterion(predictions, dummy_y)
        
        # Backward Pass (Backpropagation)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            # Calculate running accuracy
            _, predicted_classes = torch.max(predictions, 1)
            correct = (predicted_classes == dummy_y).sum().item()
            accuracy = correct / num_samples * 100
            print(f"Epoch [{epoch}/20] -> Loss: {loss.item():.4f} | Accuracy: {accuracy:.1f}%")
            
    print("\nModel training complete! The Transformer can now extract multivariate stick-slip profiles.")