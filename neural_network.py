import os
import glob
import torch
import torch.nn as nn
import numpy as np
import datetime

from torch.utils.data import Dataset, DataLoader

# --- 1. THE TIME-SERIES TRANSFORMER ARCHITECTURE ---
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
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x) + self.positional_embedding
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1) # Global Average Pooling
        output = self.fc_out(x)
        return output

# --- 2. THE CUSTOM DATASET LOADER (Reading the Folders) ---
class DaimonDataset(Dataset):
    def __init__(self, root_dir):
        """
        Scans the root directory ('DataSets') and maps files to labels.
        """
        self.file_paths = []
        self.labels = []
        
        # 1. Define the dictionary mapping your folder names to integer labels
        self.class_map = {
            "Glass": 0,
            "Plastic": 1,
            "Wood": 2,
            "Metal": 3
        }
        
        # 2. Search through the folders and collect the paths to every .npy file
        for material_name, label_idx in self.class_map.items():
            # Use glob to find all .npy files inside Material/Orientation folders
            search_path = os.path.join(root_dir, material_name, "*", "*.npy")
            found_files = glob.glob(search_path, recursive=True)
            
            for file_path in found_files:
                self.file_paths.append(file_path)
                self.labels.append(label_idx)
                
        print(f"Dataset Loaded: Found {len(self.file_paths)} total trial files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Load the specific .npy file from the hard drive
        file_path = self.file_paths[idx]
        data_matrix = np.load(file_path)
        
        # 2. Get the corresponding label
        label = self.labels[idx]
        
        # 3. Convert both to PyTorch Tensors
        # Convert data to Float32 for neural network math
        tensor_x = torch.tensor(data_matrix, dtype=torch.float32) 
        # Convert label to Long for Classification loss
        tensor_y = torch.tensor(label, dtype=torch.long)          
        
        return tensor_x, tensor_y

# --- 3. TRAINING LOOP & MEMORY SAVING ---
if __name__ == "__main__":
    # --- Configuration ---
    DATASET_FOLDER = "DataSets" # Ensure this matches your actual folder name
    BATCH_SIZE = 16 
    EPOCHS = 50
    NUM_CLASSES = 4  # Includes Glass, Plastic, Wood, Metal
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Initialize the Dataset
    if not os.path.exists(DATASET_FOLDER):
        print(f"ERROR: Cannot find folder '{DATASET_FOLDER}'. Please create it and add your data.")
        exit()
        
    dataset = DaimonDataset(root_dir=DATASET_FOLDER)
    
    # ---------------------------------------------------------
    # NEW: THE TRAIN / TEST SPLIT
    # ---------------------------------------------------------
    total_samples = len(dataset)
    
    # Calculate sizes (e.g., 50 train / 10 test per 60 items is ~83% / 17%)
    # This math automatically scales even if you collect more than 60 reps!
    train_size = int(total_samples * (50.0 / 60.0)) 
    test_size = total_samples - train_size
    
    print(f"\nSplitting Data -> Training on {train_size} files, Testing on {test_size} files.")
    
    # Randomly shuffle and split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create TWO DataLoaders now (one for studying, one for the final exam)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize the Model
    model = TactileTransformerClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. The Training Loop
    print("\nStarting Transformer Training...")
    
    for epoch in range(1, EPOCHS + 1):
        # --- A. TRAINING PHASE (Studying) ---
        model.train() # Tell the model it is allowed to learn and change weights
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct_train += (predicted_classes == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = (correct_train / total_train) * 100

        # --- B. TESTING PHASE (The Exam) ---
        model.eval()  # Freeze weights, turn off Dropout layers
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad(): # Disable memory-heavy gradient tracking during tests
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                
                running_test_loss += loss.item()
                _, predicted_classes = torch.max(predictions, 1)
                correct_test += (predicted_classes == batch_y).sum().item()
                total_test += batch_y.size(0)
                
        epoch_test_loss = running_test_loss / len(test_loader)
        epoch_test_acc = (correct_test / total_test) * 100
            
        # --- C. PRINT EPOCH RESULTS ---
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Acc: {epoch_train_acc:5.1f}% | Test Acc: {epoch_test_acc:5.1f}% (Test Loss: {epoch_test_loss:.4f})")

    # --- 4. SAVING THE MODEL'S MEMORY ---
    print("\nTraining Complete.")
    
    # 1. Define the target directory
    model_dir = "Saved_Models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 2. Build the precise path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"daimon_transformer_{timestamp}.pth"
    save_path = os.path.join(model_dir, filename)
    
    # 3. Save
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Model intelligence saved permanently to: {save_path}")
    
    # Create a dynamic filename using the current date and time
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f"daimon_transformer_{timestamp}.pth"
    
    # Use torch.save for the model weights
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Model intelligence saved permanently to: {save_path}")