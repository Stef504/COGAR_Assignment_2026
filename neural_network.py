import os
import glob
import torch
import torch.nn as nn
import numpy as np
import datetime
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        Scans the root directory and maps files to labels recursively.
        """
        self.file_paths = []
        self.labels = []
        
        # 1. Define the dictionary mapping your NEW folder names to integer labels
        self.class_map = {
            "Rubber": 0,
            "Plastic": 1,
            "Metal": 2
        }
        
        # 2. Search through the folders and collect the paths
        for material_name, label_idx in self.class_map.items():
            # NEW FIX: The '**' tells it to search infinitely deep into subfolders!
            search_path = os.path.join(root_dir, material_name, "**", "*.npy")
            found_files = glob.glob(search_path, recursive=True)
            
            for file_path in found_files:
                self.file_paths.append(file_path)
                self.labels.append(label_idx)
                
        print(f"Dataset Loaded: Found {len(self.file_paths)} total trial files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data_matrix = np.load(file_path)
        label = self.labels[idx]
        
        tensor_x = torch.tensor(data_matrix, dtype=torch.float32) 
        tensor_y = torch.tensor(label, dtype=torch.long)          
        return tensor_x, tensor_y

# --- 3. TRAINING LOOP & MEMORY SAVING ---
if __name__ == "__main__":
    
    print("========================================")
    print("  TACTILE TRANSFORMER TRAINING MODULE   ")
    print("========================================")
    print("Please select your root Dataset folder from the popup window...")
    
    # --- INTERACTIVE TKINTER FOLDER SELECTION ---
    root = tk.Tk()
    root.attributes('-topmost', True) 
    root.withdraw() 
    
    user_path = filedialog.askdirectory(title="Select the Root Dataset Folder")
    
    if not user_path:
        print("\n[WARNING] No folder selected. Exiting script.")
        exit()
        
    DATASET_FOLDER = user_path
    print(f"\n[SYSTEM] Loading data from: {DATASET_FOLDER}")

    # --- Configuration ---
    BATCH_SIZE = 16 
    EPOCHS = 50
    NUM_CLASSES = 3  # Updated for Rubber, Plastic, Metal
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Initialize the Dataset
    dataset = DaimonDataset(root_dir=DATASET_FOLDER)

    total_samples = len(dataset)
    if total_samples == 0:
        print(f"\n[ERROR] Found 0 .npy files inside {DATASET_FOLDER}.")
        print("Ensure your folders are named exactly: 'Rubber', 'Plastic', 'Metal'.")
        exit()
    
    # --- THE TRAIN / TEST SPLIT ---
    train_size = int(total_samples * (50.0 / 60.0)) 
    test_size = total_samples - train_size
    
    print(f"[DATA] Total Files Found: {total_samples}")
    print(f"\nSplitting Data -> Training on {train_size} files, Testing on {test_size} files.")
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize the Model
    model = TactileTransformerClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. The Training Loop
    print("\nStarting Transformer Training...")
    
    for epoch in range(1, EPOCHS + 1):
        # --- A. TRAINING PHASE ---
        model.train() 
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

        # --- B. TESTING PHASE ---
        model.eval()  
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        all_true_labels = []
        all_model_guesses = []
        
        with torch.no_grad(): 
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                
                running_test_loss += loss.item()
                _, predicted_classes = torch.max(predictions, 1)
                correct_test += (predicted_classes == batch_y).sum().item()
                total_test += batch_y.size(0)

                all_model_guesses.extend(predicted_classes.cpu().numpy())
                all_true_labels.extend(batch_y.cpu().numpy())
                
        epoch_test_loss = running_test_loss / len(test_loader)
        epoch_test_acc = (correct_test / total_test) * 100
            
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Acc: {epoch_train_acc:5.1f}% | Test Acc: {epoch_test_acc:5.1f}% (Test Loss: {epoch_test_loss:.4f})")

        # --- Detailed breakdown every 10 epochs ---
        if epoch % 10 == 0 or epoch == EPOCHS:
            print("\n" + "-"*50)
            print(f"DETAILED MATERIAL BREAKDOWN (Epoch {epoch}):")
            target_names = ["Rubber", "Plastic", "Metal"] # Updated!
            
            report = classification_report(all_true_labels, all_model_guesses, target_names=target_names, zero_division=0)
            print(report)
            print("-"  * 50 + "\n")

    # --- 4. SAVING THE MODEL'S MEMORY ---
    print("\nTraining Complete.")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(SCRIPT_DIR, "Saved_Models")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"daimon_transformer_{timestamp}.pth"
    save_path = os.path.join(model_dir, filename)
    
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Model intelligence saved permanently to: {save_path}")

    # --- 5. PRINT REPORTS AND GRAPHICS ---
    target_names = ["Rubber", "Plastic", "Metal"] # Updated!
    print("\nFINAL DETAILED MATERIAL BREAKDOWN:")
    report = classification_report(all_true_labels, all_model_guesses, target_names=target_names, zero_division=0)
    print(report)
    
    print("\nGenerating final Confusion Matrix graphic...")
    
    cm = confusion_matrix(all_true_labels, all_model_guesses)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Tactile Transformer Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Material", fontsize=12, fontweight='bold')
    plt.ylabel("True Material", fontsize=12, fontweight='bold')
    
    cm_filename = os.path.join(model_dir, f"confusion_matrix_{timestamp}.pdf")
    plt.savefig(cm_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Publication-ready Confusion Matrix saved to: {cm_filename}")
    
    plt.show()