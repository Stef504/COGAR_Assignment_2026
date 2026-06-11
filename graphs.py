import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def open_interactive_graph():
    # 1. Open a file dialog so you can easily pick which experiment to view
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Global Data NPZ", filetypes=[("Numpy Zipped", "*.npz")])
    
    if not file_path:
        print("No file selected.")
        return

    # 2. Load the raw data
    data = np.load(file_path)
    time_arr = data['time']
    depth_arr = data['depth']
    shear_arr = data['shear']
    
    # Calculate velocities
    velocity = np.diff(shear_arr) / np.diff(time_arr)
    normal_velocity = np.diff(depth_arr) / np.diff(time_arr)

    # Plotting
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 1: Indentation Profile (Left Axis) vs Scaled Shear (Right Twin Axis)
    # Using a twin axis prevents the massive integrated shear values from flattening your depth curve!
    ax1.plot(time_arr, depth_arr, color='blue', linewidth=2, label="Depth (mm)")
    ax1.set_ylabel("Depth (mm)", color='blue', fontsize=12)
    ax1.set_title("Tactile Profile Relationships over Time", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1_right = ax1.twinx()
    ax1_right.plot(time_arr, np.array(shear_hist), color='red', linewidth=1.5, linestyle='--', label="Integrated Shear")
    ax1_right.set_ylabel("Shear Displacement (mm)", color='red', fontsize=12)
    
    # Combine legends smoothly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # === Subplot 2: Isolated Shear Force Profile ===
    ax2.plot(time_arr, np.array(shear_hist), color='red', linewidth=2)
    ax2.set_title("Total Integrated Shear Displacement over Time", fontsize=14)
    ax2.set_ylabel("Shear Volumetric Integral (mm)", color='red', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    #Velocity Profile (Dip Detection)
    ax3.plot(time_arr[1:], velocity, color='green')
    ax3.set_title("Tangential Velocity - For velocity changes in shear map", fontsize=14)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Velocity (mm/s)", color='green', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Subplot 4: NEW UN-STACKED TANGENTIAL SLIP VELOCITY (FRICTION SLIP ANALYSIS)
    # This captures the high-frequency transitions of the hold-and-release pattern directly.
    ax4.plot(time_arr[1:], normal_velocity, color='crimson', linewidth=1.5)
    ax4.set_title("Depth Velocity - Changes in depth map", fontsize=13)
    ax4.set_xlabel("Time (s)", fontsize=11)
    ax4.set_ylabel("Velocity (mm/s^2)", color='crimson', fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(f"Interactive Analysis: {file_path.split('/')[-1]}", fontsize=14)
    plt.tight_layout()
    
    # This will pop open the window WITH the interactive zoom toolbar!
    plt.show()

if __name__ == "__main__":
    open_interactive_graph()