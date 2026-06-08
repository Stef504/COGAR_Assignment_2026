import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dmrobotics import Sensor, put_arrows_on_image
from utilities import preprocess_experiment_run

# --- 1. CONFIGURATION & CALIBRATION ---
# These constants are derived from your sensor datasheet or manual calibration
PIXEL_TO_MM = 20       # pixels per mm
PIXEL_AREA = (16.0 * 12.0 / (320 * 240))   # mm^2 per pixel (calibrated from a known reference object) [cite: 261] 
GEL_THICKNESS = 20.1      # L in mm
DEPTH_THRESHOLD = 0.015  # Noise floor for contact detection
H_INITIAL = 10.0         # Physical height of your test object

# 2. CONFIGURATION VARIABLES (Change these manually in your script before you hit run!)
MATERIAL_NAME = "Plastic_Jug"       # Change to "Wood", "Glass", etc.
ORIENTATION   = "Down"   # Change to "Vertical", "Horizontal", etc.
REPETITION    = 1               # Change from 1 through 10 as you repeat the action

if __name__ == "__main__":


    dev_serial_id = "S2508080077"
    sensor = Sensor(dev_serial_id)
    
    # Data Storage for Matplotlib
    time_hist, shear_hist, depth_hist, area_hist, x_edge_hist, y_edge_hist = [], [], [], [], [], []
    
    start_time = time.time()
    is_compressing = False
    contact_start_depth = 0.0
    # Pre-initialize visualizations to avoid NameErrors
    grad_vis = np.zeros((240, 320), dtype=np.uint8)

    # 2. HARDWARE TARE 
    # Ensure nothing is touching the gel right now! 
    print("Taring sensor baseline. Ensure surface is completely untouched...")
    sensor.reset() 
    time.sleep(1.0) # Give the internal CUDA solver a moment to finalize the clear [cite: 268]

    print("Experiment Active. Press 'q' to stop and generate graphs.")

    while True:
        # --- 2. DATA ACQUISITION ---
        img_raw = sensor.getRawImage()
        depth_map = sensor.getDepth()
        shear_map = sensor.getShear() # (H, W, 2)
        curr_time = time.time() - start_time
        black_img = np.zeros((240, 320, 3), dtype=np.uint8)

        depth_smooth = cv2.GaussianBlur(depth_map, (5, 5), 0) # Reduce noise for better contact detection
        # --- 3. CONTACT MASKING (The Foundation) ---
        # Identifying exactly where the object touches the gel
        contact_mask = (depth_smooth > DEPTH_THRESHOLD).astype(np.uint8)
        contact_pixel_count = np.sum(contact_mask)
        contact_area_mm2 = contact_pixel_count * PIXEL_AREA # Convert pixel count to mm^2 using calibration

        # --- 4. MASKED FORCE VECTOR SUMMATION ---
        # We only sum shear vectors inside the contact area 

        # --- 4. CORRECTED FORCE VECTOR AVERAGING ---
       # Isolates the vectors within the mask. Summing across thousands of active pixels 
        # yields the integrated total lateral deformation metric across the footprint.
        masked_shear = shear_map * contact_mask[:, :, np.newaxis] # [cite: 981]
        total_shear_x = np.sum(masked_shear[:, :, 0]) # [cite: 981]
        total_shear_y = np.sum(masked_shear[:, :, 1]) # [cite: 981]
        total_shear_force = np.sqrt(total_shear_x**2 + total_shear_y**2) #

        vector_vis = put_arrows_on_image(black_img, masked_shear * 20) # Scale up by 20 for visibility 
        # --- 5. GRAIN / TEXTURE ANALYSIS (Sobel Derivatives) ---
        # Apply mask to raw image before edge detection
        if len(img_raw.shape) == 3:
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_raw

        
        
        # --- 6. TRIGGER LOGIC FOR STRAIN ---
        # Capturing the 'Initial' state at first meaningful contact
        max_depth = np.max(depth_map)
        if not is_compressing and max_depth > DEPTH_THRESHOLD:
            is_compressing = True
            contact_start_depth = max_depth
            print(f"Contact triggered at {curr_time:.2f}s")

        # Record History
        time_hist.append(curr_time)
        depth_hist.append(max_depth)
        shear_hist.append(total_shear_force)
        area_hist.append(contact_area_mm2)

        # --- 7. VISUALIZATION ---
        cv2.imshow('1. Raw Image', gray)
        cv2.imshow('2. Depth Heatmap', cv2.applyColorMap((depth_smooth*100).astype('uint8'), cv2.COLORMAP_HOT))        
        # Display the live directional field
        cv2.imshow('3. Tangential Shear Vectors', vector_vis)
        
        k = cv2.waitKey(3)
        if k & 0xFF == ord('q'):
            break

    sensor.disconnect()
    cv2.destroyAllWindows()


    # --- 8. POST-EXPERIMENT ANALYSIS (NumPy & Matplotlib) ---
    depth_arr = np.array(depth_hist)
    shear_arr = np.array(shear_hist)
    time_arr = np.array(time_hist)
    
    # Calculate Tnagential Velocity (Derivative) to find when the gel transitions from sticking to slipping (the "dip" in the shear curve)
    velocity = np.diff(shear_hist) / np.diff(time_arr)

    # Calculate decompressesion/ compression slip velocity from the depth
    normal_velocity = np.diff(depth_arr) / np.diff(time_arr)
    
    # Strain Calculation: (Final H - Initial H) / Initial H
    delta_depth = depth_arr[-1] - contact_start_depth
    h_final = H_INITIAL - delta_depth
    final_strain = (h_final - H_INITIAL) / H_INITIAL

    print(f"\nResults Summary:")
    max_active_area = np.max(area_hist)
    min_active_area = np.min(area_hist)
    print(f"Max Contact Area: {max_active_area:.2f} mm^2")
    print(f"Min Contact Area: {min_active_area:.2f} mm^2)")
    print(f"Final deformation of sensor: {delta_depth:.4f}")

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
    
    # Place your text annotation safely pinned underneath Subplot 2
    ax2.text(0.02, 0.05, f"Calculated Final Strain: {delta_depth:.4f}", 
             transform=ax2.transAxes, fontsize=11, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8))
    # Annotate the Strain on the plot for clarity, since it's a key result of the experiment
    plt.figtext(0.15, 0.02, f"Calculated Final Strain: {delta_depth:.4f}",fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

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


    # === AUTOMATIC GRAPH SAVING LOGIC ===
    # 1. Create a "plots" directory if it doesn't exist on your PC yet
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # 2. Build a matching graph filename programmatically
    graph_filename = f"plots/{MATERIAL_NAME}_{ORIENTATION}_rep{REPETITION}_analysis.png"
    
    # 3. Save the crisp high-resolution figure to your PC disk
    plt.savefig(graph_filename, dpi=300)
    print(f"[SUCCESS] Diagnostic plot automatically saved to: {graph_filename}")

    # === AUTOMATIC DATA MATRIX SAVING LOGIC ===
    # Create a "data" directory if it doesn't exist yet
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. Aligns derivatives and shapes your 3-5 second swipe into a fixed 500-step matrix
    # Format and save the numerical sequence array for your Transformer
    #fixed_length_sequence = preprocess_experiment_run(time_hist, depth_hist, shear_hist)
    #matrix_filename = f"data/{MATERIAL_NAME}_{ORIENTATION}_rep{REPETITION}.npy"
    #np.save(matrix_filename, fixed_length_sequence)
    #print(f"[SUCCESS] Time-series data matrix saved to: {matrix_filename}")


    plt.tight_layout()
    plt.show()

