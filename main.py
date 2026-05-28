import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dmrobotics import Sensor, put_arrows_on_image

# --- 1. CONFIGURATION & CALIBRATION ---
# These constants are derived from your sensor datasheet or manual calibration
PIXEL_TO_MM = 20       # pixels per mm
PIXEL_AREA = (16.0 * 12.0 / (320 * 240))   # mm^2 per pixel (calibrated from a known reference object) [cite: 261] 
GEL_THICKNESS = 20.1      # L in mm
DEPTH_THRESHOLD = 0.015  # Noise floor for contact detection
H_INITIAL = 10.0         # Physical height of your test object

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

    # 2. HARDWARE TARE (CRITICAL PLACE TO ADD THIS) [cite: 262, 743]
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

        
        # Derivatives to find grain orientation

        # 1. Math: Derivatives to find grain orientation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 2. Connection: Combine X and Y into a Magnitude Map for visualization
        # This is the step that was missing
        grad_mag = cv2.magnitude(grad_x, grad_y)

        # Only consider pixels where the gradient is 'strong' enough to be a grain, TRAIL and ERROR
        MAG_THRESHOLD = 0
        # This creates a binary map: 255 where grain is strong, 0 elsewhere
        _, grain_mask = cv2.threshold(grad_mag, MAG_THRESHOLD, 255, cv2.THRESH_BINARY)
        grain_mask = grain_mask.astype(np.uint8)

        masked_gray = cv2.bitwise_and(gray, gray, mask=contact_mask)
        # 4. Count Directional Edges
        # We only count gradients where the grain_mask is active
        # Use np.abs because an edge can be a transition from light-to-dark or dark-to-light
        edge_count_x = np.sum((np.abs(grad_x) > MAG_THRESHOLD) & (masked_gray > 0))
        edge_count_y = np.sum((np.abs(grad_y) > MAG_THRESHOLD) & (masked_gray > 0))

        # Record edge counts for later analysis
        x_edge_hist.append(edge_count_x)
        y_edge_hist.append(edge_count_y)

        # 4. Calculation: Edge density for your material analysis
        edge_density = np.sum(np.abs(grad_x) + np.abs(grad_y))

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
        cv2.imshow('2. Gradient Map (Grains)', grad_vis)
        cv2.imshow('3. Depth Heatmap', cv2.applyColorMap((depth_smooth*100).astype('uint8'), cv2.COLORMAP_HOT))        
        # Display the live directional field
        cv2.imshow('4. Tangential Shear Vectors', vector_vis)
        
        k = cv2.waitKey(3)
        if k & 0xFF == ord('q'):
            break

    sensor.disconnect()
    cv2.destroyAllWindows()

    # --- 8. POST-EXPERIMENT ANALYSIS (NumPy & Matplotlib) ---
    depth_arr = np.array(depth_hist)
    time_arr = np.array(time_hist)
    
    # Calculate Velocity (Derivative) to find the mechanical 'Dip'
    velocity = np.diff(depth_arr) / np.diff(time_arr)
    
    # Strain Calculation: (Final H - Initial H) / Initial H
    delta_depth = depth_arr[-1] - contact_start_depth
    h_final = H_INITIAL - delta_depth
    final_strain = (h_final - H_INITIAL) / H_INITIAL

    print(f"\nResults Summary:")
    max_active_area = np.max(area_hist)
    min_active_area = np.min(area_hist)
    print(f"Max Contact Area: {max_active_area:.2f} mm^2")
    print(f"Min Contact Area: {min_active_area:.2f} mm^2)")
    print(f"Final Strain: {final_strain:.4f}")

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
    # FIXED: This was previously overlapping or skipping, leaving a blank graph!
    ax2.plot(time_arr, np.array(shear_hist), color='red', linewidth=2)
    ax2.set_title("Total Integrated Shear Displacement over Time", fontsize=14)
    ax2.set_ylabel("Shear Volumetric Integral (mm)", color='red', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Place your text annotation safely pinned underneath Subplot 2
    ax2.text(0.02, 0.05, f"Calculated Final Strain: {final_strain:.4f}", 
             transform=ax2.transAxes, fontsize=11, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8))
    # Annotate the Strain on the plot for clarity, since it's a key result of the experiment
    plt.figtext(0.15, 0.02, f"Calculated Final Strain: {final_strain:.4f}",fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    #Velocity Profile (Dip Detection)
    ax3.plot(time_arr[1:], velocity, color='green')
    ax3.set_title("Dip Detection - For velocity changes in shear map", fontsize=14)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Velocity (mm/s)", color='green', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    #Topology Profile (Contact Area)
    ax4.plot(time_arr, x_edge_hist, color='purple', linewidth=2, label='Vertical Grains (X)')
    ax4.plot(time_arr, y_edge_hist, color='orange', linewidth=2, label='Horizontal Grains (Y)')
    ax4.set_title("Grain Analysis (Directional Edge Counts)", fontsize=14) [cite: 992]
    ax4.set_xlabel("Time (s)", fontsize=12) [cite: 992]
    ax4.set_ylabel("Number of Edges", fontsize=12) [cite: 992]
    ax4.legend(loc='upper left')
    ax4.grid(True, linestyle='--', alpha=0.5) [cite: 992]

    plt.tight_layout()
    plt.show()