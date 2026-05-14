import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dmrobotics import Sensor, put_arrows_on_image

# --- 1. CONFIGURATION & CALIBRATION ---
# These constants are derived from your sensor datasheet or manual calibration
PIXEL_TO_MM = 20       # pixels per mm
PIXEL_AREA = PIXEL_TO_MM**2 
GEL_THICKNESS = 20.1      # L in mm
DEPTH_THRESHOLD = 0.02   # Noise floor for contact detection
H_INITIAL = 10.0         # Physical height of your test object

if __name__ == "__main__":
    dev_serial_id = "S2508080042"
    sensor = Sensor(dev_serial_id)
    
    # Data Storage for Matplotlib
    time_hist, shear_hist, depth_hist, area_hist = [], [], [], []
    
    start_time = time.time()
    is_compressing = False
    contact_start_depth = 0.0

    print("Experiment Active. Press 'q' to stop and generate graphs.")

    while True:
        # --- 2. DATA ACQUISITION ---
        img_raw = sensor.getRawImage()
        depth_map = sensor.getDepth()
        shear_map = sensor.getShear() # (H, W, 2)
        curr_time = time.time() - start_time

        depth_smooth = cv2.GaussianBlur(depth_map, (5, 5), 0) # Reduce noise for better contact detection
        # --- 3. CONTACT MASKING (The Foundation) ---
        # Identifying exactly where the object touches the gel
        contact_mask = (depth_smooth > DEPTH_THRESHOLD).astype(np.uint8)
        contact_pixel_count = np.sum(contact_mask)
        contact_area_mm2 = contact_pixel_count * PIXEL_AREA # Convert pixel count to mm^2 using calibration

        # --- 4. MASKED FORCE VECTOR SUMMATION ---
        # We only sum shear vectors inside the contact area 
        masked_shear = shear_map * contact_mask[:, :, np.newaxis]
        total_shear_x = np.sum(masked_shear[:, :, 0])
        total_shear_y = np.sum(masked_shear[:, :, 1])
        total_shear_force = np.sqrt(total_shear_x**2 + total_shear_y**2)

        # --- 5. GRAIN / TEXTURE ANALYSIS (Sobel Derivatives) ---
        # Apply mask to raw image before edge detection
        if len(img_raw.shape) == 3:
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_raw

        masked_gray = cv2.bitwise_and(gray, gray, mask=contact_mask)
        
        # Derivatives to find grain orientation
        grad_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
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
    print(f"Total Area: {area_hist[-1]:.2f} mm^2")
    print(f"Final Strain: {final_strain:.4f}")

    # Plotting
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    #Identation Profile (from Depth Map)
    ax1.plot(time_arr, depth_arr, color='blue', linewidth=2)
    ax1.set_ylabel("Depth (mm)", color = 'blue', fontsize=12)
    ax1.set_title("Depth (Object identation) Profile over Time", fontsize =14)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.plot(time_arr, np.array(shear_hist)/10, label="Shear (Scaled)", color='red')
    ax1.set_title("Force Profile over Time", fontsize=14)
    ax1.legend()

    #Shear Force Profile
    ax2.plot(time_arr, np.array(shear_hist), color='red', linewidth=2)
    ax2.set_title("Shear Force Profile over Time", fontsize=14)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Shear Force (N)", color='red', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    

    # Annotate the Strain on the plot for your report
    plt.figtext(0.15, 0.02, f"Calculated Final Strain: {final_strain:.4f}",fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    #Velocity Profile (Dip Detection)
    ax3.plot(time_arr[1:], velocity, color='green')
    ax3.set_title("Dip Detection - For velocity changes in shear map", fontsize=14)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Velocity (mm/s)", color='green', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()