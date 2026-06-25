import numpy as np

# === DEFINE THE PREPROCESSING FUNCTION HERE ===
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
    multivariate_matrix = np.column_stack((depth_aligned, shear_aligned, normal_velocity, shear_slip_velocity))
    
    # Interpolate/Resize to a fixed length (target_len) so the Transformer layers match
    current_len = multivariate_matrix.shape[0]
    indices = np.linspace(0, current_len - 1, target_len).astype(int)
    fixed_length_sequence = multivariate_matrix[indices]
    
    return fixed_length_sequence