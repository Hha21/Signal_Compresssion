from json import load
import numpy as np 
import os
from scipy.io import wavfile 
import pysindy as ps

# Process entire dataset into a single array:

def load_and_normalise(file_path):
    signal_threshold = 7142

    print('Processing: ' + file_path)
    sampling_rate, data = wavfile.read(file_path)
    data = np.clip(data, -signal_threshold, signal_threshold)
    data = data.astype(np.float32) / signal_threshold
    return data, sampling_rate

def generate_windows(signal_data, window_size, step_size):
    windows = [
        signal_data[i:i + window_size]
        for i in range(0, len(signal_data) - window_size + 1, step_size)
    ]
    return np.array(windows)

def compute_derivatives(data, dt, method = 'smoothed_finite_differences'):

    if method == 'smooted_finite_differences':
        diff_method = ps.SmoothedFiniteDifference()
    elif method == 'savitzky_golay':
        method = ps.SavitzkyGolay()
    elif method == 'spectral':
        diff_method = ps.SpectralDerivative()
    else:
        diff_method = ps.SmoothedFiniteDifference()

    dx = diff_method._differentiate(data, dt)
    ddx = diff_method._differentiate(dx, dt)
    
    return dx, ddx
    
def data_pipeline(window_size = 1000, step_size = 500, method = 'smoothed_finite_differences', batch_size = 5, val_split=0.2):

    folder_path = '.\data\data_neuralink'

    all_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    n_batches = len(all_files) // batch_size + (len(all_files) % batch_size != 0)

    for batch_idx in range(n_batches):

        print(f'Processing batch {batch_idx + 1} of {n_batches}')
        batch_files = all_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        batch_data = {'t': [], 'x': [], 'dx': [], 'ddx': []}

        for filename in batch_files:

            file_path = os.path.join(folder_path, filename)
            raw_data, sample_rate = load_and_normalise(file_path)
            dt = 1.0 / sample_rate
            windows = generate_windows(raw_data, window_size, step_size)
            time_array = np.arange(window_size) * dt
            print(f'Windows Shape: {windows.shape}')

            dx, ddx = [], []
            
            for window in windows:
                dx_window, ddx_window = compute_derivatives(window.reshape(-1, 1), dt, method)
                dx.append(dx_window.reshape(-1))
                ddx.append(ddx_window.reshape(-1))

                
            batch_data['t'].extend([time_array] * len(windows))
            batch_data['x'].extend(windows)
            batch_data['dx'].extend(dx)
            batch_data['ddx'].extend(ddx)
        
        for key in batch_data:
            batch_data[key] = np.array(batch_data[key])
        
        n_examples = batch_data['x'].shape[0]
        n_train = int((1 - val_split) * n_examples)

        training_data = {key: batch_data[key][:n_train] for key in batch_data}
        validation_data = {key: batch_data[key][n_train:] for key in batch_data}

        print(f"Batch {batch_idx + 1} shapes - training: x: {training_data['x'].shape}, validation: x: {validation_data['x'].shape}")
        
        yield training_data, validation_data

# for batch_data in data_pipeline(window_size=1000, step_size=500, method='smoothed_finite_differences', batch_size=150):
    
#     print(f'Processed a batch with data shapes - t: {batch_data["t"].shape} x: {batch_data["x"].shape}, dx: {batch_data["dx"].shape}, ddx: {batch_data["ddx"].shape}')
