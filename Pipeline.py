from json import load
import numpy as np 
import os
from scipy.io import wavfile 
import pysindy as ps

# Process entire dataset into a single array:

def load_and_normalise():
    folder_path = '.\data\data_neuralink'
    signal_threshold = 7142

    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            print('Processing: ' + filename)
            file_path = os.path.join(folder_path, filename)
            sampling_rate, data = wavfile.read(file_path)
            all_data.extend(data)

    all_data = np.clip(all_data, -signal_threshold, signal_threshold)
    all_data = all_data.astype(np.float32) / signal_threshold

    print(f'Shape is: {all_data.shape}')
    print(f'Sampling Rate: {sampling_rate}')

    return all_data, sampling_rate

def generate_windows(signal_data, window_size, step_size):
    windows = [
        signal_data[i:i + window_size]
        for i in range(0, len(signal_data) - window_size + 1, step_size)
    ]
    return np.array(windows)


def compute_derivatives(data, dt, method = 'smoothed_finite_differences', use_ddx = True):

    if method == 'smooted_finite_differences':
        diff_method = ps.SmoothedFiniteDifference()
    elif method == 'savitzky_golay':
        method = ps.SavitzkyGolay()
    elif method == 'spectral':
        diff_method = ps.SpectralDerivative()
    else:
        diff_method = ps.SmoothedFiniteDifference()

    dx = diff_method._differentiate(data, dt)

    if use_ddx:
        ddx = diff_method._differentiate(dx, dt)
        return dx, ddx
    else:
        return dx
    
def data_pipeline(window_size = 1000, step_size = 500):

    data, sample_rate = load_and_normalise()
    dt = 1.0 / sample_rate
    windows = generate_windows(data, window_size, step_size)
    print(f'Windows Shape: {windows.shape}')
    # print(f'Window 1: {windows[0,:]}')
    # print(f'Window 2: {windows[1,:]}')
    i = 0
    for window in windows:
        i += 1
        print(f'Computing dx: {i} of {windows.shape[0]}')
        dx = compute_derivatives(window.reshape(-1,1), dt, method='smoothed_finite_differences', use_ddx=False)
    
    print(f'dx shape: {dx.shape}')
    print(dx)



    

data_pipeline(window_size=500, step_size=250)
