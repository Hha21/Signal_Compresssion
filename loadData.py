import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile 

folder_path = '.\data\data_neuralink'

global_max = -np.inf
global_min = np.inf
threshold = 7142
exceedance_counter = 0
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        print('Processing: ' + filename)
        file_path = os.path.join(folder_path, filename)
        sampling_rate, data = wavfile.read(file_path)

        global_max = max(global_max, np.max(data))
        global_min = min(global_min, np.min(data))

        if data.dtype != 'int16':
            print(f'Data is {data.dtype}')

        if np.max(data) >= threshold:
            exceedance_counter += 1
        # all_data.extend(data)
        
# all_data = np.array(all_data)
# print(f'All Data Shape is {all_data.shape}')
# print(f'99th percentile: {np.percentile(all_data, 99)}')

data = np.clip(data, -threshold, threshold)
data = data.astype(np.float32) / threshold

print(f'Number of times exceeded: {exceedance_counter}')
print(f'Sampling Rate: {sampling_rate} Hz')
print(f'Data Shape: {data.shape}')
print(f'Data Type: {data.dtype}')

print (f'Max of all Data: {global_max}')
print(f'Min of all Data: {global_min}')

print(f"Scaled data (first 10 points): {data[:10]}")
print(f"Scaled data range: {data.min()} to {data.max()}")


ms_per_sample = 1000 / sampling_rate
time_range_ms = 50
samples_to_plot = int(time_range_ms / ms_per_sample)
time_axis_ms = np.linspace(0, time_range_ms, samples_to_plot)

plt.figure(figsize=(12, 6))
if data.ndim == 1: 
    plt.plot(np.linspace(0, len(data) / sampling_rate, num=len(data)), data)
    plt.title('Sample Neuro Signal')

plt.xlabel('Time [s]')
plt.ylabel('Normalised Amplitude')
plt.grid(True)

plt.figure(figsize=(12, 6))
plt.plot(time_axis_ms, data[:samples_to_plot])
plt.title('Signal on Millisecond Timescale')
plt.xlabel('Time [ms]')
plt.ylabel('Normalized Amplitude')
plt.grid(True)
plt.show()

plt.show()
