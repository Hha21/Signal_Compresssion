import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import wave
import random


# Function to load a .wav file and convert it to 10-bit integers
def load_wav_10bit(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get file parameters
        params = wav_file.getparams()
        n_channels = params.nchannels
        sampwidth = params.sampwidth
        framerate = params.framerate
        n_frames = params.nframes

        # Read frames and convert to numpy array
        frames = wav_file.readframes(n_frames)
        signal = np.frombuffer(frames, dtype=np.int16)  # Assuming 16-bit format

        # Convert to 10-bit by right-shifting to drop 6 least significant bits
        signal_10bit = np.right_shift(signal, 6)  # To get 10-bit equivalent
        return signal_10bit, framerate

# Function to apply a low-pass filter to the signal
def low_pass_filter(signal, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    # Get the butterworth filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def spike_pass_filter(signal, cutoff_freq, sample_rate, order=3):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    # Get the butterworth filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def band_pass_filter(signal, lowcut, highcut, sample_rate, order=3):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    # Get the butterworth filter coefficients
    b, a = butter(order, [low, high], btype='band', analog=False)
    # Apply the filter
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def plot_random_segment(signal, filtered_signal, sample_rate, segment_ms=50):
    # Calculate number of samples for the segment
    segment_samples = int((segment_ms / 1000) * sample_rate)

    # Get a random start point for the 50ms segment
    max_start_index = len(signal) - segment_samples
    start_index = random.randint(0, max_start_index)
    end_index = start_index + segment_samples

    # Extract segment from original and filtered signals
    segment_signal = signal[start_index:end_index]

    segment_filtered = filtered_signal[start_index:end_index]
    segment_spikes = segment_signal - segment_filtered

    band_passed_spikes = band_pass_filter(segment_spikes, lowcut, highcut, framerate)

    # filtered_spikes = low_pass_filter(segment_spikes, spike_cutoff_frequency, framerate)

    # Plotting the signals
    time = np.linspace(0, segment_ms, num=segment_samples)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, segment_signal, label='Original Signal (10-bit)')
    plt.plot(time, segment_filtered, label='Filtered Signal (Mean)', linestyle='--')
    plt.legend()
    plt.title('Original Signal and Filtered Mean Signal (Random 50ms Segment)')

    plt.subplot(2, 1, 2)
    plt.plot(time, segment_spikes, label='Spikes (Original)', linestyle = '--')
    plt.plot(time,band_passed_spikes, label='Spikes (Filtered)', color='r')
    plt.legend()
    plt.title('Spikes (Difference between Original and Mean Signal) - Random 50ms Segment')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')

    plt.grid()
    plt.tight_layout()
    plt.show()


# Path to your .wav file
file_path = r'.\data\data_neuralink\0b049a37-dc68-42f6-bbe5-d7d1cd699ccb.wav'

# Load the data and filter it
signal_10bit, framerate = load_wav_10bit(file_path)
cutoff_frequency = 1000

spike_cutoff_frequency = 5000

lowcut = 300  # 300 Hz
highcut = 3000  # 3 kHz

# Apply low-pass filter to get the mean signal
filtered_signal = low_pass_filter(signal_10bit, cutoff_frequency, framerate)

plot_random_segment(signal_10bit, filtered_signal, framerate)