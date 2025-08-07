import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import os

# load folder with photothermal speckle images 
def load_data(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img.astype(np.float64))
    return np.array(images)

# clculate Fourier Transform and remove DC componenet
def calculate_fft(images, fs, zero_pad_factor = 2):
    h, w, N = images.shape[1], images.shape[2], images.shape[0]
    padded_N = N * zero_pad_factor
    fft_result = np.zeros(h, w, padded_N//2)
    freqs = np.linspace(0, fs/2, padded_N//2) 

    for i in range(h):
        for j in range(w):
            intensity = images[:, i, j]
            intensity = -np.mean(intensity)
            padded_signal = np.pad(intensity, (0, padded_N - N), 'constant')
            freq_range = fft(padded_signal)
            fft_result[i, j, :] = np.abs(freq_range[:padded_N//2])

    return fft_result, freqs
def plot_fft(fft_result, freqs):
    avg_signal = np.mean(fft_result, axis=(0,1))
    plt.figure(figsize=(10,5))
    plt.plot(freqs, avg_signal)
    plt.title("Fourier Transform of Photothermal Speckle Images")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

def speckle_analysis(folder, fs, zero_pad_factor = 2):
    images = load_data(folder)
    if images.size == 0:
        print("No images")
        return

    fft_result, freqs = calculate_fft(images, fs, zero_pad_factor)
    plot_fft(fft_result, freqs)

speckle_analysis("folder_path")