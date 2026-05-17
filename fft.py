import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

FILE_NAME = './rec/A.txt'
SAMPLE_RATE = 44100

def main():
    try:
        with open(FILE_NAME, 'r') as file:
            data = np.array([float(x) for x in file.read().replace(',', ' ').split() if x.strip()])
    except FileNotFoundError:
        return

    data = data[:44100]
    data = data - np.mean(data)
    N = len(data)

    yf = fft(data)
    xf = fftfreq(N, 1 / SAMPLE_RATE)[:N // 2]
    magnitude = (2.0 / N) * np.abs(yf[:N // 2])

    peaks, _ = find_peaks(magnitude, distance=int(100 / (SAMPLE_RATE / N)))
    valid_peaks = [p for p in peaks if 0 <= xf[p] <= 4500]
    valid_peaks.sort(key=lambda p: magnitude[p], reverse=True)
    top_4_indices = sorted(valid_peaks[:4])

    top_4_freqs = xf[top_4_indices]
    top_4_mags = magnitude[top_4_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(xf, magnitude, color='blue', label='FFT')
    plt.plot(top_4_freqs, top_4_mags, "rx", markersize=10, label='4 Formante(?)')

    for f, m in zip(top_4_freqs, top_4_mags):
        plt.annotate(f'{int(f)}Hz', (f, m), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title("FFT")
    plt.xlabel("Frecventa (Hz)")
    plt.ylabel("Amplitudine")
    plt.xlim(0, 5000)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()