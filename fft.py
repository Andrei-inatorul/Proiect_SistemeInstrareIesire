import numpy as np
import matplotlib.pyplot as plt
import math

import serial
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

SERIAL_PORT = "COM4"
FILE_NAME = './rec/E.txt'
SAMPLE_RATE = 44100
BAUD_RATE = 115200

CENTROIZI_VOCALE = {
    'A': (700, 1200),
    'E': (450, 1800),
    'I': (300, 2200),
    'O': (450, 850),
    'U': (300, 650)
}

def detecteaza_vocala(f1, f2):
    min_dist = float('inf')
    best_vowel = 'Necunoscut'

    for vowel, (cent_f1, cent_f2) in CENTROIZI_VOCALE.items():
        # distanta euclidiana modificata
        # (f1 - cent_f1) ** 2  -> cat de departe ne aflam de F1 ideal
        # ((f2 - f1) - (cent_f2 - cent_f1)) ** 2) - > distanta dintre F1 si F2
        # inmultim cu 1.5 pt ca vrem sa fie mai importanta o diferenta de distanta  mai mica decat ca F1 sa fie pozitionat perfect
        dist = math.sqrt((f1 - cent_f1) ** 2 + 1.5 * ((f2 - f1) - (cent_f2 - cent_f1)) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_vowel = vowel

    return best_vowel


def get_max_in_range(freqs, mags, min_f, max_f):
    #Cautam maximul din intervalul specificat
    valid_idx = np.where((freqs >= min_f) & (freqs <= max_f))[0]
    if len(valid_idx) == 0:
        return None, None
    best_idx = valid_idx[np.argmax(mags[valid_idx])]
    return freqs[best_idx], mags[best_idx]


def detectie_vocala(data):
    # try:
    #     with open(FILE_NAME, 'r') as file:
    #         data = np.array([float(x) for x in file.read().replace(',', ' ').split() if x.strip()])
    # except FileNotFoundError:
    #     print("Fisierul nu a fost gasit.")
    #     return

    data = data[:44100]
    data = data - np.mean(data)

    # Ridicam frecventele inalte pt a fi mai observabile
    data = np.append(data[0], data[1:] - 0.95 * data[:-1])

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / SAMPLE_RATE)[:N // 2]
    magnitude = (2.0 / N) * np.abs(yf[:N // 2])
    freq_resolution = SAMPLE_RATE / N

    # gasim armonicele
    harmonic_peaks, _ = find_peaks(magnitude, distance=int(80 / freq_resolution), prominence=2)
    hx = xf[harmonic_peaks]
    hy = magnitude[harmonic_peaks]

    valid_formants = []

    # Cautam F1, F2, F3
    # F1 in zona 250 - 900 hz
    f1_freq, f1_mag = get_max_in_range(hx, hy, 250, 900)

    if f1_freq is not None:
        valid_formants.append((f1_freq, f1_mag))

        #cautam F2 (F1+200 hz, 2500Hz)
        f2_min = max(f1_freq + 200, 600)
        f2_freq, f2_mag = get_max_in_range(hx, hy, f2_min, 2500)

        if f2_freq is not None:
            valid_formants.append((f2_freq, f2_mag))

            # cautam f3 (F2+300 hz, 3500 hz)
            f3_min = max(f2_freq + 300, 1500)
            f3_freq, f3_mag = get_max_in_range(hx, hy, f3_min, 3500)

            if f3_freq is not None:
                valid_formants.append((f3_freq, f3_mag))

    detected_vowel = "Nu am putut detecta"

    if len(valid_formants) >= 2:
        f1_val = valid_formants[0][0]
        f2_val = valid_formants[1][0]

        detected_vowel = detecteaza_vocala(f1_val, f2_val)

        print("-" * 30)
        print(f"F1 extras: {f1_val:.0f} Hz")
        print(f"F2 extras: {f2_val:.0f} Hz")
        if len(valid_formants) >= 3:
            print(f"F3 extras: {valid_formants[2][0]:.0f} Hz")
        print(f"Distanța F2 - F1: {f2_val - f1_val:.0f} Hz")
        print(f"VOCALA ESTIMATĂ: {detected_vowel}")
        print("-" * 30)

    #grafic
    plt.figure(figsize=(11, 6))

    plt.plot(xf, magnitude, color='blue', alpha=0.3, label='FFT cu amplificare a frecventelor inalte')
    plt.plot(hx, hy, 'k.', markersize=4, alpha=0.5, label='Armonice detectate')

    colors = ['red', 'green', 'purple']
    labels = ['F1', 'F2', 'F3']

    for i, (f_freq, f_mag) in enumerate(valid_formants):
        plt.axvline(x=f_freq, color=colors[i], linestyle='--', linewidth=2,
                    label=f'{labels[i]} (~{int(f_freq)} Hz)')
        plt.plot(f_freq, f_mag, marker='o', markersize=8, color=colors[i])

    plt.title(f"Vocala detectata: '{detected_vowel}'")
    plt.xlabel("Frecvența")
    plt.ylabel("Amplitudine")
    plt.xlim(0, 3500)
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()
    return detected_vowel




if __name__ == "__main__":
    sig = []

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Nu s-a putut face conexiuntea: {e}")
        exit()

    recording = False
    print("Astept inregistrare de la placa...")


    while True:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()

                if line == "START_COM":
                    print("Primesc date...")
                    sig = []
                    recording = True

                elif line == "STOP_COM":
                    print(f"Transfer complet!")
                    recording = False
                    break

                elif recording:
                    try:
                        sig.append(int(line))
                    except ValueError:
                        pass

        if sig == []:
            continue
        plt.title("Forma de unda fragment de inregistrare")
        plt.plot(sig[10000:10700])
        plt.show()

        vocala = detectie_vocala(sig)
        msg = f"Detectat: {vocala}\n".encode("ascii")
        ser.write(msg)