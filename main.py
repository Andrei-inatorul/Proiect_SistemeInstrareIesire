from pathlib import Path
import numpy as np

import serial
import time

SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

filesDir = Path("./rec")
# alphas = [0.2, 0.2, 0.2, 1.0]
# colors = ["red", "green", "blue", "black"]
import matplotlib.pyplot as plt

import pickle

def alpha_filter(data, alpha=0.1):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]
    return filtered_data

def normalize(data):
    data = np.array(data, dtype=np.float64)
    return (data - np.mean(data)) / np.std(data)

def get_template(sig, s1, s2, s3, per, title="", afis = 4):
    sig1 = sig[s1:s1 + per]
    sig2 = sig[s2:s2 + per]
    sig3 = sig[s3:s3 + per]

    sig1 = normalize(sig1)
    sig2 = normalize(sig2)
    sig3 = normalize(sig3)

    plt.title(f"{title}")
    if afis >= 1:
        plt.plot(sig1, "r-", alpha=0.5)
    if afis >= 2:
        plt.plot(sig2, "g-", alpha=0.5)
    if afis >= 3:
        plt.plot(sig3, "b-", alpha=0.5)

    sig_template = np.mean([sig1, sig2, sig3], axis=0)
    sig_template = alpha_filter(sig_template, 0.5)
    if(afis >= 4):
        plt.plot(sig_template, "black", alpha=1.0, linewidth=3.0)
    return sig_template
ser = None

with open('templates.pkl', 'rb') as file:
    templates = pickle.load(file)

vocala = ["A", "E", "I", "O", "U"]

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except Exception as e:
    print(f"Nu s-a putut face conexiuntea: {e}")
    exit()

fig, axs = plt.subplots(5, 1, figsize=(10, 12))
for i in range(5):
    axs[i].plot(templates[i], color='black')
    axs[i].set_title(f"Template: {vocala[i]}", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

while True:
    sig = []
    recording = False
    print("Astept inregistrare de la placa...")
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
    plt.title("Forma de unda fragment de inregistrare")
    plt.plot(sig[10000:10700])
    plt.show()
    sig_array = np.array(sig, dtype=np.float64)
    scores = []

    for i, t in enumerate(templates):
        t_norm = (t - np.mean(t)) / np.std(t)
        window_size = len(t_norm)

        if len(sig_array) < window_size:
            scores.append(0)
            continue

        # fereastra glisanta
        windows = np.lib.stride_tricks.sliding_window_view(sig_array, window_size)

        # normalizare locala
        windows_mean = np.mean(windows, axis=1, keepdims=True)
        windows_std = np.std(windows, axis=1, keepdims=True)
        windows_std[windows_std == 0] = 1e-10
        windows_norm = (windows - windows_mean) / windows_std

        # calculare corelatie (corelare?) pt toate ferestrele
        corrs = np.dot(windows_norm, t_norm) / window_size

        max_val = np.max(corrs)
        scores.append(max_val)
        print(f"Vocala {vocala[i]}: Similaritate maxima = {max_val:.4f}")

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    print("-" * 30)
    print(f"Predictie: {vocala[best_idx]}")
    msg = f"{vocala[best_idx]}\n".encode("ascii")
    ser.write(msg)
    print("-" * 30)
