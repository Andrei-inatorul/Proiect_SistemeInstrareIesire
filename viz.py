from pathlib import Path
import numpy as np

import serial
import time

SAMPLE_RATE = 44100
FFT_WINDOW = 1024
MAX_FREQ = 4500

filesDir = Path("./recordings")
alphas = [0.2, 0.2, 0.2, 1.0]
colors = ["red", "green", "blue", "black"]
import matplotlib.pyplot as plt

import pickle

freqs = np.fft.rfftfreq(FFT_WINDOW, 1 / SAMPLE_RATE)
n_bins = np.searchsorted(freqs, MAX_FREQ)

def alpha_filter(data, alpha=0.1):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]
    return filtered_data

def normalize(data):
    data = np.array(data, dtype=np.float64)
    return (data - np.mean(data)) / np.std(data)

def spectru(segment):
    segment = np.array(segment, dtype=np.float64)
    segment = segment - np.mean(segment)
    # pre-emphasis: x[n] - 0.97*x[n-1] amplifica frecventele inalte si face F2 mai vizibil
    segment = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])
    mag = np.abs(np.fft.rfft(segment, n=FFT_WINDOW))[:n_bins]
    return mag / (np.linalg.norm(mag) + 1e-10)

def spectru_medio(sig):
    # media spectrelor din jumatatea de mijloc a inregistrarii (evita tranzitii)
    start = len(sig) // 4
    end = 3 * len(sig) // 4
    specs = []
    for i in range(start, end - FFT_WINDOW, FFT_WINDOW // 2):
        specs.append(spectru(sig[i:i + FFT_WINDOW]))
    return np.mean(specs, axis=0)

# def find_best_template_with_confidence(main_signal, templates, names):
#     scores = []
#
#     # Standardize the main signal once to save processing time
#     sig_norm = (main_signal - np.mean(main_signal)) / np.std(main_signal)
#
#     for template in templates:
#         # Standardize template
#         temp_norm = (template - np.mean(template)) / np.std(template)
#
#         # Cross-correlation
#         correlation = np.correlate(sig_norm, temp_norm, mode='valid') / len(template)
#         scores.append(np.max(correlation))
#
#     scores = np.array(scores)
#     best_idx = np.argmax(scores)
#     best_score = scores[best_idx]
#
#     # Confidence Logic: How much better is the winner than the others?
#     # We use a simple softmax-style normalization or a relative ratio
#     other_scores = np.delete(scores, best_idx)
#     avg_others = np.mean(other_scores)
#
#     # Confidence is the lead the winner has over the average 'noise'
#     confidence = (best_score - avg_others) / (1 - avg_others) * 100
#
#     return {
#         "vowel": names[best_idx],
#         "score": round(best_score, 3),
#         "confidence": f"{max(0, min(100, confidence)):.1f}%"
#     }

def get_template(sig1, sig2, sig3, title="", afis = 4):
    spec1 = spectru_medio(sig1)
    spec2 = spectru_medio(sig2)
    spec3 = spectru_medio(sig3)

    plt.title(f"{title}")
    if afis >= 1:
        plt.plot(freqs[:n_bins], spec1, "r-", alpha=0.5)
    if afis >= 2:
        plt.plot(freqs[:n_bins], spec2, "g-", alpha=0.5)
    if afis >= 3:
        plt.plot(freqs[:n_bins], spec3, "b-", alpha=0.5)

    spec_template = np.mean([spec1, spec2, spec3], axis=0)
    if(afis >= 4):
        plt.plot(freqs[:n_bins], spec_template, "black", alpha=1.0, linewidth=3.0)
    return spec_template

vowels_data = {"A": [], "E": [], "I": [], "O": [], "U": []}
sig = []

for file in sorted(Path(filesDir).rglob('*.txt')):
    fn = file.name.removesuffix(".txt")
    temp = []
    with file.open() as txtFile:
        txt = txtFile.read()
        txt = txt.split(", ")
        txt.pop(-1)
        temp.extend(map(int, txt))
    litera = fn[0]
    if litera in vowels_data and len(fn) == 2 and fn[1].isdigit():
        vowels_data[litera].append(temp)
    elif fn == "sig":
        sig = temp

A, E, I, O, U = [vowels_data[v] for v in "AEIOU"]

signals = [A, E, I, O, U]
plt.figure()

plt.subplot(6, 1, 1)
A_template = get_template(*A, "A")
plt.xlabel("Frecventa (Hz)")
plt.subplot(6, 1, 2)
E_template = get_template(*E, "E")
plt.xlabel("Frecventa (Hz)")
plt.subplot(6, 1, 3)
I_template = get_template(*I, "I")
plt.xlabel("Frecventa (Hz)")
plt.subplot(6, 1, 4)
O_template = get_template(*O, "O")
plt.xlabel("Frecventa (Hz)")
plt.subplot(6, 1, 5)
U_template = get_template(*U, "U")
plt.xlabel("Frecventa (Hz)")
plt.subplot(6, 1, 6)

vocala = ["A", "E", "I", "O", "U"]
templates = [A_template, E_template, I_template, O_template, U_template]

# normalizeaza per banda: fiecare bin FFT e impartit la media lui peste toate templateurile
# reduce dominanta benzilor cu energie mare (joase frecvente) fata de formante
band_mean = np.mean(templates, axis=0)
band_mean[band_mean == 0] = 1e-10
templates = [t / band_mean for t in templates]
templates = [t / (np.linalg.norm(t) + 1e-10) for t in templates]

with open('templates.pkl', 'wb') as file:
    pickle.dump((templates, band_mean), file)

sig_array = np.array(sig, dtype=np.float64)
scores = []

if len(sig_array) >= FFT_WINDOW:
    step = 256
    windows = np.lib.stride_tricks.sliding_window_view(sig_array, FFT_WINDOW)[::step]

    # eliminare DC si FFT pe toate ferestrele odata
    specs = np.abs(np.fft.rfft(windows - np.mean(windows, axis=1, keepdims=True), axis=1))[:, :n_bins]
    # aplica aceeasi normalizare per banda ca la template-uri
    specs = specs / band_mean
    norms = np.linalg.norm(specs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    specs_norm = specs / norms

    for i, t in enumerate(templates):
        # similaritate cosinus fata de template spectral
        corrs = specs_norm @ t
        max_val = np.max(corrs)
        scores.append(max_val)
        print(f"Vocala {vocala[i]}: Similaritate maxima = {max_val:.4f}")

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    other_scores = np.delete(scores, best_idx)
    avg_others = np.mean(other_scores)

    print("-" * 30)
    print(f"Predictie: {vocala[best_idx]}")
    print("-" * 30)

plt.show()
