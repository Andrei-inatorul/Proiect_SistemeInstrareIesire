from pathlib import Path
import numpy as np

import serial
import time

filesDir = Path("./rec")
alphas = [0.2, 0.2, 0.2, 1.0]
colors = ["red", "green", "blue", "black"]
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

A = []
E = []
I = []
O = []
U = []
sig = []

for file in Path(filesDir).rglob('*.txt'):
    fn = file.name.removesuffix(".txt")
    litera = fn[0]
    temp = []
    with file.open() as txtFile:
        txt = txtFile.read()
        txt = txt.split(", ")
        txt.pop(-1)
        temp.extend(map(int, txt))
    if litera == "A":
        A.extend(temp)
    elif litera == "E":
        E.extend(temp)
    elif litera == "I":
        I.extend(temp)
    elif litera == "O":
        O.extend(temp)
    elif litera == "U":
        U.extend(temp)
    else:
        sig = temp

signals = [A, E, I, O, U]
plt.figure()

plt.subplot(6, 1, 1)
A_template = get_template(A, 5670+150, 10022, 24739, 419, "A")
plt.subplot(6, 1, 2)
E_template = get_template(E, 5270, 10110, 20214, 385, "E")
plt.subplot(6, 1, 3)
I_template = get_template(I, 5270+61, 10110+150+97, 20214+241+97, 340, "I")
plt.subplot(6, 1, 4)
O_template = get_template(O, 5670+150+215, 10022+150+92, 24739+150, 390, "O")
plt.subplot(6, 1, 5)
U_template = get_template(U, 5670+270, 10022, 24739, 373, "U")
plt.subplot(6, 1, 6)

vocala = ["A", "E", "I", "O", "U"]
templates = [A_template, E_template, I_template, O_template, U_template]

with open('templates.pkl', 'wb') as file:
    pickle.dump(templates, file)

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

    # calculare corelatie pt toate ferestrele
    corrs = np.dot(windows_norm, t_norm) / window_size

    max_val = np.max(np.abs(corrs))
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