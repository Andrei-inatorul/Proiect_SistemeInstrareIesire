import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

SAMPLE_RATE = 44100
FFT_WINDOW = 1024
MAX_FREQ = 4500

freqs = np.fft.rfftfreq(FFT_WINDOW, 1 / SAMPLE_RATE)
n_bins = np.searchsorted(freqs, MAX_FREQ)

vocala = ["A", "E", "I", "O", "U"]

with open('templates.pkl', 'rb') as f:
    templates, band_mean = pickle.load(f)

def clasifica(fisier):
    with open(fisier) as f:
        txt = f.read().split(", ")
        txt = [x for x in txt if x.strip()]
        sig_array = np.array(list(map(float, txt)), dtype=np.float64)

    if len(sig_array) < FFT_WINDOW:
        print(f"{fisier}: semnal prea scurt!")
        return

    step = 256
    windows = np.lib.stride_tricks.sliding_window_view(sig_array, FFT_WINDOW)[::step]

    # eliminare DC, pre-emphasis si FFT pe toate ferestrele odata
    # pre-emphasis: x[n] - 0.97*x[n-1] amplifica frecventele inalte si face F2 mai vizibil
    windows_dc = windows - np.mean(windows, axis=1, keepdims=True)
    windows_pe = np.hstack([windows_dc[:, :1], windows_dc[:, 1:] - 0.97 * windows_dc[:, :-1]])
    specs = np.abs(np.fft.rfft(windows_pe, axis=1))[:, :n_bins]
    specs = specs / band_mean
    norms = np.linalg.norm(specs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    specs_norm = specs / norms

    scores = []
    for t in templates:
        corrs = specs_norm @ t
        scores.append(np.max(corrs))

    best_idx = np.argmax(scores)
    print(f"{Path(fisier).name}: {[f'{vocala[i]}={scores[i]:.3f}' for i in range(5)]}  ->  {vocala[best_idx]}")

if len(sys.argv) > 1:
    for fisier in sys.argv[1:]:
        clasifica(fisier)
else:
    # ruleaza pe toate fisierele din recordings/
    for fisier in sorted(Path("recordings").glob("*.txt")):
        clasifica(str(fisier))
