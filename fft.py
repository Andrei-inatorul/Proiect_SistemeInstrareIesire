
import pickle
from pathlib import Path
import numpy as np
import scipy
from numpy.fft import fft, fftshift, fftfreq
from matplotlib import pyplot as plt

templates = []
filesDir = "./rec"
def normalize(data):
    data = np.array(data, dtype=np.float64)
    return (data - np.mean(data)) / np.std(data)


def alpha_filter(data, alpha=0.1):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]
    return filtered_data

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

with open('templates.pkl', 'rb') as file:
    templates = pickle.load(file)

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


plt.figure()
for i, y  in enumerate(signals):
    y = y[:44100]
    m = max(y)
    mi = min(y)
    n = np.array(y) - (mi + (m - mi) / 2)
    y=n
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / 44100.0
    x = np.linspace(0.0, N * T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)


    plt.title(f"{vocala[i]}")
    plt.subplot(2, 1, 1)
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.subplot(2, 1, 2)
    plt.plot(y, "-r")
    plt.show()