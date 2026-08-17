import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

in_wav  = "E:\LCS_Project_File\ECG_Filter_Analysis_Simulation\ecg_input_data_100.wav"
out_wav = "E:\LCS_Project_File\ECG_Filter_Analysis_Simulation\ecg_output_data_100.wav"
fs_in, x = read(in_wav)
fs_out, y = read(out_wav)

if x.ndim > 1: x = x[:, 0]
if y.ndim > 1: y = y[:, 0]

x = x.astype(np.float64)
y = y.astype(np.float64)

print("Input fs:", fs_in)
print("Output fs:", fs_out)

if fs_in != fs_out:
    raise ValueError("Sampling rates do not match!")

fs = int(fs_in)

# --- Match lengths ---
N = min(len(x), len(y))
x = x[:N]
y = y[:N]
print("samples:", N, "duration(s):", N/fs)

print("Input std:", np.std(x))
print("Output std:", np.std(y))

# --- Remove DC ---
x = x - np.mean(x)
y = y - np.mean(y)

# --- Use ONE normalization scale (preserve amplitude difference) ---
scale = np.max(np.abs(x)) + 1e-12
x = x / scale
y = y / scale

# --- Delay estimate ---
x0 = (x - np.mean(x)) / (np.std(x) + 1e-12)
y0 = (y - np.mean(y)) / (np.std(y) + 1e-12)

corr = np.correlate(y0, x0, mode="full")
shift = np.argmax(corr) - (N - 1)

max_shift = int(0.5 * fs)
shift = int(np.clip(shift, -max_shift, max_shift))

# --- Apply shift ---
if shift > 0:
    y_al = y[shift:]
    x_al = x[:len(y_al)]
elif shift < 0:
    x_al = x[-shift:]
    y_al = y[:len(x_al)]
else:
    x_al, y_al = x, y

if len(x_al) < 10:
    raise ValueError("Aligned signal too short. Check LTspice export time range.")

# ---distortion Metrics ---
err = x_al - y_al
MAE  = np.mean(np.abs(err))
MSE  = np.mean(err**2)

if np.std(x_al) < 1e-9 or np.std(y_al) < 1e-9:
    CORR = np.nan
else:
    CORR = np.corrcoef(x_al, y_al)[0, 1]

print("\n--- Distortion Metrics ---")
print("MAE :", MAE)
print("MSE :", MSE)
print("Corr:", CORR)

t = np.arange(len(x_al)) / fs
Tshow = min(6.0, t[-1])
mask = t <= Tshow

plt.figure()
plt.plot(t[mask], x_al[mask], label="Original signal")
plt.plot(t[mask], y_al[mask], label="Filtered Signal", alpha=0.8)
plt.title("Original Signal vs Filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(t[mask], err[mask])
plt.title("Error (Input - Output)")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.grid(True)
plt.show()
