import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

raw = pd.read_csv(r"D:\scope_68.csv")
data = raw.iloc[1:].apply(pd.to_numeric, errors="coerce").dropna()
data.columns = ["time", "input", "filter", "preamp"]

time = data["time"].to_numpy()
input_ecg = data["input"].to_numpy()
preamp = data["preamp"].to_numpy()
time_ms = (time - time[0]) * 1000

# delay
corr = correlate(preamp - np.mean(preamp), input_ecg - np.mean(input_ecg), mode="full")
lags = correlation_lags(len(preamp), len(input_ecg), mode="full")
delay_samples = int(lags[np.argmax(corr)])

sample_time = np.median(np.diff(time))

if delay_samples > 0:
    input_aligned = input_ecg[:-delay_samples]
    preamp_aligned = preamp[delay_samples:]
    time_aligned = time[:-delay_samples]
elif delay_samples < 0:
    input_aligned = input_ecg[-delay_samples:]
    preamp_aligned = preamp[:delay_samples]
    time_aligned = time[-delay_samples:]
else:
    input_aligned = input_ecg
    preamp_aligned = preamp
    time_aligned = time

time_ms_aligned = (time_aligned - time_aligned[0]) * 1000

# gain 
input_pp = np.ptp(input_ecg)
preamp_pp = np.ptp(preamp)
gain_pp = preamp_pp / input_pp

# preamp for MSE calculation
A = np.column_stack((input_aligned, np.ones_like(input_aligned)))
gain_fit, offset = np.linalg.lstsq(A, preamp_aligned, rcond=None)[0]
corrected_preamp = (preamp_aligned - offset) / gain_fit

# Squared error
error = input_aligned - corrected_preamp
squared_error = error**2
mse = np.mean(squared_error)
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Input ECG
axes[0].plot(time_ms, input_ecg, color="blue", label="Input ECG")
axes[0].set_title("Input ECG Signal (MIT-BIH 124)")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude (V)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Pre-amplifier output
axes[1].plot(time_ms, preamp, color="red", label="Pre-amplifier output")
axes[1].set_title("Pre-Amplifier ECG Output")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Amplitude (V)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[1].text(
    0.98, 0.95,
    f"Gain = {gain_pp:.2f}",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

# Squared error
axes[2].semilogy(
    time_ms_aligned,
    np.maximum(squared_error, np.finfo(float).tiny),
    color="green",
    label="Squared error"
)
axes[2].axhline(mse, color="black", linestyle="--", label=f"MSE = {mse:.3e} V²")
axes[2].set_title("Squared Error")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Squared error (V²)")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig("ecg_input_preamp_error_3plot.png", dpi=300, bbox_inches="tight")
plt.show()