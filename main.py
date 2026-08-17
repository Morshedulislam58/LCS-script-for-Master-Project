import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io.wavfile import write

# ✅ Correct path for your project structure
filename = "data/record100/100"

signals, fields = wfdb.rdsamp(filename)
fs = int(fields["fs"])

record_len_in_samples = 4096
ecg = signals[:record_len_in_samples, 0]

print("Sampling rate (Hz):", fs)
print("Loaded samples:", len(ecg))
print("Duration (s):", len(ecg) / fs)

t = np.arange(len(ecg)) / fs
plt.figure()
plt.plot(t, ecg)
plt.title("Raw ECG (first 4096 samples)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# ----- Make LTspice-friendly WAV (16-bit PCM) -----
ecg_dc = ecg - np.mean(ecg)
peak = np.max(np.abs(ecg_dc))
ecg_norm = ecg_dc / peak

ecg_int16 = (np.clip(ecg_norm, -1.0, 1.0) * 32767).astype(np.int16)
write("ECG_input_data_100.wav", fs, ecg_int16)
print("ECG_input_data_100.wav")
