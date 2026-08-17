import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io.wavfile import write

filename = "data/record100/100"
record = wfdb.rdsamp(filename)
fs = record[1]["fs"]
record_len_in_samples = 4096
unfiltered_record = record[0][:record_len_in_samples, 0]
t = np.arange(0, len(unfiltered_record) / fs, 1 / fs)
plt.plot(t, unfiltered_record)
plt.title("Raw ECG (For data mitb/100)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()


# ---------- WAV export ----------

ecg_dc = unfiltered_record - np.mean(unfiltered_record)
ecg_norm = ecg_dc / np.max(np.abs(ecg_dc))
ecg_intput = (np.clip(ecg_norm, -1.0, 1.0) * 32767).astype(np.int16)


write("ecg_input_data_100.wav", int(fs), ecg_intput)
print("ecg_input_data_100.wav")