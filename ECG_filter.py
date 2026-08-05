import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

CSV_PATH = r"D:\scope_68.csv"

# ---- 1) Load the data (time, CH1, CH2, CH3) ----
df = pd.read_csv(CSV_PATH, skiprows=1)
t = df.iloc[:, 0].values
ch1 = df.iloc[:, 1].values
ch2 = df.iloc[:, 2].values
ch3 = df.iloc[:, 3].values
dt = np.median(np.diff(t))

channels = {"Input ECG Signal": ch1, "Filter Output": ch2, "Pre-Amplifier Output": ch3}
colors = {"Input ECG Signal": "darkgoldenrod", "Filter Output": "green", "Pre-Amplifier Output": "blue"}

# ----  Peak-to-peak amplitude ----
vpp = {name: x.max() - x.min() for name, x in channels.items()}

# ----  Amplifier gain ----
gain = vpp["Pre-Amplifier Output"] / vpp["Input ECG Signal"]

# ----  Delay  ----
min_dist = int(0.003 / dt)  # beats at least 3 ms apart
base1, base2 = np.median(ch1), np.median(ch2)
thresh1 = base1 + 0.5 * (ch1.max() - base1)
thresh2 = base2 + 0.5 * (ch2.max() - base2)
peaks1, _ = find_peaks(ch1, height=thresh1, distance=min_dist)
peaks2, _ = find_peaks(ch2, height=thresh2, distance=min_dist)

delay_list = []
for i1 in peaks1:
    i2 = peaks2[np.argmin(np.abs(t[peaks2] - t[i1]))]
    delay_list.append(t[i2] - t[i1])
delay_s = np.mean(delay_list)

print(f"Input Signal  : {vpp['Input ECG Signal']:.4f} Vp-p")
print(f"Filter Output : {vpp['Filter Output']:.4f} Vp-p")
print(f"PreAmplifier Output : {vpp['Pre-Amplifier Output']:.4f} Vp-p")
print(f"Amplifier gain): {gain:.4f}")
print(f"Delay 1-> 2   : {delay_s*1e3:.4f} ms")


time = t - t[0]
plt.figure(figsize=(9, 5))
for name, x in channels.items():
    plt.plot(time, x, color=colors[name], label=name)

stats_text = (
    f"\u2022 Input Signal  = {vpp['Input ECG Signal']:.3f} Vp-p\n"
    f"\u2022 Filter Output : {vpp['Filter Output']:.3f} Vp-p\n"
    f"\u2022 PreAmplifier Output : {vpp['Pre-Amplifier Output']:.3f} Vp-p\n"
    f"\u2022 Gain = {gain:.4f}\n"
    f"\u2022 Delay (1\u21922) = {delay_s*1e3:.3f} ms"
)
plt.gca().text(0.01, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=9, va="top", ha="left")

plt.xlabel("time (s)")
plt.ylabel("Voltage (V)")
plt.title("ECG Signals (MIT-BIH-124): INPUT vs FILTER OUTPUT vs PRE-AMPLIFIER OUTPUT")
plt.legend(fontsize=10, loc="upper right")
plt.grid(axis='y', alpha=0.4)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ecg_gain_vpp.png", dpi=140)
plt.show()