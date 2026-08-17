import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV file paths
file_l1_l8  = r"E:\LCS_Project_File\PCB_LCS_Output\LCS_Out_L1_L8.csv"
file_l9_l16 = r"E:\LCS_Project_File\PCB_LCS_Output\LCS_Out_L9_L16.csv"
SMOOTH = 11

# Read oscilloscope CSV
def read_scope_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[1:].copy()
    df = df.iloc[:, :3]
    df.columns = ["Time", "Voltage", "Digital"]
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
    df["Digital"] = pd.to_numeric(df["Digital"], errors="coerce")
    df = df.dropna(subset=["Time", "Voltage", "Digital"])
    df["Digital"] = df["Digital"].astype(int)

    return df

df1 = read_scope_csv(file_l1_l8)
df2 = read_scope_csv(file_l9_l16)
n = min(len(df1), len(df2))
df1 = df1.iloc[:n].reset_index(drop=True)
df2 = df2.iloc[:n].reset_index(drop=True)
time = df1["Time"].to_numpy()
voltage_raw = df1["Voltage"].to_numpy()

# Estimate sampling time
dt = np.mean(np.diff(time))

# Remove DC
v_centered = voltage_raw - np.mean(voltage_raw)

# FFT frequency estimation
fft_values = np.fft.rfft(v_centered)
fft_freqs = np.fft.rfftfreq(len(v_centered), dt)

# Remove DC component
dominant_index = np.argmax(np.abs(fft_values[1:])) + 1
freq = fft_freqs[dominant_index]

# sine wave with phase:
w = 2 * np.pi * freq
X = np.column_stack([
    np.sin(w * time),
    np.cos(w * time),
    np.ones_like(time)
])
A, B, C = np.linalg.lstsq(X, voltage_raw, rcond=None)[0]
Output = A * np.sin(w * time) + B * np.cos(w * time) + C
d1 = df1["Digital"].to_numpy(np.int64)
d2 = df2["Digital"].to_numpy(np.int64)

for bit in range(8):
    df1[f"L{bit + 1}"] = (d1 // (2 ** bit)) % 2
    df2[f"L{bit + 9}"] = (d2 // (2 ** bit)) % 2

# Reconstruct output
level = np.zeros(n)
for k in range(1, 9):
    level += df1[f"L{k}"].to_numpy()

for k in range(9, 17):
    level += df2[f"L{k}"].to_numpy()
level = (
    pd.Series(level)
    .rolling(SMOOTH, center=True, min_periods=1)
    .median()
    .round()
    .astype(int)
    .to_numpy()
)
level = np.clip(level, 0, 16)

#L1-L16 Output
bits = {}
for k in range(1, 17):
    bits[k] = (level >= k).astype(int)
fig, ax = plt.subplots(figsize=(15, 9))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
sine_base = 11.4
sine_height = 2.0

# Normalize sine
sine_norm = (Output - np.min(Output)) / (np.max(Output) - np.min(Output))
sine_plot = sine_base + sine_norm * sine_height
ax.plot(
    time,
    sine_plot,
    color="yellow",
    linewidth=2.4
)
ax.text(
    time.min(),
    sine_base + sine_height / 2,
    "Amp_Out",
    color="yellow",
    fontsize=11,
    ha="right",
    va="center"
)

# Plot L1-L16 digital outputs
spacing = 0.55
amplitude = 0.35
for index, level_number in enumerate(range(16, 0, -1)):
    y_base = (16 - index) * spacing
    signal = bits[level_number]
    if level_number >= 13:
        color = "red"
    elif level_number >= 9:
        color = "cyan"
    else:
        color = "lime"
    ax.step(
        time,
        y_base + signal * amplitude,
        where="post",
        color=color,
        linewidth=1.35
    )
    ax.text(
        time.min(),
        y_base + amplitude / 2,
        f"L{level_number}",
        color=color,
        fontsize=9,
        ha="right",
        va="center"
    )
ax.set_title(
    "PreAmplifer Output with L1-L16 Digital Outputs",
    color="white",
    fontsize=16,
    pad=15
)
ax.set_xlabel("Time (s)", color="white", fontsize=12)
ax.set_xlim(time.min(), time.max())
ax.set_ylim(0, 14.0)
ax.set_yticks([])
ax.grid(True, which="major", color="gray", alpha=0.45, linewidth=0.7)
ax.grid(True, which="minor", color="gray", alpha=0.25, linewidth=0.4)
ax.minorticks_on()
ax.tick_params(axis="x", colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
ax.text(
    0.01,
    1.02,
    "L1-L16 Digital Outputs",
    transform=ax.transAxes,
    color="lime",
    fontsize=11,
    ha="left"
)
plt.tight_layout()
plt.show()