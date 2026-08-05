import csv
import matplotlib.pyplot as plt

times, volts, digital = [], [], []
with open( r"F:\All analysis\scope_32.csv") as f:
    reader = csv.reader(f)
    next(reader)  # header row
    next(reader)  # units row
    for row in reader:
        times.append(float(row[0]) * 1000)  # seconds -> ms
        volts.append(float(row[1]))          # Channel 1 voltage
        digital.append(int(row[2]))          # D0-D7 packed as one decimal number

# --- Decode the digital bits ---

active_bits = [b for b in range(8) if len({(d >> b) & 1 for d in digital}) > 1]
colors = ["magenta", "cyan", "green", "white", "red", "orange", "blue", "purple"]


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 6), sharex=True,
    gridspec_kw={"height_ratios": [3, 2]}
)
fig.patch.set_facecolor("black")

# Analog channel
ax1.set_facecolor("black")
ax1.plot(times, volts, color="yellow", linewidth=1)
ax1.set_xticks(range(-50, 51, 10))
ax1.set_yticks(range(0, 6, 1))
ax1.grid(True, color="gray", alpha=0.4)
ax1.tick_params(colors="white")
ax1.set_ylabel("Amplitude (V)", color="white")
ax1.set_title("EKG_N_100_2225_4744 \n Decode state and corresponding sample events", color="yellow")

# Digital channels 
ax2.set_facecolor("black")
for i, bit in enumerate(active_bits):
    bits = [((d >> bit) & 1) * 0.8 + i for d in digital]  # squash + offset per row
    ax2.step(times, bits, where="post", color=colors[bit], linewidth=1)
    ax2.text(times[0] - 2, i + 0.3, f"D{bit}", color=colors[bit],
              ha="right", va="center", fontsize=9)

ax2.set_xticks(range(0, 5, 10))
ax2.set_yticks([])
ax2.grid(True, axis="x", color="gray", alpha=0.4)
ax2.tick_params(colors="white")
ax2.set_xlabel("Time (ms)", color="white")

plt.tight_layout()
plt.savefig("scope_plot.png", dpi=150, facecolor="black")
plt.show()