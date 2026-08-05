import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = r"F:scope_32.csv"
# ---------------- Load CSV ----------------
times, volts, digital = [], [], []

with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)  # header row
    next(reader)  # units row

    for row in reader:
        times.append(float(row[0]) * 1000.0)  
        volts.append(float(row[1]))           
        digital.append(int(row[2]))           

times = np.array(times)
volts = np.array(volts)
digital = np.array(digital)
times = times - times[0]   

# ---------------- Normalize signal like ECG examples ----------------
x = volts - np.mean(volts)
x = x / (np.max(np.abs(x)) + 1e-12)

# ----------------  active digital bits ----------------
active_bits = [b for b in range(8) if len(np.unique((digital >> b) & 1)) > 1]

event_code = np.zeros_like(digital)

for k, bit in enumerate(active_bits):
    event_code += ((digital >> bit) & 1) << k

# ---------------- Event positions from digital changes ----------------
change_idx = np.where(np.diff(event_code) != 0)[0] + 1
event_idx = np.unique(np.r_[0, change_idx, len(times) - 1])

event_times = times[event_idx]
event_values = x[event_idx]

# ---------------- Linear interpolation reconstruction ----------------
x_interp = np.interp(times, event_times, event_values)

mse = np.mean((x - x_interp) ** 2)

# ---------------- Average sample rate  ----------------
avg_sample_rate = len(event_times) / ((times[-1] - times[0]) / 1000.0)  # events per second

# ---------------- Error metrics for the log-scale ----------------
error_curve = (x - x_interp) ** 2
error_floor = max(np.max(error_curve) * 1e-6, 1e-12)  
error_curve_plot = np.clip(error_curve, error_floor, None)

print("Active bits:", active_bits)
print("Number of events:", len(event_times))
print("Average sample rate: {:.1f} Hz".format(avg_sample_rate))
print("MSE:{:.3f}".format(mse))


# ---------------- Plot ----------------
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(9, 8), sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 0.6]}
)

signal_name = "EKG_N_100_2225_4744"

ax1.plot(times, x, label="original signal", linewidth=1.5)
dirac_base = np.min(x) - 0.45
ax1.hlines(dirac_base, times[0], times[-1], linewidth=1.0)
ax1.vlines(event_times, ymin=dirac_base, ymax=event_values, color="orange", linewidth=1.2, label="Level crossing samples")
ax1.set_title(f"{signal_name}\nEvent-based states and samples")
ax1.set_ylabel("amplitude")
ax1.legend(loc="lower left")
ax1.set_ylim(dirac_base - 0.05, np.max(x) + 0.15)

# ---  event count and sample rate ---
stats_text = f"Number of samples, N = {len(event_times)}"
ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes,
         fontsize=9, ha="right", va="top",
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

ax2.plot(times, x, label="original signal", linewidth=1.5)
ax2.plot(times, x_interp, label="linear interpolation", linewidth=1.5)
ax2.set_title("Reconstruction with linear interpolation")
ax2.set_ylabel("amplitude")
ax2.legend(loc="lower left")

ax3.semilogy(times, error_curve_plot, color="crimson", linewidth=1.0)
ax3.axhline(mse, color="black", linestyle=":", linewidth=1.2, label=f"MSE = {mse:.3f}")   
ax3.set_title("Sample-wise squared error (logarithmic scale)")
ax3.set_ylabel("squared error\n(log scale)")
ax3.set_xlabel("time (ms)")
ax3.legend(loc="upper right", fontsize=8)                                                  
ax3.set_xlim(times[0], times[-1])

plt.tight_layout()
plt.savefig("event_based_reconstruction.png", dpi=300)
plt.show()