import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"D:\scope_9.csv"
# ======================================================================
df = pd.read_csv(CSV_PATH, skiprows=1)
t, ch1, ch2 = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values
dt = np.median(np.diff(t))
n = len(t)

# ---- signal frequency ----
spectrum = np.abs(np.fft.rfft((ch1 - ch1.mean()) * np.hanning(n)))
freqs = np.fft.rfftfreq(n, dt)
spectrum[0] = 0  # ignore DC
f_guess = freqs[np.argmax(spectrum)]


def fit_sine(t, x, f):
    """Fit x = A*sin(2*pi*f*t + phase) + offset. Returns A, phase, offset, fitted curve, R^2."""
    w = 2 * np.pi * f
    M = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    a, b, offset = np.linalg.lstsq(M, x, rcond=None)[0]
    A = np.hypot(a, b)
    phase = np.arctan2(b, a)
    fitted = M @ [a, b, offset]
    r2 = 1 - np.sum((x - fitted) ** 2) / np.sum((x - x.mean()) ** 2)
    return A, phase, offset, fitted, r2


best_f, best_r2 = f_guess, -1
df_bin = freqs[1] - freqs[0]
span = max(df_bin * 2, f_guess * 0.1)
f_lo = max(f_guess - span, freqs[1])
f_hi = f_guess + span
for f in np.linspace(f_lo, f_hi, 3000):
    _, _, _, _, r2 = fit_sine(t, ch1, f)
    if r2 > best_r2:
        best_f, best_r2 = f, r2

# --- Fit both channels at the frequency ----
A1, phase1, off1, fit1, r2_1 = fit_sine(t, ch1, best_f)
A2, phase2, off2, fit2, r2_2 = fit_sine(t, ch2, best_f)

# --- phase difference, time delay, and gain ----
period = 1 / best_f
phase_deg = np.degrees(phase2 - phase1)
phase_deg = (phase_deg + 180) % 360 - 180               
delay_s = (phase2 - phase1) / (2 * np.pi * best_f)
delay_s = (delay_s + period / 2) % period - period / 2    
gain = A2 / A1


print(f"File          : {CSV_PATH}")
print(f"Frequency     : {best_f:.2f} Hz  (period {period*1e3:.2f} ms)")
print(f"CH1 amplitude : {A1:.4f} V   (fit R^2 = {r2_1:.2f})")
print(f"CH2 amplitude : {A2:.4f} V   (fit R^2 = {r2_2:.2f})")
print(f"Gain (CH2/CH1): {gain:.4f}")
print(f"Phase(1->2)   : {phase_deg:.3f} deg")
print(f"Delay(1->2)   : {delay_s*1e3:.3f} ms")


time = t - t[0]
plt.figure(figsize=(9, 4))
plt.plot(time, fit1, color="darkgoldenrod", label="Input Signal(1)")
plt.plot(time, fit2, color="darkgreen", label="Filter Output(2)")

stats_text = (
    f"• Frequency = {best_f:.2f} Hz\n"
    f"• Phase(1→2) = {phase_deg:.3f}°\n"
    f"• Delay(1→2) = {delay_s*1e3:.3f} ms\n"
    f"• Gain = {gain:.3f}"
)
plt.gca().text(
    0.75, 0.50, stats_text, transform=plt.gca().transAxes,
    fontsize=9, va="top", ha="left"
)
plt.xlabel("time (s)")
plt.ylabel("Voltage (V)")
plt.title("INPUT VS FILTER OUTPUT")
plt.legend(fontsize=10, loc="best")
plt.grid(axis='y', linestyle='solid', linewidth=1, alpha=0.6)
plt.margins(x=0)
plt.tight_layout()
plt.show()                             