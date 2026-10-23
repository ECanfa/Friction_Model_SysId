import pandas
import matplotlib.pyplot as plt
import numpy as np

SIM_DT = 0.001
RAMP_DURATION = 0.3
INPUT_CSV = "../Datasets/testC.csv"
NUM_VELOCITIES = 18
num_traj = 40


data = pandas.read_csv(INPUT_CSV, sep = ',')


frictions = np.array([data[f"tau_friction_estimated_{i}"] for i in range(7)])


velocities = np.array([data[f"velocity_rad_{i}_s"] for i in range(7)])
time = data["t"]


n_rows = 20
fig, axes = plt.subplots(4, 2, figsize=(15, 2.2 * n_rows + 1),
                         sharex=True, sharey=True, squeeze=False)
axes = axes.flatten()
points_per_traj = int(len(frictions[0])/18)
for i in range(7):
    ax = axes[i]
    t = time[points_per_traj:2*points_per_traj]
    ax.plot(t, frictions[i,points_per_traj:2*points_per_traj], label="Attrito", color="#1a1a1a", linewidth=1.5, alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.5)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")

for ax in axes[num_traj:]:          # nasconde i riquadri inutilizzati
    ax.set_visible(False)

fig.supxlabel("Velocità [Rad/s]", fontsize=12, fontweight="bold")
fig.supylabel("Coppia di attrito [Nm]", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()