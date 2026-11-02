import pandas
import matplotlib.pyplot as plt
import numpy as np

SIM_DT = 0.001
RAMP_DURATION = 0.3
INPUT_CSV = "../Datasets/test_True.csv"
INPUT_CSV_NO_CONTROL = "../Datasets/test_False.csv"

data = pandas.read_csv(INPUT_CSV_NO_CONTROL, sep = ',')
data_control = pandas.read_csv(INPUT_CSV, sep = ',')


v_measured_control = np.array([data_control[f"velocity_rad_{j}_s"] for j in range(7)])
v_desired_control = np.array([data_control[f"velocity_des_{j}_s"] for j in range(7)])
errore_control = np.array([data_control[f"errore_{j}"] for j in range(7)])

v_measured = np.array([data[f"velocity_rad_{j}_s"] for j in range(7)])
v_desired = np.array([data[f"velocity_des_{j}_s"] for j in range(7)])
errore = np.array([data[f"errore_{j}"] for j in range(7)])
time = data["t"]


fig, axes = plt.subplots(7, 2, figsize=(15, 12),
                         sharex=True, sharey=True, squeeze=False)
axes = axes.flatten()
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
i = 0
for j in range(7):
    
    ax = axes[i]
    ax.plot(time, v_measured[j], label="Velocità misurata", color="#EC0303",  linestyle="--", linewidth=1.5)
    ax.plot(time, v_desired[j], label="Veloctià desiderata", color="#090001", linewidth=1.5,alpha=0.7 )
    ax.set_title(f"Giunto {j}. NRMSE:{rmse(v_measured[j], v_desired[j])/v_desired.std():.4f}", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")

    i+=1
    ax = axes[i]
    ax.plot(time, v_measured_control[j], label="Velocità misurata con retroazione", color="#0F03EC",  linestyle="--", linewidth=1.5)
    ax.plot(time, v_desired_control[j], label="Veloctià desiderata", color="#090001", linewidth=1.5,alpha=0.7 )
    ax.set_title(f"Giunto {j}. NRMSE:{rmse(v_measured_control[j], v_desired_control[j])/v_desired_control.std():.4f}", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")
    i+=1

fig.supxlabel("Tempo [s]", fontsize=12, fontweight="bold")
fig.supylabel("Velocità [Rad/s]", fontsize=12, fontweight="bold")
fig.suptitle("Confronto tra efficacia inseguitore di velocità con e senza retroazione", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()
fig_error, axes_error =  plt.subplots(7, 2, figsize=(15, 12),
                         sharex=True, sharey=True, squeeze=False)

axes_error = axes_error.flatten()

i = 0
for j in range(7):
    ax = axes_error[i]
    ax.plot(time, errore[j], label="Errore di inseguimento", color="#EC0303", linewidth=1.5)
    ax.set_title(f"Errore di inseguimento per il giunto {i+1} senza retroazione dell'attrito", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    i+=1
    ax = axes_error[i]
    ax.plot(time, errore_control[j], label="Errore di inseguimento", color="#EC0303", linewidth=1.5)
    ax.set_title(f"Errore di inseguimento per il giunto {i+1} con retroazione dell'attrito", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    i+=1



fig_error.supxlabel("Tempo [s]", fontsize=12, fontweight="bold")
fig_error.supylabel("Errore di inseguimento [Rad/s]", fontsize=12, fontweight="bold")
fig_error.suptitle("Errore di inseguimento per ogni giunto", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()

