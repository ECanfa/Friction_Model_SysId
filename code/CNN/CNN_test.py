import numpy as np
import pandas
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------
CHECKPOINT = "tcn_checkpoint.pth"
INPUT_CSV = "../Datasets/testC.csv"
DATASET_NAME = "Test C"
LOW_VEL_THRESHOLD = 0.1     # rad/s, zona di Stribeck: RMSE riportata anche separatamente
MAX_PLOTS = 15              # numero massimo di traiettorie mostrate nel grafico
PLOT_COLS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# MODELLO (identico a quello dello script di training)
# ----------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        layers, c = [], 1
        for d in dilations:
            layers += [nn.Conv1d(c, channels, kernel_size, dilation=d), nn.ReLU()]
            c = channels
        self.body = nn.Sequential(*layers)
        self.final = nn.Conv1d(channels, 1, kernel_size=1)
        self.receptive_field = 1 + sum(d * (kernel_size - 1) for d in dilations)

    def forward(self, x):                 # x: [B, 1, L + RF - 1] -> [B, 1, L]
        return self.final(self.body(x))


ckpt = torch.load(CHECKPOINT, map_location=device)
model = TCN(ckpt["channels"], ckpt["kernel_size"], tuple(ckpt["dilations"])).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

RF = model.receptive_field
PAD = RF - 1
x_scale = ckpt["x_scale"]      
y_scale = ckpt["y_scale"]
print(f"Checkpoint: {CHECKPOINT} | campo recettivo: {RF} campioni | "
      f"val RMSE (training): {ckpt['best_val_rmse_Nm']:.3f} Nm")


@torch.no_grad()
def predict(x):
    """Stessa procedura del training: RF-1 zeri in testa (la traiettoria parte da ferma),
    ingresso normalizzato, uscita riportata in Nm."""
    xp = np.concatenate([np.zeros(PAD, np.float32), x.astype(np.float32) / x_scale])
    xt = torch.from_numpy(xp).view(1, 1, -1).to(device)
    return model(xt).view(-1).cpu().numpy() * y_scale


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------------------------------------------------
# DATI DI TEST (liste: le traiettorie possono avere lunghezze diverse)
# ----------------------------------------------------------------------
data = pandas.read_csv(INPUT_CSV)

t_list, x_list, y_list = [], [], []
for traj_id, g in data.groupby("traj_id", sort=True):
    t_list.append(g["t"].to_numpy())
    x_list.append(g["velocity_rad_s"].to_numpy(np.float32))
    y_list.append(g["torque_friction_estimated_Nm"].to_numpy(np.float32))

num_traj = len(y_list)
print(f"{DATASET_NAME}: {num_traj} traiettorie")

# ----------------------------------------------------------------------
# VALUTAZIONE
# ----------------------------------------------------------------------
preds, rmse_all, rmse_low = [], [], []
for x, y in zip(x_list, y_list):
    y_hat = predict(x)
    preds.append(y_hat)
    rmse_all.append(rmse(y_hat, y))

    mask = np.abs(x) < LOW_VEL_THRESHOLD     # sulla velocita' misurata, non su quella di riferimento
    rmse_low.append(rmse(y_hat[mask], y[mask]) if mask.any() else np.nan)

rmse_all = np.array(rmse_all)
rmse_low = np.array(rmse_low)

# ----------------------------------------------------------------------
# GRAFICI
# ----------------------------------------------------------------------
n_plot = min(num_traj, MAX_PLOTS)
rows = int(np.ceil(n_plot / PLOT_COLS))
fig, axes = plt.subplots(5, 3, figsize=(15, 12), sharex=True, sharey=True)
axes = np.atleast_1d(axes).flatten()

for i in range(n_plot):
    ax = axes[i]
    ax.plot(t_list[i], y_list[i], label="Reale", color="#1a1a1a", linewidth=1.5, alpha=0.7)
    ax.plot(t_list[i], preds[i], label="TCN", color="#e63946", linestyle="--", linewidth=1.5)
    ax.set_title(f"Traiettoria {i + 1} (RMSE: {rmse_all[i]:.2f} Nm)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")
for ax in axes[n_plot:]:
    ax.axis("off")

fig.supxlabel("Tempo [s]", fontsize=12, fontweight="bold")
fig.supylabel("Coppia di attrito [Nm]", fontsize=12, fontweight="bold")
fig.suptitle(f"{DATASET_NAME}: coppia di attrito stimata vs predetta", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# RIEPILOGO
# ----------------------------------------------------------------------
print(f"RMSE medio: {rmse_all.mean():.2f}Nm ± {rmse_all.std():.2f}Nm")
print(f"RMSE min: {rmse_all.min():.2f}Nm  |  RMSE max: {rmse_all.max():.2f}Nm")
