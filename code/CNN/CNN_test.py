import numpy as np
import pandas
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------
giunto = 6
CHECKPOINT = f"tcn_checkpoint_{giunto}.pth"
INPUT_CSV = "../Datasets/testB.csv"
DATASET_NAME = "Test C"
LOW_VEL_THRESHOLD = 0.1     
MAX_PLOTS = 15             
PLOT_COLS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# MODELLO 
# ----------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, dilations):
        super().__init__()
        layers, c = [], in_channels
        for d in dilations:
            layers += [nn.Conv1d(c, channels, kernel_size, dilation=d), nn.ReLU()]
            c = channels
        self.body = nn.Sequential(*layers)
        self.final = nn.Conv1d(channels, 1, kernel_size=1)
        self.receptive_field = 1 + sum(d * (kernel_size - 1) for d in dilations)

    def forward(self, x):                 # x: [B, C, L + RF - 1] -> [B, 1, L]
        return self.final(self.body(x))


ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model = TCN(ckpt["in_channels"], ckpt["channels"], ckpt["kernel_size"],
            tuple(ckpt["dilations"])).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

RF = model.receptive_field
PAD = RF - 1
input_cols = ckpt["input_cols"]                                       # ordine dei canali del training
x_mean = np.asarray(ckpt["x_mean"], np.float32).reshape(-1, 1)        # [C, 1]
x_std = np.asarray(ckpt["x_std"], np.float32).reshape(-1, 1)          # [C, 1]
y_scale = ckpt["y_scale"]

VEL_IDX = input_cols.index(f"velocity_rad_{giunto}_s")                          # canale della velocita'

print(f"Checkpoint: {CHECKPOINT} | campo recettivo: {RF} campioni | "
      f"val RMSE (training): {ckpt['best_val_rmse_Nm']:.3f} Nm")
print(f"Canali di ingresso: {input_cols}")


@torch.no_grad()
def predict(x):
    """Stessa procedura del training: normalizzazione per canale, PAD campioni in testa
    ottenuti replicando il primo campione normalizzato, uscita riportata in Nm.
    x: [C, T] grezzo -> [T]"""
    xn = ((x - x_mean) / x_std).astype(np.float32)
    pad = np.repeat(xn[:, :1], PAD, axis=1)
    xp = np.concatenate([pad, xn], axis=1)                            # [C, PAD + T]
    xt = torch.from_numpy(xp).unsqueeze(0).to(device)                 # [1, C, PAD + T]
    return model(xt).view(-1).cpu().numpy() * y_scale


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------------------------------------------------
# DATI DI TEST 
# ----------------------------------------------------------------------
data = pandas.read_csv(INPUT_CSV)

t_list, x_list, y_list = [], [], []
for traj_id, g in data.groupby("traj_id", sort=True):
    t_list.append(g["t"].to_numpy())
    x_list.append(g[input_cols].to_numpy(np.float32).T)               # [C, T]
    y_list.append(g[f"tau_friction_estimated_{giunto}"].to_numpy(np.float32))

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

    mask = np.abs(x[VEL_IDX]) < LOW_VEL_THRESHOLD  
    rmse_low.append(rmse(y_hat[mask], y[mask]) if mask.any() else np.nan)

rmse_all = np.array(rmse_all)
rmse_low = np.array(rmse_low)

y_all = np.concatenate(y_list)
p_all = np.concatenate(preds)
rmse_global = rmse(p_all, y_all)
baseline = float(y_all.std())                                  # media per confronto con errore della rete
noise_floor = 0.02 * float(np.sqrt(np.mean(y_all ** 2)))       # stima del rumore non imparabile dalla rete

# ----------------------------------------------------------------------
# GRAFICI
# ----------------------------------------------------------------------
n_plot = min(num_traj, MAX_PLOTS)
rows = int(np.ceil(n_plot / PLOT_COLS))
fig, axes = plt.subplots(rows, PLOT_COLS, figsize=(15, 2.4 * rows + 1),
                         sharex=True, sharey=True, squeeze=False)
axes = axes.flatten()

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
print(f"RMSE medio: {rmse_all.mean():.3f} Nm ± {rmse_all.std():.3f} Nm")
print(f"RMSE min: {rmse_all.min():.3f} Nm  |  RMSE max: {rmse_all.max():.3f} Nm")
print(f"RMSE globale: {rmse_global:.3f} Nm")
if not np.all(np.isnan(rmse_low)):
    print(f"RMSE a bassa velocita' (|v| < {LOW_VEL_THRESHOLD} rad/s): "
          f"{np.nanmean(rmse_low):.3f} Nm ± {np.nanstd(rmse_low):.3f} Nm")
print(f"Baseline (std del target): {baseline:.3f} Nm  |  soglia di rumore ~{noise_floor:.3f} Nm")