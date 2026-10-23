import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------
SIM_DT = 0.001
INPUT_CSV = "../Datasets/testC.csv"
DATASET_NAME = "Test B"          # coerente con INPUT_CSV
CHECKPOINT = "lstm_checkpoint.pth"
N_COLS = 3

# ----------------------------------------------------------------------
# MODELLO (stessa architettura del training)
# ----------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):      # x: [B, T, C] -> [B, T, 1]
        lstm_out, hidden = self.lstm(x, hidden)
        return self.linear(lstm_out), hidden


checkpoint = torch.load(CHECKPOINT, weights_only=True)
model = LSTM(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["output_dim"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

input_cols = checkpoint["input_cols"]                                   # ordine dei canali del training
u_mean = np.asarray(checkpoint["u_mean"], np.float32).reshape(-1, 1)    # [C, 1]
u_std = np.asarray(checkpoint["u_std"], np.float32).reshape(-1, 1)      # [C, 1]
y_std = checkpoint["y_std"]
print(f"Canali di ingresso: {input_cols}")

# ----------------------------------------------------------------------
# DATI
# ----------------------------------------------------------------------
data = pandas.read_csv(INPUT_CSV)

x_test, y_test = [], []
for _, g in data.groupby("traj_id", sort=True):
    x_test.append(g[input_cols].to_numpy(np.float32).T)                 # [C, T]
    y_test.append(g["torque_friction_estimated_Nm"].to_numpy(np.float32))

num_traj = len(y_test)
print(f"Traiettorie di test: {num_traj}")

# ----------------------------------------------------------------------
# PREDIZIONE (una traiettoria alla volta: nessun vincolo sulla lunghezza)
# ----------------------------------------------------------------------
y_pred = []
with torch.no_grad():
    for x in x_test:
        xn = ((x) / u_std).astype(np.float32)                  # normalizzazione come nel training
        xt = torch.from_numpy(np.ascontiguousarray(xn.T)).unsqueeze(0)  # [1, T, C]
        out, _ = model(xt)
        y_pred.append(out.view(-1).numpy() * y_std)                     # denormalizzazione

# ----------------------------------------------------------------------
# METRICHE
# ----------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

rmse_scores = [rmse(y, p) for y, p in zip(y_test, y_pred)]

y_all = np.concatenate(y_test)
p_all = np.concatenate(y_pred)
rmse_global = rmse(y_all, p_all)
baseline = float(y_all.std())                       # "predico la media"
noise_floor = 0.02 * float(np.sqrt(np.mean(y_all ** 2)))   # stima del rumore irriducibile (2%)

print(f"RMSE medio per traiettoria: {np.mean(rmse_scores):.3f} Nm ± {np.std(rmse_scores):.3f} Nm")
print(f"RMSE min: {np.min(rmse_scores):.3f} Nm  |  RMSE max: {np.max(rmse_scores):.3f} Nm")
print(f"RMSE globale: {rmse_global:.3f} Nm")
print(f"Baseline (std del target): {baseline:.3f} Nm  |  soglia di rumore ~{noise_floor:.3f} Nm")

# ----------------------------------------------------------------------
# GRAFICI
# ----------------------------------------------------------------------
n_rows = int(np.ceil(num_traj / N_COLS))
fig, axes = plt.subplots(n_rows, N_COLS, figsize=(15, 2.2 * n_rows + 1),
                         sharex=True, sharey=True, squeeze=False)
axes = axes.flatten()

for i in range(num_traj):
    ax = axes[i]
    t = np.arange(len(y_test[i])) * SIM_DT
    ax.plot(t, y_test[i], label="Reale", color="#1a1a1a", linewidth=1.5, alpha=0.7)
    ax.plot(t, y_pred[i], label="LSTM", color="#e63946", linestyle="--", linewidth=1.5)
    ax.set_title(f"Traiettoria {i+1} (RMSE: {rmse_scores[i]:.2f} Nm)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")

for ax in axes[num_traj:]:          # nasconde i riquadri inutilizzati
    ax.set_visible(False)

fig.supxlabel("Tempo [s]", fontsize=12, fontweight="bold")
fig.supylabel("Coppia di attrito [Nm]", fontsize=12, fontweight="bold")
fig.suptitle(f"{DATASET_NAME}: coppia di attrito stimata vs predetta", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()