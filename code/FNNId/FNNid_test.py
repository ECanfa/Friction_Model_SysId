import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas

SIM_DT = 0.001
DURATION_PER_STEP = 0.7
INPUT_CSV = "../Datasets/testB.csv"
NUM_VELOCITIES = 10

data = pandas.read_csv(INPUT_CSV, sep = ',')

friction = data["torque_friction_estimated_Nm"]
velocities = data["velocity_rad_s"]

x_test_data = []
y_test_data = []

for traj_id,traj_df in data.groupby("traj_id", sort=True):
    x_test_data.append(traj_df["velocity_rad_s"].to_numpy())
    y_test_data.append(traj_df["torque_friction_estimated_Nm"].to_numpy())

num_traj = len(y_test_data)
print(num_traj)

# in array numpy per efficienza 
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)

# --- ARCHITETTURA IDENTICA a quella usata in training ---
class RNN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.rnn = nn.RNN(in_channels, hidden)
        self.fc = nn.Linear(hidden, out_channels)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out)
        return out

checkpoint = torch.load('LinNet_checkpoint.pth', weights_only=False)
model = RNN(checkpoint['in_channels'], checkpoint['hidden'], checkpoint['out_channels'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

u_mean, u_std = checkpoint['u_mean'], checkpoint['u_std']
y_mean, y_std = checkpoint['y_mean'], checkpoint['y_std']

with torch.no_grad():
        
    X_test_tensor = torch.tensor(x_test_data, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32).unsqueeze(-1)
    X_test_n = (X_test_tensor - u_mean) / u_std          # normalizzazione input come per training


    y_test_pred_n= model(X_test_n)
    y_test_pred = y_test_pred_n * y_std + y_mean          # denormalizzazione output per fit
    y_test_pred = y_test_pred.squeeze(-1).numpy()



    fig, axes = plt.subplots(5, 3, figsize=(15, 12), sharex=True, sharey=True)

    axes = axes.flatten()

    fit_scores = []
    for i in range(num_traj):
        y_target = y_test_data[i, :]
        y_estimated = y_test_pred[i, :]
        fit_percentage = (1 - (np.linalg.norm(y_target - y_estimated) / np.linalg.norm(y_target - np.mean(y_target)))) * 100
        fit_scores.append(fit_percentage)
        
        
        ax = axes[i]
        
        ax.plot(y_target, label="Reale", color="#1a1a1a", linewidth=1.5, alpha=0.7)
        ax.plot(y_estimated, label="LSTM", color="#e63946", linestyle="--", linewidth=1.5)
        
        ax.set_title(f"Traiettoria {i+1} (FIT: {fit_percentage:.1f}%)", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.5)
        
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")


    fig.supxlabel("Campioni Temporali ($k$)", fontsize=12, fontweight='bold')
    fig.supylabel("Attrito [Nm]", fontsize=12, fontweight='bold')
    fig.suptitle("Validazione del modello", fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.show()

print(f"FIT medio: {np.mean(fit_scores):.2f}% ± {np.std(fit_scores):.2f}%")
print(f"FIT min: {np.min(fit_scores):.2f}%  |  FIT max: {np.max(fit_scores):.2f}%")