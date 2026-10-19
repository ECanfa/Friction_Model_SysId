import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import root_mean_squared_error

SIM_DT = 0.001
DURATION_PER_STEP = 0.7
DATASET_NAME = "Test C"
INPUT_CSV = "../Datasets/testC.csv"
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

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.linear(lstm_out)
        return predictions, hidden
    
checkpoint = torch.load('lstm_checkpoint.pth', weights_only=False)
model = LSTM(checkpoint['input_dim'], checkpoint['hidden_dim'], checkpoint['output_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


u_std = checkpoint['u_std']
y_std = checkpoint['y_std']

with torch.no_grad():
        
    X_test_tensor = torch.tensor(x_test_data, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32).unsqueeze(-1)
    X_test_n = (X_test_tensor ) / u_std          # normalizzazione input come per training


    y_test_pred_n, _ = model(X_test_n)
    y_test_pred = y_test_pred_n * y_std           # denormalizzazione output per fit
    y_test_pred = y_test_pred.squeeze(-1).numpy()



    fig, axes = plt.subplots(6, 3, figsize=(15, 12), sharex=True, sharey=True)

    axes = axes.flatten()

    RMSE_scores = []
    for i in range(num_traj):
        y_target = y_test_data[i, :]
        y_estimated = y_test_pred[i, :]
        rme = root_mean_squared_error(y_target, y_estimated)
        RMSE_scores.append(rme)
        
        ax = axes[i]
        
        ax.plot(y_target, label="Reale", color="#1a1a1a", linewidth=1.5, alpha=0.7)
        ax.plot(y_estimated, label="ResNet", color="#e63946", linestyle="--", linewidth=1.5)
        
        ax.set_title(f"Traiettoria {i+1} (RMSE: {rme:.1f}Nm)", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.5)
        
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")


    fig.supxlabel("Campioni Temporali ($k$)", fontsize=12, fontweight='bold')
    fig.supylabel("Uscita del Sistema", fontsize=12, fontweight='bold')
    fig.suptitle(f"{DATASET_NAME}: coppia di attrito stimata vs predetta", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()

print(f"RMSE medio: {np.mean(RMSE_scores):.2f}Nm ± {np.std(RMSE_scores):.2f}Nm")
print(f"RMSE min: {np.min(RMSE_scores):.2f}Nm  |  RMSE max: {np.max(RMSE_scores):.2f}Nm")