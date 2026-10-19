import pandas
import numpy as np
import torch
import torch.nn as nn
import random

print("PyTorch versione:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

SIM_DT = 0.001
DURATION_PER_STEP = 0.4
INPUT_A_CSV = "../Datasets/trainingA.csv"
INPUT_B_CSV = "../Datasets/trainingB.csv"
NUM_VELOCITIES = 10

dataA = pandas.read_csv(INPUT_A_CSV, sep=',')
dataB = pandas.read_csv(INPUT_B_CSV, sep=',')

# --- Costruiamo LISTE di traiettorie a lunghezza variabile (niente np.array rettangolare) ---
x_train_data_A = []
y_train_data_A = []
for traj_id, traj_df in dataA.groupby("traj_id", sort=True):
    x_train_data_A.append(traj_df["velocity_rad_s"].to_numpy(dtype=np.float32))
    y_train_data_A.append(traj_df["torque_friction_estimated_Nm"].to_numpy(dtype=np.float32))

x_train_data_B = []
y_train_data_B = []
for traj_id, traj_df in dataB.groupby("traj_id", sort=True):
    x_train_data_B.append(traj_df["velocity_rad_s"].to_numpy(dtype=np.float32))
    y_train_data_B.append(traj_df["torque_friction_estimated_Nm"].to_numpy(dtype=np.float32))

num_traj_A = len(y_train_data_A)
num_traj_B = len(y_train_data_B)
num_traj = num_traj_A + num_traj_B

all_u = np.concatenate(x_train_data_A + x_train_data_B)
all_y = np.concatenate(y_train_data_A + y_train_data_B)

u_mean, u_std = all_u.mean(), all_u.std()
y_mean, y_std = all_y.mean(), all_y.std()

# per evitare divisioni per 0
u_std = u_std if u_std > 1e-8 else 1.0
y_std = y_std if y_std > 1e-8 else 1.0

def normalize_u(arr):
    return (arr - u_mean) / u_std


def normalize_y(arr):
    return (arr - y_mean) / y_std

# normalizzazione per saturazione tanh
X_train_n_A = [normalize_u(x) for x in x_train_data_A]
y_train_n_A = [normalize_y(y) for y in y_train_data_A]

X_train_n_B = [normalize_u(x) for x in x_train_data_B]
y_train_n_B = [normalize_y(y) for y in y_train_data_B]


x_train_list = X_train_n_A + X_train_n_B
y_train_list = y_train_n_A + y_train_n_B

pairs = list(zip(x_train_list, y_train_list))
random.shuffle(pairs)
x_train_list, y_train_list = zip(*pairs)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class RNN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.rnn = nn.RNN(in_channels, hidden)
        self.fc = nn.Linear(hidden, out_channels)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out)
        return out


IN_CHANNELS = 1  
HIDDEN = 16
OUT_CHANNELS = 1

model = RNN(in_channels=IN_CHANNELS, hidden=HIDDEN, out_channels=OUT_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15
)
criterion = nn.MSELoss()
print(model)
EPOCHS = 250
MAX_GRAD_NORM = 5.0

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x,y in zip(x_train_list, y_train_list):
        optimizer.zero_grad()
        y_train = torch.from_numpy(y).to(device).view(1, -1, 1)
        x_batch = torch.from_numpy(x).to(device).view(1,-1,1)
       
        y_pred = model(x_batch)
      
        loss = criterion(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= num_traj
    scheduler.step(epoch_loss)  # una volta per epoca

    if (epoch + 1) % 50 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoca [{epoch+1}/{EPOCHS}] "
              f"Loss: {loss.item():.5f} | lr: {current_lr:.2e}")

torch.save({
    'model_state_dict': model.state_dict(),
    'in_channels': IN_CHANNELS,
    'hidden': HIDDEN,
    'out_channels': OUT_CHANNELS,
    'u_mean': u_mean, 'u_std': u_std,
    'y_mean': y_mean, 'y_std': y_std,
}, 'LinNet_checkpoint.pth')