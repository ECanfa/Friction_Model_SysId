import pandas
import numpy as np
import torch
import torch.nn as nn
import random

SIM_DT = 0.001
DURATION_PER_STEP = 0.4
INPUT_A_CSV = "../Datasets/trainingA.csv"
INPUT_B_CSV = "../Datasets/trainingB.csv"
NUM_VELOCITIES = 10
PATIENCE = 40
VAL_FRACTION = 0.2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.linear(lstm_out)
        return predictions


INPUT_DIM = 1
HIDDEN_DIM = 18
OUTPUT_DIM = 1

def load(csv):
    df = pandas.read_csv(csv)
    xs, ys = [], []
    for _, g in df.groupby("traj_id", sort=True):
        xs.append(g["velocity_rad_s"].to_numpy(np.float32))
        ys.append(g["torque_friction_estimated_Nm"].to_numpy(np.float32))
    return xs, ys

#per splite delle traiettorie in training e validazione
def split(xs, ys):
    idx = list(range(len(xs)))
    random.shuffle(idx)
    n_val = max(1, int(round(VAL_FRACTION * len(xs))))
    val, tr = idx[:n_val], idx[n_val:]
    return ([xs[i] for i in tr], [ys[i] for i in tr],
            [xs[i] for i in val], [ys[i] for i in val])

xA, yA = load(INPUT_A_CSV)
xB, yB = load(INPUT_B_CSV)
xtrA, ytrA, xvaA, yvaA = split(xA, yA)
xtrB, ytrB, xvaB, yvaB = split(xB, yB)

x_tr, y_tr = xtrA + xtrB, ytrA + ytrB
x_va, y_va = xvaA + xvaB, yvaA + yvaB

u_std = float(np.concatenate(x_tr).std())
y_std = float(np.concatenate(y_tr).std())

# per evitare divisioni per 0
u_std = u_std if u_std > 1e-8 else 1.0
y_std = y_std if y_std > 1e-8 else 1.0


def normalize_u(arr):
    return (arr) / u_std


def normalize_y(arr):
    return (arr) / y_std

# normalizzazione per saturazione tanh
x_train_list = [normalize_u(x) for x in x_tr]
y_train_list = [normalize_y(y) for y in y_tr]


pairs = list(zip(x_train_list, y_train_list))
random.shuffle(pairs)
x_train_list, y_train_list = zip(*pairs)



model = LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15
)
criterion = nn.MSELoss()

MAX_GRAD_NORM = 5.0
EPOCHS = 300


@torch.no_grad()
def predict(x):

    model.eval()
    xt = torch.from_numpy(normalize_u(x)).view(1, -1, 1).to(device)
    return model(xt).view(-1).cpu().detach().numpy() * y_std

def val_rmse(): #calcolo dell RMSE
    se, n = 0.0, 0
    for x, y in zip(x_va, y_va):
        e = predict(x) - y
        se += float((e ** 2).sum()); n += len(y)
    return (se / n) ** 0.5

best, best_state, bad = float("inf"), None, 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    tot, nb = 0.0, 0
    for x, y in zip(x_train_list, y_train_list):
        optimizer.zero_grad()

        x_batch = torch.from_numpy(x).to(device).view(1, -1, 1)
        y_train_t = torch.from_numpy(y).to(device).view(1, -1, 1)

        y_pred = model(x_batch)

        loss = criterion(y_pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        tot += loss.item()
        nb+=1

    v = val_rmse()
    scheduler.step(v)
    if v < best:
        best, bad = v, 0
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoca [{epoch+1}/{EPOCHS}] train MSE (norm): {tot/nb:.5f} | "
                  f"val RMSE: {v:.4f} Nm | best: {best:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")
    if bad >= PATIENCE: #per early stopping
        print(f"Early stopping all'epoca {epoch+1}")
        break

model.load_state_dict(best_state)
torch.save({
    'model_state_dict':  model.state_dict(),
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'output_dim': OUTPUT_DIM,
    'u_std': u_std,
    'y_std': y_std,
}, 'lstm_checkpoint.pth')

print("Modello salvato in: lstm_checkpoint.pth")