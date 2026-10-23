import random
import numpy as np
import pandas
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------
SEED = 0
INPUT_A_CSV = "../Datasets/trainingA.csv"
INPUT_B_CSV = "../Datasets/trainingB.csv"
INPUT_COLS = ["velocity_rad_s", "tau_load"]   # ordine dei canali

INPUT_DIM = len(INPUT_COLS)
HIDDEN_DIM = 16
OUTPUT_DIM = 1

VAL_FRACTION = 0.2
PATIENCE = 40
EPOCHS = 300
LR = 1e-3
MAX_GRAD_NORM = 5.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# ----------------------------------------------------------------------
# MODELLO
# ----------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):      # x: [B, T, C] -> [B, T, 1]
        lstm_out, _ = self.lstm(x, hidden)
        return self.linear(lstm_out)


# ----------------------------------------------------------------------
# DATI
# ----------------------------------------------------------------------
def load(csv):
    df = pandas.read_csv(csv)
    xs, ys = [], []
    for _, g in df.groupby("traj_id", sort=True):
        xs.append(g[INPUT_COLS].to_numpy(np.float32).T)           # [C, T]
        ys.append(g["torque_friction_estimated_Nm"].to_numpy(np.float32))
    return xs, ys

# split delle traiettorie in training e validazione
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
print(f"Train: {len(x_tr)} traiettorie, validazione: {len(x_va)}")

# statistiche PER CANALE (calcolate solo sul training), forma [C, 1]
x_cat = np.concatenate(x_tr, axis=1)
u_mean = x_cat.mean(axis=1, keepdims=True).astype(np.float32)
u_std = x_cat.std(axis=1, keepdims=True).astype(np.float32)
u_std = np.where(u_std > 1e-8, u_std, 1.0).astype(np.float32)   # evita divisioni per 0

y_std = float(np.concatenate(y_tr).std())
y_std = y_std if y_std > 1e-8 else 1.0

def normalize_u(x):                     # [C, T] -> [C, T] float32
    return ((x) / u_std).astype(np.float32)

def to_input(x_norm):                   # [C, T] -> tensor [1, T, C]
    return torch.from_numpy(np.ascontiguousarray(x_norm.T)).unsqueeze(0).to(device)

def to_target(y_norm):                  # [T] -> tensor [1, T, 1]
    return torch.from_numpy(y_norm).view(1, -1, 1).to(device)

train_data = [(to_input(normalize_u(x)), to_target(y / y_std)) for x, y in zip(x_tr, y_tr)]

# ----------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------
model = LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=15
)
criterion = nn.MSELoss()

@torch.no_grad()
def predict(x):                         # x: [C, T] grezzo -> [T] in Nm
    model.eval()
    return model(to_input(normalize_u(x))).view(-1).cpu().numpy() * y_std

def val_rmse():
    se, n = 0.0, 0
    for x, y in zip(x_va, y_va):
        e = predict(x) - y
        se += float((e ** 2).sum())
        n += len(y)
    return (se / n) ** 0.5

best, best_state, bad = float("inf"), None, 0

for epoch in range(EPOCHS):
    model.train()
    random.shuffle(train_data)          # nuovo ordine a ogni epoca
    tot, nb = 0.0, 0
    for xt, yt in train_data:
        optimizer.zero_grad()
        loss = criterion(model(xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        tot += loss.item()
        nb += 1

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
    if bad >= PATIENCE:
        print(f"Early stopping all'epoca {epoch+1}")
        break

model.load_state_dict(best_state)
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "output_dim": OUTPUT_DIM,
    "input_cols": INPUT_COLS,
    "u_mean": u_mean.squeeze(1).tolist(),
    "u_std": u_std.squeeze(1).tolist(),
    "y_std": y_std,
    "best_val_rmse_Nm": best,
    "seed": SEED,
}, "lstm_checkpoint.pth")

print("Modello salvato in: lstm_checkpoint.pth")