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

CHANNELS = 16
KERNEL_SIZE = 9
DILATIONS = (1, 6, 36, 216)

WINDOW = 2048          # campioni di output per finestra
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-3
MAX_GRAD_NORM = 5.0
VAL_FRACTION = 0.2
ES_PATIENCE = 40       # early stopping, in epoche

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ----------------------------------------------------------------------
# MODELLO: convoluzioni dilatate causali, senza padding interno
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

model = TCN(CHANNELS, KERNEL_SIZE, DILATIONS).to(device)
RF = model.receptive_field
PAD = RF - 1 #calcolo del pad in maniera congruente al campo recettivo
print(model, f"\nCampo recettivo: {RF} campioni")

# ----------------------------------------------------------------------
# DATI
# ----------------------------------------------------------------------
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
print(f"Train: {len(x_tr)} traiettorie, validazione: {len(x_va)}")

# calcolo della deviazione standard
x_scale = float(np.concatenate(x_tr).std())
y_scale = float(np.concatenate(y_tr).std())

def prepad(x):
    #padding dei campioni iniziali per garantire sufficiente spostamento in relazione al padding
    return np.concatenate([np.zeros(PAD, np.float32), x / x_scale])

xp_tr = [prepad(x) for x in x_tr]
yn_tr = [y / y_scale for y in y_tr]
assert all(len(x) >= WINDOW for x in x_tr), "traiettoria piu' corta della finestra"

def epoch_batches():
    windows = []
    for k, x in enumerate(x_tr):
        for _ in range(max(1, len(x) // WINDOW)):   # numero di finestre ~ durata
            windows.append((k, random.randint(0, len(x) - WINDOW)))
    random.shuffle(windows)
    for i in range(0, len(windows), BATCH_SIZE):
        b = windows[i:i + BATCH_SIZE]
        xb = np.stack([xp_tr[k][s:s + WINDOW + PAD] for k, s in b])
        yb = np.stack([yn_tr[k][s:s + WINDOW] for k, s in b])
        yield (torch.from_numpy(xb).unsqueeze(1).to(device),
               torch.from_numpy(yb).unsqueeze(1).to(device))

@torch.no_grad()
def predict(x):

    model.eval()
    xt = torch.from_numpy(prepad(x)).view(1, 1, -1).to(device)
    return model(xt).view(-1).cpu().numpy() * y_scale

def val_rmse(): #calcolo dell RMSE
    se, n = 0.0, 0
    for x, y in zip(x_va, y_va):
        e = predict(x) - y
        se += float((e ** 2).sum()); n += len(y)
    return (se / n) ** 0.5


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
criterion = nn.MSELoss()

best, best_state, bad = float("inf"), None, 0
for epoch in range(EPOCHS):
    model.train()
    tot, nb = 0.0, 0
    for xb, yb in epoch_batches():
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        tot += loss.item(); nb += 1

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
    if bad >= ES_PATIENCE: #per early stopping
        print(f"Early stopping all'epoca {epoch+1}")
        break

model.load_state_dict(best_state)
torch.save({
    "model_state_dict": model.state_dict(),
    "channels": CHANNELS,
    "kernel_size": KERNEL_SIZE,
    "dilations": DILATIONS,
    "receptive_field": RF,
    "x_scale": x_scale,
    "y_scale": y_scale,
    "best_val_rmse_Nm": best,
    "seed": SEED,
}, "tcn_checkpoint.pth")