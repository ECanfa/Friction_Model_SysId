import visualtorch
import torch.nn as nn
import torch
from torchview import draw_graph

giunto = 1

# Forza temporaneamente la CPU per evitare conflitti di device con visualtorch
device = "cpu" 

CHECKPOINT = f"../LSTMId/lstm_checkpoint_{giunto}.pth"
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

# Calcola la tupla corretta
input_shape = (1, 2, 2)


dummy_input = torch.randn(input_shape) # Input di prova con le stesse dimensioni
torch.onnx.export(model, dummy_input, "LSTM.onnx")
