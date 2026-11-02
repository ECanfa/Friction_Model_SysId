import numpy as np
from robodk.robolink import Robolink, ITEM_TYPE_PROGRAM

PROG_NAME = "Prog1"
OUT_FILE = "/home/emanuele/Desktop/Prog1_joints.csv"
DT = 0.04          # s, passo di campionamento (200 Hz) -> usato anche dallo script MuJoCo
N_JOINTS = 7

RDK = Robolink(port=20500)
prog = RDK.Item(PROG_NAME, ITEM_TYPE_PROGRAM)
if not prog.Valid():
    raise SystemExit(f"Programma '{PROG_NAME}' non trovato")


error_msg, jl, error_code = prog.InstructionListJoints(mm_step=1, deg_step=1, time_step=DT)
print("RoboDK:", error_msg, error_code)

arr = np.array(jl.rows if hasattr(jl, "rows") else jl, dtype=float)
if arr.shape[0] <= 30 < arr.shape[1] or (arr.shape[0] < arr.shape[1] and arr.shape[0] <= 30):
    arr = arr.T                                   # una riga per campione
q_deg = arr[:, :N_JOINTS]

np.savetxt(OUT_FILE, q_deg, delimiter=",", fmt="%.6f")
print(f"Salvati {len(q_deg)} campioni ({len(q_deg) * DT:.2f} s se il passo e' davvero {DT} s) in {OUT_FILE}")
print("Prima riga:", q_deg[0])
print("Ultima riga:", q_deg[-1])