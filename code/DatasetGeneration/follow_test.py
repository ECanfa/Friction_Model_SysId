import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import imageio
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------
SEED = 2
MODES = [True, False]                   # True = con retroazione attrito (TCN), False = senza
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS = [f"../CNN/tcn_checkpoint_{i}.pth" for i in range(7)]

MODEL_PATH = "../kuka_iiwa_14/scene.xml"
OUTPUT_CSV = "../Datasets/test_{}.csv"            # control_True.csv / control_False.csv

ROBODK_CSV = "/home/emanuele/Desktop/Prog1_joints.csv"   # 7 colonne: giunti in gradi
DT_RDK = 0.04                                          # s, passo di campionamento del CSV

SIM_DT = 0.001
N_JOINTS = 7

# Visualizzazione (replay a fine simulazione)
SHOW_VIEWER = True                      # finestra interattiva (su macOS lanciare con `mjpython`)
SAVE_VIDEO = True                       # salva un mp4 per ogni modalita'
VIDEO_PATH = "../Datasets/velocity_tracking_{}.mp4"   # _True.mp4 / _False.mp4
VIDEO_FPS = 50
VIDEO_SIZE = (640, 480)                 # (larghezza, altezza)

# ----------------------------------------------------------------------
# PARAMETRI CONTROLLO / ATTRITO
# ----------------------------------------------------------------------
TORQUE_LIMIT = np.array([320.0, 320.0, 180.0, 180.0, 110.0, 40.0, 40.0])
SCALE = TORQUE_LIMIT / 320.0

KP_VEL = 150.0 * SCALE
KI_VEL = 18.0 * SCALE

FC0 = 0.025 * TORQUE_LIMIT
FS0 = 1.5 * FC0
FV0 = 0.8 * SCALE
SIGMA_0_0 = 1e4 * SCALE
SIGMA_1_0 = 100.0 * SCALE
VS = 0.1

TAU_REF = 50.0 * SCALE
K_LOAD_C = 0.5
K_LOAD_S = 0.5
K_LOAD_SIGMA0 = 0.3
LOAD_MAX = 3.0
LOAD_FILTER_ALPHA = 0.02

ARMATURE = np.array([0.5, 0.5, 0.3, 0.3, 0.2, 0.1, 0.1])

# ----------------------------------------------------------------------
# TCN
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

    def forward(self, x):
        return self.final(self.body(x))

modelli, x_mean, x_std, y_scale, RFS = [], [], [], [], []
if True in MODES:
    for i in range(N_JOINTS):
        ckpt = torch.load(CHECKPOINTS[i], map_location=device, weights_only=True)
        net = TCN(ckpt["in_channels"], ckpt["channels"], ckpt["kernel_size"],
                  tuple(ckpt["dilations"])).to(device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        modelli.append(net)
        RFS.append(net.receptive_field)
        x_mean.append(np.asarray(ckpt["x_mean"], np.float32).reshape(-1, 1))
        x_std.append(np.asarray(ckpt["x_std"], np.float32).reshape(-1, 1))
        y_scale.append(ckpt["y_scale"])

@torch.no_grad()
def predict_last(log, step, i):
    n = N_JOINTS
    lo = max(0, step + 1 - RFS[i])
    x = np.stack([log[lo:step + 1, 1 + n + i],          # velocita' misurata
                  log[lo:step + 1, 1 + 3 * n + i]])     # coppia di carico filtrata
    xn = ((x - x_mean[i]) / x_std[i]).astype(np.float32)
    missing = RFS[i] - xn.shape[1]
    if missing > 0:
        xn = np.concatenate([np.repeat(xn[:, :1], missing, axis=1), xn], axis=1)
    xt = torch.from_numpy(xn).unsqueeze(0).to(device)
    return float(modelli[i](xt)[0, 0, -1]) * y_scale[i]

# ----------------------------------------------------------------------
# MODELLO E ATTUATORI (controllo in coppia)
# ----------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
model.opt.timestep = SIM_DT

joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, N_JOINTS + 1)]
actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                         for i in range(1, N_JOINTS + 1)])
qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
qvel_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])

for k, act in enumerate(actuator_ids):
    model.actuator_gaintype[act] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_gainprm[act, 0] = 1.0
    model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_NONE
    model.actuator_biasprm[act, :] = 0.0
    model.actuator_ctrllimited[act] = 1
    model.actuator_ctrlrange[act] = [-TORQUE_LIMIT[k], TORQUE_LIMIT[k]]

model.dof_armature[qvel_adr] = ARMATURE
mujoco.mj_setConst(model, data)

# ----------------------------------------------------------------------
# VELOCITA' DEI GIUNTI DAL DATASET (posizioni in gradi -> rad/s)
# ----------------------------------------------------------------------
q = np.deg2rad(np.loadtxt(ROBODK_CSV, delimiter=",")[:, :N_JOINTS])
qd = np.gradient(q, DT_RDK, axis=0)

t_rdk = np.arange(len(q)) * DT_RDK
t_sim = np.arange(int(t_rdk[-1] / SIM_DT)) * SIM_DT
qd_ref = np.column_stack([np.interp(t_sim, t_rdk, qd[:, j]) for j in range(N_JOINTS)])
Q_START = q[0].copy()
n_steps = len(qd_ref)
print(f"Riferimento: {n_steps * SIM_DT:.2f} s, v_max = {np.abs(qd_ref).max():.2f} rad/s")

# ----------------------------------------------------------------------
# COLONNE DEL LOG (l'ordine serve a predict_last)
# ----------------------------------------------------------------------
cols = (["t"]
        + [f"position_{j}_rad" for j in range(N_JOINTS)]
        + [f"velocity_rad_{j}_s" for j in range(N_JOINTS)]
        + [f"velocity_des_{j}_s" for j in range(N_JOINTS)]
        + [f"tau_load_{j}" for j in range(N_JOINTS)]
        + [f"tau_friction_estimated_{j}" for j in range(N_JOINTS)]
        + [f"torque_friction_true_{j}" for j in range(N_JOINTS)]
        + [f"errore_{j}" for j in range(N_JOINTS)])

# ----------------------------------------------------------------------
# SIMULAZIONE
# ----------------------------------------------------------------------
def run(use_tcn):
    rng = np.random.default_rng(SEED)             # stesso rumore in entrambe le modalita'
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_adr] = Q_START
    mujoco.mj_forward(model, data)

    tau_inertia = np.zeros(model.nv)
    integral_error = np.zeros(N_JOINTS)
    z = np.zeros(N_JOINTS)
    tau_load_filt = np.zeros(N_JOINTS)
    friction = np.zeros(N_JOINTS)
    log = np.empty((n_steps, len(cols)))

    for step in range(n_steps):

        # 1. CONTROLLO: PI di velocita' -> coppia
        v_meas = data.qvel[qvel_adr]
        v_des = qd_ref[step]
        err = v_des - v_meas

        tau_cmd = KP_VEL * err + KI_VEL * integral_error + data.qfrc_bias[qvel_adr]
        if use_tcn:
            tau_cmd = tau_cmd - friction

        integrate = (np.abs(tau_cmd) < TORQUE_LIMIT) | (err * tau_cmd < 0)
        integral_error += np.where(integrate, err * SIM_DT, 0.0)
        data.ctrl[actuator_ids] = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)

        # 2. COPPIA DI CARICO, filtrata
        mujoco.mj_mulM(model, data, tau_inertia, data.qacc)
        tau_load = tau_inertia[qvel_adr] + data.qfrc_bias[qvel_adr]
        tau_load_filt += LOAD_FILTER_ALPHA * (np.abs(tau_load) - tau_load_filt)
        load_ratio = np.minimum(tau_load_filt / TAU_REF, LOAD_MAX)

        # 3. ATTRITO LuGre con parametri adattati al carico
        Fc = FC0 * (1.0 + K_LOAD_C * load_ratio)
        Fs = np.maximum(FS0 * (1.0 + K_LOAD_S * load_ratio), 1.05 * Fc)
        sigma0 = SIGMA_0_0 * (1.0 + K_LOAD_SIGMA0 * load_ratio)
        sigma1 = SIGMA_1_0 * np.sqrt(sigma0 / SIGMA_0_0)

        qdot = v_meas
        g = Fc + (Fs - Fc) * np.exp(-(qdot / VS) ** 2)
        r = np.abs(qdot) * sigma0 / g
        z_new = (z + SIM_DT * qdot) / (1.0 + SIM_DT * r)
        zdot = qdot - r * z_new
        signal = sigma0 * z_new + sigma1 * zdot + FV0 * qdot
        tau_friction = signal + rng.normal(0.0, 0.02 * np.abs(signal))
        z = z_new

        data.qfrc_applied[qvel_adr] = -tau_friction

        # 4. DINAMICA E LOG
        mujoco.mj_forward(model, data)
        mujoco.mj_mulM(model, data, tau_inertia, data.qacc)
        tau_est = (tau_inertia[qvel_adr] + data.qfrc_bias[qvel_adr]
                   - data.qfrc_constraint[qvel_adr] - data.qfrc_actuator[qvel_adr])

        log[step] = np.concatenate([
            [data.time], data.qpos[qpos_adr], v_meas, v_des,
            tau_load_filt, tau_est, tau_friction, v_des - v_meas,
        ])

        mujoco.mj_step(model, data)

        # 5. STIMA TCN per lo step successivo (usa solo il passato)
        if use_tcn:
            for i in range(N_JOINTS):
                friction[i] = predict_last(log, step, i)

    return log

replays = {}                            # modalita' -> pose simulate [n_steps, 7]

for use_tcn in MODES:
    print(f"Simulazione con retroazione attrito = {use_tcn}")
    log = run(use_tcn)
    if not np.isfinite(log).all():
        raise RuntimeError("Valori non finiti: simulazione instabile")
    df = pd.DataFrame(log, columns=cols)
    df.insert(0, "traj_id", 0)
    out = OUTPUT_CSV.format(use_tcn)
    df.to_csv(out, index=False)
    replays[use_tcn] = log[:, 1:1 + N_JOINTS].copy()
    rmse = np.sqrt(np.mean(log[:, -N_JOINTS:] ** 2, axis=0))
    print(f"  salvato {out} | RMSE velocita' per giunto [rad/s]: {np.round(rmse, 4)}")

# ----------------------------------------------------------------------
# VISUALIZZAZIONE (replay delle pose simulate) E VIDEO MP4
# ----------------------------------------------------------------------
data_vis = mujoco.MjData(model)
STRIDE = int(round(1.0 / (VIDEO_FPS * SIM_DT)))

def apply_camera(cam):
    cam.lookat[:] = [0.3, 0.0, 0.4]
    cam.distance = 2.2
    cam.azimuth = 140
    cam.elevation = -20

def set_pose(q):
    data_vis.qpos[qpos_adr] = q
    mujoco.mj_forward(model, data_vis)

if SAVE_VIDEO:
    renderer = mujoco.Renderer(model, height=VIDEO_SIZE[1], width=VIDEO_SIZE[0])
    cam = mujoco.MjvCamera()
    apply_camera(cam)
    for use_tcn, q_replay in replays.items():
        frames = []
        for k in range(0, len(q_replay), STRIDE):
            set_pose(q_replay[k])
            renderer.update_scene(data_vis, camera=cam)
            frames.append(renderer.render())
        path = VIDEO_PATH.format(use_tcn)
        imageio.mimsave(path, frames, fps=VIDEO_FPS)
        print(f"Video salvato in {path}")
    renderer.close()

if SHOW_VIEWER:
    for use_tcn, q_replay in replays.items():
        print(f"Replay nel viewer: retroazione attrito = {use_tcn} (chiudi la finestra per passare al successivo)")
        with mujoco.viewer.launch_passive(model, data_vis) as v:
            apply_camera(v.cam)
            for k in range(0, len(q_replay), STRIDE):
                if not v.is_running():
                    break
                set_pose(q_replay[k])
                v.sync()
                time.sleep(1.0 / VIDEO_FPS)
            time.sleep(1.0)