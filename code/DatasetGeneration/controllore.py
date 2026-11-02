import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import imageio
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# CONFIGURAZIONE GENERALE
# ----------------------------------------------------------------------
SEED = 2
rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS = [f"../CNN/tcn_checkpoint_{i}.pth" for i in range(7)]
USE_TCN_COMP = True
MODEL_PATH = "../kuka_iiwa_14/scene.xml"
OUTPUT_CSV = f"../Datasets/control_{USE_TCN_COMP}.csv"

# Visualizzazione (avviene a fine simulazione, come replay a tempo reale)
SHOW_VIEWER = True                      # finestra interattiva (su macOS lanciare con `mjpython`)
SAVE_VIDEO = True                       # salva anche un mp4
VIDEO_PATH = f"../Datasets/pick_place_{USE_TCN_COMP}.mp4"
VIDEO_FPS = 50
SHOW_MARKERS = True                     # sfere SOLO visive (no collisioni) su punto di presa e di rilascio

                     # compensazione dell'attrito con le TCN

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

    def forward(self, x):                 # x: [B, C, L + RF - 1] -> [B, 1, L]
        return self.final(self.body(x))

modelli, x_mean, x_std, y_scale, RFS = [], [], [], [], []
for i in range(7):
    ckpt = torch.load(CHECKPOINTS[i], map_location=device, weights_only=True)
    net = TCN(ckpt["in_channels"], ckpt["channels"], ckpt["kernel_size"],
              tuple(ckpt["dilations"])).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    modelli.append(net)
    RFS.append(net.receptive_field)
    x_mean.append(np.asarray(ckpt["x_mean"], np.float32).reshape(-1, 1))   # [C, 1]
    x_std.append(np.asarray(ckpt["x_std"], np.float32).reshape(-1, 1))     # [C, 1]
    y_scale.append(ckpt["y_scale"])

# ----------------------------------------------------------------------
# PARAMETRI ROBOT / CONTROLLO / ATTRITO
# ----------------------------------------------------------------------
SIM_DT = 0.001
N_JOINTS = 7

TORQUE_LIMIT = np.array([320.0, 320.0, 180.0, 180.0, 110.0, 40.0, 40.0])
SCALE = TORQUE_LIMIT / 320.0

# Controllo di velocita' (PI), scalato per giunto
KP_VEL = 150.0 * SCALE
KI_VEL = 18.0 * SCALE
# Anello esterno di posizione: v_des = qd_ref + KP_POS * (q_ref - q)
KP_POS = 8.0

# Attrito LuGre
FC0 = 0.025 * TORQUE_LIMIT
FS0 = 1.5 * FC0
FV0 = 0.8 * SCALE
SIGMA_0_0 = 1e4 * SCALE
SIGMA_1_0 = 100.0 * SCALE
VS = 0.1

# Dipendenza dal carico
TAU_REF = 50.0 * SCALE
K_LOAD_C = 0.5
K_LOAD_S = 0.5
K_LOAD_SIGMA0 = 0.3
LOAD_MAX = 3.0
LOAD_FILTER_ALPHA = 0.02

ARMATURE = np.array([0.5, 0.5, 0.3, 0.3, 0.2, 0.1, 0.1])

JOINT_MARGIN = 0.15
CONSTRAINT_TOL = 1e-9

# ----------------------------------------------------------------------
# PARAMETRI DEL TASK PICK-AND-PLACE
# ----------------------------------------------------------------------
NUM_TRAJ = 1
MAX_ATTEMPTS = 3 * NUM_TRAJ

CTRL_HZ = 200                           # frequenza di generazione del riferimento
CTRL_DT = 1.0 / CTRL_HZ

R_RANGE = (0.45, 0.65)                  # raggio (m) dei punti nel piano orizzontale
PICK_ANGLE = np.deg2rad([-70.0, -25.0]) # settore di presa
PLACE_ANGLE = np.deg2rad([25.0, 70.0])  # settore di rilascio
Z_PICK_RANGE = (0.25, 0.35)             # quota della flangia alla presa/rilascio (m)
HOVER = 0.20                            # quota extra del punto "sopra" (m)
START_CYL = np.array([0.45, 0.0, 0.55]) # posa iniziale (r, theta, z)

# durata (s) dei segmenti: sopra-presa, discesa, pausa presa, salita,
#                          trasporto, discesa, pausa rilascio, salita, ritorno
SEG_DURATIONS = np.array([1.6, 0.8, 0.5, 0.8, 2.0, 0.8, 0.5, 0.8, 1.6])
TIME_SCALE_RANGE = (1.0, 2.5)           # >1 = piu' lento (piu' campioni in zona Stribeck)
V_MAX_REF = 1.2                         # rad/s, velocita' massima ammessa nel riferimento
IK_TOL = 3e-3                           # m

# IK differenziale
W_ROT = 0.3
IK_DAMPING = 0.05
K_NULL = 0.3
Q_NOM = np.array([0.0, 0.6, 0.0, -1.4, 0.0, 0.9, 0.0])   # postura "preferita" (nullspace)
DOWN = np.array([0.0, 0.0, -1.0])

# ----------------------------------------------------------------------
# MODELLO E ATTUATORI (controllo in coppia)
# ----------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
model.opt.timestep = SIM_DT

joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, N_JOINTS + 1)]
actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                         for i in range(1, N_JOINTS + 1)])
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
qvel_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])

JOINT_LO = np.array([model.jnt_range[j][0] if model.jnt_limited[j] else -np.inf for j in joint_ids])
JOINT_HI = np.array([model.jnt_range[j][1] if model.jnt_limited[j] else np.inf for j in joint_ids])

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
# CINEMATICA INVERSA (su un MjData separato, non tocca la simulazione)
# ----------------------------------------------------------------------
ik_data = mujoco.MjData(model)
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

# asse locale del sito che a q=0 punta verso l'alto (= asse della flangia): lo vogliamo verso il basso
ik_data.qpos[:] = 0.0
mujoco.mj_forward(model, ik_data)
TOOL_AXIS_LOCAL = ik_data.site_xmat[site_id].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])

def to_cyl(p):
    return np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0]), p[2]])

def from_cyl(c):
    return np.array([c[0] * np.cos(c[1]), c[0] * np.sin(c[1]), c[2]])

def solve_ik(target, q_init, iters):
    """Posizione 3D del sito + asse utensile verso il basso (imbardata libera).
    Numero di iterazioni FISSO: il riferimento risulta liscio."""
    q = q_init.copy()
    for _ in range(iters):
        ik_data.qpos[qpos_adr] = q
        mujoco.mj_forward(model, ik_data)
        pos_err = target - ik_data.site_xpos[site_id]
        R = ik_data.site_xmat[site_id].reshape(3, 3)
        rot_err = np.cross(R @ TOOL_AXIS_LOCAL, DOWN)
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, qvel_adr], W_ROT * jacr[:, qvel_adr]])
        err = np.concatenate([pos_err, W_ROT * rot_err])
        Jp = J.T @ np.linalg.inv(J @ J.T + IK_DAMPING ** 2 * np.eye(6))
        dq = Jp @ err + (np.eye(N_JOINTS) - Jp @ J) @ (K_NULL * (Q_NOM - q))
        dq = np.clip(dq, -0.2, 0.2)
        q = np.clip(q + dq, JOINT_LO + 0.05, JOINT_HI - 0.05)
    ik_data.qpos[qpos_adr] = q
    mujoco.mj_forward(model, ik_data)
    return q, float(np.linalg.norm(target - ik_data.site_xpos[site_id]))

# Configurazione iniziale: utensile verso il basso, davanti al robot
START_POS = from_cyl(START_CYL)
Q_START, err0 = solve_ik(START_POS, Q_NOM, 400)
if err0 > IK_TOL:
    print(f"ATTENZIONE: IK della posa iniziale non convergente (err = {err0*1e3:.1f} mm). "
          f"Modifica START_CYL o Q_NOM.")

def reset_traj():
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_adr] = Q_START
    mujoco.mj_forward(model, data)

# ----------------------------------------------------------------------
# CONTROLLO DI STABILITA' NUMERICA
# ----------------------------------------------------------------------
reset_traj()
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, data, M)
J_eff = np.diag(M)[qvel_adr]

sigma_gain = 1.0 + K_LOAD_SIGMA0 * LOAD_MAX
c_tot = KP_VEL + np.sqrt(sigma_gain) * SIGMA_1_0 + FV0
margin = SIM_DT * c_tot / J_eff
w_dt = SIM_DT * np.sqrt(sigma_gain * SIGMA_0_0 / J_eff)

print("Giunto | J_eff [kg m^2] | (KP+s1+FV)*dt/J | w*dt   (meglio < 0.5; instabile oltre ~2)")
for k in range(N_JOINTS):
    flag = "  <-- ATTENZIONE" if max(margin[k], w_dt[k]) > 0.5 else ""
    print(f"  {k+1}    | {J_eff[k]:10.4f}     | {margin[k]:.3f}           | {w_dt[k]:.3f}{flag}")

# ----------------------------------------------------------------------
# GENERAZIONE DEL RIFERIMENTO PICK-AND-PLACE
# ----------------------------------------------------------------------
def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5

def smooth(x, w=7):
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(w) / w
    return np.column_stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])])

def sample_targets():
    """Punti di presa e rilascio in coordinate cilindriche (r, theta, z)."""
    r1, r2 = rng.uniform(*R_RANGE, size=2)
    z1, z2 = rng.uniform(*Z_PICK_RANGE, size=2)
    pick = np.array([r1, rng.uniform(*PICK_ANGLE), z1])
    place = np.array([r2, rng.uniform(*PLACE_ANGLE), z2])
    if rng.random() < 0.5:                       # a volte si inverte il verso dello spostamento
        pick, place = place, pick
    return pick, place

def build_cartesian_path(c_pick, c_place, time_scale):
    up = np.array([0.0, 0.0, HOVER])
    knots = [START_CYL, c_pick + up, c_pick, c_pick, c_pick + up,
             c_place + up, c_place, c_place, c_place + up, START_CYL]
    pts = []
    for a, b, T in zip(knots[:-1], knots[1:], SEG_DURATIONS * time_scale):
        n = max(int(round(T * CTRL_HZ)), 1)
        for s in min_jerk(np.arange(n) / n):
            pts.append(from_cyl(a + s * (b - a)))   # interpolazione cilindrica: il trasporto e' un arco
    pts.append(from_cyl(knots[-1]))
    return np.array(pts)

def make_reference(c_pick, c_place):
    """Restituisce q_ref, qd_ref campionati a SIM_DT e il massimo errore IK."""
    ts = rng.uniform(*TIME_SCALE_RANGE)
    for _ in range(5):
        pts = build_cartesian_path(c_pick, c_place, ts)
        q = np.zeros((len(pts), N_JOINTS))
        q[0] = Q_START
        worst = 0.0
        for k in range(1, len(pts)):
            q[k], e = solve_ik(pts[k], q[k - 1], 5)
            worst = max(worst, e)
        q = smooth(q)
        qd = np.gradient(q, CTRL_DT, axis=0)
        vmax = np.abs(qd).max()
        if vmax <= V_MAX_REF:
            break
        ts *= 1.05 * vmax / V_MAX_REF               # troppo veloce: rallento e rigenero

    t_ctrl = np.arange(len(q)) * CTRL_DT
    t_sim = np.arange(int(t_ctrl[-1] / SIM_DT)) * SIM_DT
    q_ref = np.column_stack([np.interp(t_sim, t_ctrl, q[:, j]) for j in range(N_JOINTS)])
    qd_ref = np.column_stack([np.interp(t_sim, t_ctrl, qd[:, j]) for j in range(N_JOINTS)])
    return q_ref, qd_ref, worst, vmax

def limit_near_joint_range(v_des, qpos):
    q = qpos[qpos_adr]
    v = v_des.copy()
    near_hi = (v > 0) & (q > JOINT_HI - JOINT_MARGIN)
    near_lo = (v < 0) & (q < JOINT_LO + JOINT_MARGIN)
    v[near_hi] *= np.clip((JOINT_HI[near_hi] - q[near_hi]) / JOINT_MARGIN, 0.0, 1.0)
    v[near_lo] *= np.clip((q[near_lo] - JOINT_LO[near_lo]) / JOINT_MARGIN, 0.0, 1.0)
    return v

# ----------------------------------------------------------------------
# STIMA DELL'ATTRITO CON LE TCN (solo finestra recente, causale)
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_last(log, step, i):
    n = N_JOINTS
    lo = max(0, step + 1 - RFS[i])
    x = np.stack([log[lo:step + 1, 1 + n + i],          # velocita' misurata
                  log[lo:step + 1, 1 + 3 * n + i]])     # coppia di carico filtrata
    xn = ((x - x_mean[i]) / x_std[i]).astype(np.float32)
    missing = RFS[i] - xn.shape[1]
    if missing > 0:                                     # stesso padding dello script originale
        xn = np.concatenate([np.repeat(xn[:, :1], missing, axis=1), xn], axis=1)
    xt = torch.from_numpy(xn).unsqueeze(0).to(device)
    return float(modelli[i](xt)[0, 0, -1]) * y_scale[i]

# ----------------------------------------------------------------------
# COLONNE DEL LOG
# ----------------------------------------------------------------------
cols = (["t"]
        + [f"position_{j}_rad" for j in range(N_JOINTS)]
        + [f"velocity_rad_{j}_s" for j in range(N_JOINTS)]
        + [f"velocity_des_{j}_s" for j in range(N_JOINTS)]
        + [f"tau_load_{j}" for j in range(N_JOINTS)]
        + [f"tau_friction_estimated_{j}" for j in range(N_JOINTS)]
        + [f"torque_friction_true_{j}" for j in range(N_JOINTS)]
        + [f"constraint_{j}" for j in range(N_JOINTS)]
        + [f"position_des_{j}_rad" for j in range(N_JOINTS)]
        + ["ee_x", "ee_y", "ee_z"]
        + [f"errore_{j}" for j in range(N_JOINTS)])
constraint_cols = [f"constraint_{j}" for j in range(N_JOINTS)]

# ----------------------------------------------------------------------
# LOOP SULLE TRAIETTORIE
# ----------------------------------------------------------------------
tau_inertia = np.zeros(model.nv)
accepted, attempts, discarded = 0, 0, 0
replays = []

while accepted < NUM_TRAJ and attempts < MAX_ATTEMPTS:
    attempts += 1

    c_pick, c_place = sample_targets()
    q_ref, qd_ref, ik_err, vmax = make_reference(c_pick, c_place)
    if ik_err > IK_TOL:
        discarded += 1
        print(f"[tentativo {attempts}] scartata: IK non convergente ({ik_err*1e3:.1f} mm)")
        continue

    n_steps = len(q_ref)
    reset_traj()
    integral_error = np.zeros(N_JOINTS)
    z = np.zeros(N_JOINTS)
    tau_load_filt = np.zeros(N_JOINTS)
    friction = np.zeros(N_JOINTS)
    log = np.empty((n_steps, len(cols)))
    print(f"[tentativo {attempts}] durata {n_steps*SIM_DT:.1f} s, v_max rif = {vmax:.2f} rad/s")

    for step in range(n_steps):

        # 1. CONTROLLO: anello di posizione esterno -> PI di velocita' -> coppia
        v_meas = data.qvel[qvel_adr].copy()
        v_des = limit_near_joint_range(
            qd_ref[step] + KP_POS * (q_ref[step] - data.qpos[qpos_adr]), data.qpos)
        err = v_des - v_meas

        tau_cmd = KP_VEL * err + KI_VEL * integral_error + data.qfrc_bias[qvel_adr] - friction

        integrate = (np.abs(tau_cmd) < TORQUE_LIMIT) | (err * tau_cmd < 0)
        integral_error += np.where(integrate, err * SIM_DT, 0.0)

        data.ctrl[actuator_ids] = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)

        # 2. COPPIA DI CARICO (qacc dello step precedente), filtrata
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

        errore = v_des - v_meas
        log[step] = np.concatenate([
            [data.time],
            data.qpos[qpos_adr],
            v_meas,
            v_des,
            tau_load_filt,
            tau_est,
            tau_friction,
            data.qfrc_constraint[qvel_adr],
            q_ref[step],
            data.site_xpos[site_id],
            errore
        ])

        mujoco.mj_step(model, data)

        # 5. STIMA TCN per lo step successivo (usa solo il passato)
        if USE_TCN_COMP:
            for i in range(N_JOINTS):
                friction[i] = predict_last(log, step, i)

    # --- accettazione della traiettoria ---
    df = pd.DataFrame(log, columns=cols)
    has_constraints = df[constraint_cols].abs().to_numpy().max() > CONSTRAINT_TOL
    is_finite = np.isfinite(log).all()

    if has_constraints or not is_finite:
        discarded += 1
        motivo = "vincoli attivi" if has_constraints else "valori non finiti (instabilita')"
        print(f"[tentativo {attempts}] scartata: {motivo}")
        continue

    df.insert(0, "traj_id", accepted)
    df.to_csv(OUTPUT_CSV, mode="w" if accepted == 0 else "a", header=(accepted == 0), index=False)
    replays.append(dict(q=log[:, 1:1 + N_JOINTS].copy(),
                        pick=from_cyl(c_pick), place=from_cyl(c_place)))
    accepted += 1
    print(f"[tentativo {attempts}] ok ({accepted}/{NUM_TRAJ})")

if accepted == 0:
    raise RuntimeError("Tutte le traiettorie sono state scartate")
if accepted < NUM_TRAJ:
    print(f"ATTENZIONE: accettate solo {accepted}/{NUM_TRAJ} traiettorie in {attempts} tentativi")

print(f"Dataset salvato in {OUTPUT_CSV}: {accepted} traiettorie ({discarded} scartate su {attempts} tentativi)")

# ----------------------------------------------------------------------
# VISUALIZZAZIONE (replay delle traiettorie registrate)
# ----------------------------------------------------------------------
data_vis = mujoco.MjData(model)
STRIDE = int(round(1.0 / (VIDEO_FPS * SIM_DT)))

def apply_camera(cam):
    cam.lookat[:] = [0.3, 0.0, 0.4]
    cam.distance = 2.2
    cam.azimuth = 140
    cam.elevation = -20

def add_markers(scn, rp):
    """Sfere puramente visive (non fanno parte del modello fisico)."""
    for pos, rgba in ((rp["pick"], [0.1, 0.8, 0.2, 0.6]), (rp["place"], [0.9, 0.2, 0.2, 0.6])):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.03, 0, 0]), pos, np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

def set_pose(q):
    data_vis.qpos[qpos_adr] = q
    mujoco.mj_forward(model, data_vis)

if SAVE_VIDEO:
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    apply_camera(cam)
    frames = []
    for rp in replays:
        for k in range(0, len(rp["q"]), STRIDE):
            set_pose(rp["q"][k])
            renderer.update_scene(data_vis, camera=cam)
            if SHOW_MARKERS:
                add_markers(renderer.scene, rp)
            frames.append(renderer.render())
    imageio.mimsave(VIDEO_PATH, frames, fps=VIDEO_FPS)
    print(f"Video salvato in {VIDEO_PATH}")

if SHOW_VIEWER:
    with mujoco.viewer.launch_passive(model, data_vis) as v:
        apply_camera(v.cam)
        for rp in replays:
            if SHOW_MARKERS:
                v.user_scn.ngeom = 0
                add_markers(v.user_scn, rp)
            for k in range(0, len(rp["q"]), STRIDE):
                if not v.is_running():
                    break
                set_pose(rp["q"][k])
                v.sync()
                time.sleep(1.0 / VIDEO_FPS)
            time.sleep(1.0)