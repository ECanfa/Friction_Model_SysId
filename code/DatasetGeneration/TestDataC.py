import numpy as np
import pandas as pd
import mujoco

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# NB: i parametri di attrito, controllo e armature devono essere IDENTICI
# a quelli di TrainingDataA.py / TrainingDataB.py, altrimenti il test misura
# un cambio di modello e non la generalizzazione.
# ----------------------------------------------------------------------
SEED = 1                                   # diverso da quello del training
rng = np.random.default_rng(SEED)

MODEL_PATH = "../kuka_iiwa_14/scene.xml"
OUTPUT_CSV = "../Datasets/testC.csv"
SIM_DT = 0.001
N_JOINTS = 7

TORQUE_LIMIT = np.array([320.0, 320.0, 180.0, 180.0, 110.0, 40.0, 40.0])
SCALE = TORQUE_LIMIT / 320.0

# Controllo di velocita' (PI), scalato per giunto
KP_VEL = 150.0 * SCALE
KI_VEL = 18.0 * SCALE

# Attrito LuGre: valori a carico nullo, per giunto
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

ARMATURE = np.array([0.5, 0.5, 0.3, 0.3, 0.2, 0.1, 0.1])   # kg m^2

# Profilo di test: gradini a due livelli (+v / -v), ampiezza diversa per ogni giunto
NUM_SLOW_TRAJ = 10                         # traiettorie lente (presliding / Stribeck)
NUM_FAST_TRAJ = 8                          # traiettorie veloci
SLOW_RANGE = (0.0, 0.1)                    # rad/s
FAST_RANGE = (0.6, 1.2)                    # rad/s: tenuto DENTRO il range visto in training (max 1.2)
NUM_VELOCITIES = 5                         # livelli per traiettoria
DURATION_PER_STEP = 0.7                    # s per livello
RAMP_TIME = 0.3                            # s per passare da un livello al successivo
JOINT_MARGIN = 0.15                        # rad di margine dai limiti di giunto
CONSTRAINT_TOL = 1e-9

NUM_TRAJ = NUM_SLOW_TRAJ + NUM_FAST_TRAJ
MAX_ATTEMPTS = 3 * NUM_TRAJ

STEPS_PER_LEVEL = int(round(DURATION_PER_STEP / SIM_DT))
RAMP_STEPS = int(round(RAMP_TIME / SIM_DT))
n_steps = NUM_VELOCITIES * STEPS_PER_LEVEL

# ----------------------------------------------------------------------
# MODELLO E ATTUATORI (controllo in coppia invece che in posizione)
# ----------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
model.opt.timestep = SIM_DT

joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, N_JOINTS + 1)]
actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                         for i in range(1, N_JOINTS + 1)])

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

q_home = np.zeros(model.nv)

model.dof_armature[qvel_adr] = ARMATURE
mujoco.mj_setConst(model, data)

def reset_traj():
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_home
    mujoco.mj_forward(model, data)

# ----------------------------------------------------------------------
# CONTROLLO DI STABILITA' NUMERICA (forze applicate in modo esplicito)
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
# PROFILO DI VELOCITA' DESIDERATA (stateless, indicizzato sullo step)
# ----------------------------------------------------------------------
def desired_velocity(step, vel, sign0):
    """Gradini che alternano +vel / -vel con rampe lineari.
    vel: [N_JOINTS] ampiezza, sign0: [N_JOINTS] segno del primo gradino (+1/-1).
    Il primo gradino parte da velocita' nulla."""
    idx = min(step // STEPS_PER_LEVEL, NUM_VELOCITIES - 1)
    k = step - idx * STEPS_PER_LEVEL

    v_target = sign0 * (-1) ** idx * vel
    v_prev = -v_target if idx > 0 else np.zeros(N_JOINTS)

    if k < RAMP_STEPS:
        alpha = k / RAMP_STEPS
        return v_prev + alpha * (v_target - v_prev)
    return v_target

def limit_near_joint_range(v_des, qpos):
    """Rallenta il comando vicino ai limiti di giunto per non attivare i vincoli."""
    q = qpos[qpos_adr]
    v = v_des.copy()
    near_hi = (v > 0) & (q > JOINT_HI - JOINT_MARGIN)
    near_lo = (v < 0) & (q < JOINT_LO + JOINT_MARGIN)
    v[near_hi] *= np.clip((JOINT_HI[near_hi] - q[near_hi]) / JOINT_MARGIN, 0.0, 1.0)
    v[near_lo] *= np.clip((q[near_lo] - JOINT_LO[near_lo]) / JOINT_MARGIN, 0.0, 1.0)
    return v

# ----------------------------------------------------------------------
# COLONNE DEL LOG (stesso formato degli script di training)
# ----------------------------------------------------------------------
cols = (["t"]
        + [f"position_{j}_rad" for j in range(N_JOINTS)]
        + [f"velocity_rad_{j}_s" for j in range(N_JOINTS)]
        + [f"velocity_des_rad_{j}_s" for j in range(N_JOINTS)]
        + [f"tau_load_{j}" for j in range(N_JOINTS)]
        + [f"tau_friction_estimated_{j}" for j in range(N_JOINTS)]
        + [f"torque_friction_true_{j}" for j in range(N_JOINTS)]
        + [f"constraint_{j}" for j in range(N_JOINTS)])
constraint_cols = [f"constraint_{j}" for j in range(N_JOINTS)]

# ----------------------------------------------------------------------
# LOOP SULLE TRAIETTORIE
# ----------------------------------------------------------------------
tau_inertia = np.zeros(model.nv)

accepted, attempts, discarded = 0, 0, 0

while accepted < NUM_TRAJ and attempts < MAX_ATTEMPTS:
    attempts += 1

    # le prime NUM_SLOW_TRAJ traiettorie accettate sono lente, le altre veloci
    lo, hi = SLOW_RANGE if accepted < NUM_SLOW_TRAJ else FAST_RANGE
    vel = rng.uniform(lo, hi, size=N_JOINTS)
    sign0 = rng.choice([-1.0, 1.0], size=N_JOINTS)

    # --- stato iniziale ---
    reset_traj()
    integral_error = np.zeros(N_JOINTS)
    z = np.zeros(N_JOINTS)
    tau_load_filt = np.zeros(N_JOINTS)
    log = np.empty((n_steps, len(cols)))

    for step in range(n_steps):

        # 1. CONTROLLO PI DI VELOCITA' (vettoriale)
        v_des = limit_near_joint_range(desired_velocity(step, vel, sign0), data.qpos)
        v_meas = data.qvel[qvel_adr].copy()
        err = v_des - v_meas

        tau_cmd = KP_VEL * err + KI_VEL * integral_error + data.qfrc_bias[qvel_adr]

        integrate = (np.abs(tau_cmd) < TORQUE_LIMIT) | (err * tau_cmd < 0)   # anti wind-up
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

        # 4. DINAMICA E LOG (qacc aggiornato, coerente con lo step corrente)
        mujoco.mj_forward(model, data)
        mujoco.mj_mulM(model, data, tau_inertia, data.qacc)

        tau_est = (tau_inertia[qvel_adr] + data.qfrc_bias[qvel_adr]
                   - data.qfrc_constraint[qvel_adr] - data.qfrc_actuator[qvel_adr])

        log[step] = np.concatenate([
            [data.time],
            data.qpos[qpos_adr],
            v_meas,
            v_des,
            tau_load_filt,
            tau_est,
            tau_friction,                       # ground truth (segno opposto a tau_est)
            data.qfrc_constraint[qvel_adr],
        ])

        mujoco.mj_step(model, data)

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
    accepted += 1
    tipo = "lenta" if accepted <= NUM_SLOW_TRAJ else "veloce"
    print(f"[tentativo {attempts}] ok ({accepted}/{NUM_TRAJ}, {tipo})")

if accepted == 0:
    raise RuntimeError("Tutte le traiettorie sono state scartate")
if accepted < NUM_TRAJ:
    print(f"ATTENZIONE: accettate solo {accepted}/{NUM_TRAJ} traiettorie in {attempts} tentativi")

print(f"Dataset salvato in {OUTPUT_CSV}: {accepted} traiettorie, "
      f"{accepted * n_steps} campioni ({discarded} scartate su {attempts} tentativi)")