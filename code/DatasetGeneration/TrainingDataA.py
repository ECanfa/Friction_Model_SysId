import numpy as np
import pandas as pd
import mujoco
import random



# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------

MODEL_PATH = "../kuka_iiwa_14/scene.xml"
TARGET_JOINT_NAME = "joint2"
SIM_DT = 0.001
TORQUE_LIMIT = 320.0
RAMP_TIME = 0.1         # secondi per ogni livello di velocità
NUM_VELOCITIES = 7
KP_VEL = 150
KI_VEL = 18

FC, FS, FV, VS = 8.0, 12.0, 0.8, 0.1
SIGMA_0, SIGMA_1 = 500, 7

OUTPUT_CSV = "../Datasets/trainingA.csv"
NUM_TRAJ = 40
SEGMENTS_PER_TRAJ = 5

OMEGA_MIN, OMEGA_MAX = 1.0, 4.0
AMPLITUDE_MIN, AMPLITUDE_MAX = -1.2, 1.2

MAX_ACCEL = 10   # rad/s^2

# tempo per coprire il periodo di tutti i seni
DURATION_PER_STEP = 1.5 * (2 * np.pi / OMEGA_MIN)

JOINT_MARGIN = 0.15   # rad di margine dal limite di giunto, per evitare che si attivino i vincoli

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
model.opt.timestep = SIM_DT

target_joint_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME)
target_qpos_adr    = model.jnt_qposadr[target_joint_id]
target_qvel_adr    = model.jnt_dofadr[target_joint_id]
target_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator2")
joint_range = model.jnt_range[target_joint_id]
q_home = [0.0] * 7

# configurazione attuatore-> controllo in coppia invece che in posizione
model.actuator_gaintype[target_actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
model.actuator_gainprm[target_actuator_id, 0] = 1.0
model.actuator_biastype[target_actuator_id] = mujoco.mjtBias.mjBIAS_NONE
model.actuator_biasprm[target_actuator_id, :] = 0.0
model.actuator_ctrllimited[target_actuator_id] = 1
model.actuator_ctrlrange[target_actuator_id] = [-TORQUE_LIMIT, TORQUE_LIMIT]




def reset_traj():
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_home
    for act_id in range(model.nu):
        if act_id != target_actuator_id:
            jid = model.actuator_trnid[act_id, 0]
            data.ctrl[act_id] = q_home[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)

def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (x * (x * 6 - 15) + 10)


def desired_velocity(t, amplitudes, omegas):
    """Velocita' desiderata al tempo t, con transizione morbida (C1) tra segmenti."""
    idx = min(int(t // DURATION_PER_STEP), len(amplitudes) - 1)
    t_in_step = t % DURATION_PER_STEP

    v_curr = amplitudes[idx] * np.sin(omegas[idx] * t_in_step)
    if idx > 0 and t_in_step < RAMP_TIME:
        v_prev = amplitudes[idx - 1] * np.sin(omegas[idx - 1] * DURATION_PER_STEP)
        alpha = smootherstep(t_in_step / RAMP_TIME)
        return (1 - alpha) * v_prev + alpha * v_curr
    return v_curr

'''def desired_velocity_step(t):
    """Restituisce la velocita' desiderata del giunto target al tempo t,
    con rampe lineari tra un livello costante e il successivo."""
    step_duration = DURATION_PER_STEP
    idx = int(t // step_duration)
    idx = min(idx, len(VELOCITY_LEVELS) - 1)
    t_in_step = t % step_duration

    v_target = VELOCITY_LEVELS[idx]
    v_prev = VELOCITY_LEVELS[idx - 1] if idx > 0 else 0.0

    if t_in_step < RAMP_TIME:
        alpha = t_in_step / RAMP_TIME
        return v_prev + alpha * (v_target - v_prev)
    return v_target'''

def limit_near_joint_range(v_des, qpos):
    """Rallenta il comando quando ci si avvicina al limite fisico del giunto,
    per evitare di attivare i vincoli di collisione/limite."""
    lo, hi = joint_range
    if v_des > 0 and qpos > hi - JOINT_MARGIN:
        return v_des * max(0.0, (hi - qpos) / JOINT_MARGIN)
    if v_des < 0 and qpos < lo + JOINT_MARGIN:
        return v_des * max(0.0, (qpos - lo) / JOINT_MARGIN)
    return v_des


# ----------------------------------------------------------------------
# LOOP SULLE TRAIETTORIE
# ----------------------------------------------------------------------

lista_df = []

for i in range(NUM_TRAJ):

    reset_traj()
    integral_error = 0.0
    z = 0.0
    
    OMEGAS = np.random.uniform(OMEGA_MIN, OMEGA_MAX, size=SEGMENTS_PER_TRAJ)
    max_amp = np.minimum(AMPLITUDE_MAX, MAX_ACCEL / OMEGAS)
    AMPLITUDES = np.random.uniform(-1, 1, size=SEGMENTS_PER_TRAJ) * max_amp

    n_slow = max(1, SEGMENTS_PER_TRAJ // 2)   
    AMPLITUDES[:n_slow] = np.random.uniform(-0.1, 0.1, size=n_slow)

    random.shuffle(AMPLITUDES)

    n_steps = int(DURATION_PER_STEP * SEGMENTS_PER_TRAJ / SIM_DT)


   

    tau_inertia = np.zeros(model.nv)
    log_rows = []

    for step in range(n_steps):
        t = step * SIM_DT

     
        v_des_raw = desired_velocity(t, AMPLITUDES, OMEGAS)
    
        v_des = limit_near_joint_range(v_des_raw, data.qpos[target_qpos_adr])
            
        v_meas = data.qvel[target_qvel_adr]
        err = v_des - v_meas

        tau_cmd = KP_VEL * err + KI_VEL * integral_error + data.qfrc_bias[target_qvel_adr]
        if abs(tau_cmd) < TORQUE_LIMIT or err * tau_cmd < 0:   # anti wind-up
            integral_error += err * SIM_DT

       
        qdot = v_meas
        
        g = FC + (FS - FC) * np.exp(-(qdot / VS) ** 2)
        r = np.abs(qdot) / (g / SIGMA_0)
        z_new = (z + SIM_DT * qdot) / (1 + SIM_DT * r)
        zdot = qdot - r * z_new
        signal = SIGMA_0 * z_new + SIGMA_1 * zdot + FV * qdot
        tau_friction = signal + np.random.normal(0.0, 0.02 * np.abs(signal))
        z = z_new

        
        data.ctrl[target_actuator_id] = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)
        data.qfrc_applied[target_qvel_adr] = -tau_friction
        mujoco.mj_forward(model, data)
        mujoco.mj_mulM(model, data, tau_inertia, data.qacc)

        log_rows.append({
            "traj_id": i,
            "t": data.time,
            "velocity_des_rad_s": v_des,
            "position_rad": data.qpos[target_qpos_adr],
            "velocity_rad_s": v_meas,
            "torque_actuator_Nm": data.qfrc_actuator[target_qvel_adr],
            "torque_bias_Nm": data.qfrc_bias[target_qvel_adr],
            "inertia": tau_inertia[target_qvel_adr],
            "constraint": data.qfrc_constraint[target_qvel_adr],
            "torque_friction_true_Nm": tau_friction,
        })

        mujoco.mj_step(model, data)

    df = pd.DataFrame(log_rows)
    df["torque_friction_estimated_Nm"] = (
        df["inertia"] + df["torque_bias_Nm"] - df["torque_actuator_Nm"] - df["constraint"]
    )

    if (df["constraint"] == 0).all():
        lista_df.append(df)
        print(f"traj {i}: ok")
    else:
        print(f"traj {i}: scartata, vincoli attivi")

df_total = pd.concat(lista_df, ignore_index=True)
df_total.to_csv(OUTPUT_CSV, index=False)
print(f"Dataset salvato in {OUTPUT_CSV} ({len(df_total)} campioni, {df_total['traj_id'].nunique()} traiettorie)")