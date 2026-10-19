import numpy as np
import pandas as pd
import mujoco

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------

MODEL_PATH = "../kuka_iiwa_14/scene.xml"  
TARGET_JOINT_NAME = "joint2"        
EXAMPLE_JOINT_NAME = "joint3"
SIM_DT = 0.001                      
LOG_DECIMATION = 1                  
TORQUE_LIMIT = 320.0
DURATION_PER_STEP = 0.7          # secondi per ogni livello di velocità
RAMP_TIME = 0.3                     # tempo per passaggio da una velocità all'altra

KP_VEL = 150
KI_VEL = 18

NUM_VELOCITIES = 10

FC = 8.0        # Costante attrito di coulomb
FS = 12.0       # Costante attrito statico
FV = 0.8        # Costante attrito viscoso
VS = 0.1        # velocità di stribeck

SIGMA_0 = 500   # Nm/rad
SIGMA_1 = 7    # Nm·s/rad 

NUM_TRAJ = 40
OUTPUT_CSV = "../Datasets/trainingB.csv"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
model.opt.timestep = SIM_DT

target_joint_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME)
target_qpos_adr    = model.jnt_qposadr[target_joint_id]
target_qvel_adr    = model.jnt_dofadr[target_joint_id]
target_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator2")
q_home = [0.0] * 7

# configurazione attuatore: una volta sola, e' su model non su data
model.actuator_gaintype[target_actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
model.actuator_gainprm[target_actuator_id, 0] = 1.0
model.actuator_biastype[target_actuator_id] = mujoco.mjtBias.mjBIAS_NONE
model.actuator_biasprm[target_actuator_id, :] = 0.0
model.actuator_ctrllimited[target_actuator_id] = 1
model.actuator_ctrlrange[target_actuator_id] = [-TORQUE_LIMIT, TORQUE_LIMIT]

lista_df = []

# ----------------------------------------------------------------------
# GENERAZIONE DEL PROFILO DI VELOCITA' DESIDERATA NEL TEMPO
# ----------------------------------------------------------------------

def reset_traj():
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_home
    for act_id in range(model.nu):
        if act_id != target_actuator_id:
            jid = model.actuator_trnid[act_id, 0]
            data.ctrl[act_id] = q_home[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)

def desired_velocity(t):
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
    return v_target


for i in range(NUM_TRAJ):

    reset_traj()
    integral_error = 0.0
    z = 0.0

    n_slows = int(max(1, NUM_VELOCITIES * 0.70))
    VELOCITY_LEVELS = np.random.uniform(-1.8, 1.8, size=10)
    VELOCITY_LEVELS[: n_slows] = np.random.uniform(-0.3, 0.3, size = n_slows)

    
    total_time = DURATION_PER_STEP * len(VELOCITY_LEVELS)
    n_steps = int(total_time / SIM_DT)
    n_dof = model.nv


    # ----------------------------------------------------------------------
    # LOOP DI SIMULAZIONE
    # ----------------------------------------------------------------------

    log_rows = []
    integral_error = 0.0
    z = 0.0
    tau_inertia = np.zeros(model.nv)   # buffer riusato, allocato una volta sola

    for step in range(n_steps):

        t = step * SIM_DT

        # ------------------------------------------------------------------
        # 1. CONTROLLO: calcolo della coppia di comando
        # ------------------------------------------------------------------
        v_des = desired_velocity(t)
        v_meas = data.qvel[target_qvel_adr]
        err = v_des - v_meas

        tau_pi = KP_VEL * err + KI_VEL * integral_error
        tau_cmd = tau_pi + data.qfrc_bias[target_qvel_adr]   # feedforward gravita'+Coriolis

        # anti wind-up: integro solo se non sono in saturazione,
        # oppure se l'errore spinge verso l'uscita dalla saturazione
        if abs(tau_cmd) < TORQUE_LIMIT or err * tau_cmd < 0:
            integral_error += err * SIM_DT

        # ------------------------------------------------------------------
        # 2. ATTRITO: modello LuGre valutato sulla velocita' corrente
        # ------------------------------------------------------------------
        qdot = v_meas

        g_qdot = FC + (FS - FC) * np.exp(-(qdot / VS) ** 2)
        g_qdot_norm = g_qdot / SIGMA_0
        r = np.abs(qdot) / g_qdot_norm

        z_new = (z + SIM_DT * qdot) / (1 + SIM_DT * r)
        zdot = qdot - r * z_new

        tau_viscoso = FV * qdot
        signal = SIGMA_0 * z_new + SIGMA_1 * zdot + tau_viscoso

        noise = np.random.normal(loc=0.0, scale=0.02 * np.abs(signal))
        tau_friction = signal + noise

        z = z_new

        data.ctrl[target_actuator_id] = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)
        data.qfrc_applied[target_qvel_adr] = -tau_friction   # l'attrito si oppone al moto


        mujoco.mj_forward(model, data)

        pos_k      = data.qpos[target_qpos_adr]
        vel_k      = data.qvel[target_qvel_adr]
        tau_act_k  = data.qfrc_actuator[target_qvel_adr]
        tau_bias_k = data.qfrc_bias[target_qvel_adr]
        tau_con_k  = data.qfrc_constraint[target_qvel_adr]

        mujoco.mj_mulM(model, data, tau_inertia, data.qacc)   # M(q) * qacc
        tau_in_k = tau_inertia[target_qvel_adr]               # indice di DoF, non joint id

        log_rows.append({
            "traj_id": i,
            "t": data.time,
            "position_rad": pos_k,
            "velocity_rad_s": vel_k,
            "torque_actuator_Nm": tau_act_k,
            "torque_bias_Nm": tau_bias_k,
            "inertia": tau_in_k,
            "constraint": tau_con_k,
            "torque_friction_true_Nm": tau_friction,   # ground truth, utile per validare
        })


        mujoco.mj_step(model, data)       

    df = pd.DataFrame(log_rows)

    df["torque_friction_estimated_Nm"] = df["inertia"] + df["torque_bias_Nm"] - df["torque_actuator_Nm"] - df["constraint"]
    df["traj_id"] = i
    if (df["constraint"] == 0).all():  
        lista_df.append(df)
        print("true")
    else:
        print("false")

df_total = pd.concat(lista_df, ignore_index=True)
df_total.to_csv(OUTPUT_CSV, index=False)
print(f"Dataset salvato in {OUTPUT_CSV} ({len(df_total)} campioni)")


