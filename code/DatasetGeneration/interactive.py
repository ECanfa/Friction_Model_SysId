"""
Generatore interattivo del Training Dataset A (sinusoidi a segmenti) per il giunto 2
del KUKA LBR iiwa 14.

Uso:
    python TrainingDataA_interactive.py                 # interfaccia grafica (tkinter)
    python TrainingDataA_interactive.py --headless ...  # da riga di comando, stessi limiti

Limiti del giunto 2 applicati ai parametri scelti dall'utente:
  - Velocita':   |A| <= 85 deg/s = 1.483 rad/s       (datasheet KUKA, velocita' a carico nominale)
  - Posizione:   +-120 deg = +-2.094 rad             (letta dal modello MuJoCo, coincide col datasheet)
  - Accelerazione: il datasheet non la fornisce. Si usa il minore tra MAX_ACCEL e una stima
    (TORQUE_LIMIT - attrito massimo) / M22(q_home). Il picco di accelerazione di una sinusoide
    e' A*omega, quindi omega_max <= acc_limit / A_max.
"""
import argparse
import json
import os
import queue
import threading

import numpy as np
import pandas as pd
import mujoco

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:          # tkinter assente: resta disponibile la modalita' --headless
    tk = None

# ----------------------------------------------------------------------
# CONFIGURAZIONE (invariata rispetto allo script originale)
# ----------------------------------------------------------------------
MODEL_PATH = "../kuka_iiwa_14/scene.xml"
TARGET_JOINT_NAME = "joint2"
TARGET_ACTUATOR_NAME = "actuator2"
SIM_DT = 0.001
TORQUE_LIMIT = 320.0
RAMP_TIME = 0.1                 # s, raccordo smootherstep tra segmenti
KP_VEL, KI_VEL = 150, 18
FC, FS, FV, VS = 8.0, 12.0, 0.8, 0.1
SIGMA_0, SIGMA_1 = 500, 7
SEGMENTS_PER_TRAJ = 5
SLOW_AMPLITUDE = 0.1            # rad/s, ampiezza massima dei segmenti "lenti"
JOINT_MARGIN = 0.15             # rad, margine dal fondo corsa
MAX_ACCEL = 10.0                # rad/s^2, valore dello script originale

# ----------------------------------------------------------------------
# LIMITI DEL GIUNTO 2 E DELL'INTERFACCIA
# ----------------------------------------------------------------------
V_LIMIT_DEG = 85.0                       # datasheet LBR iiwa 14 R820, asse A2
V_LIMIT = float(np.deg2rad(V_LIMIT_DEG))  # 1.483 rad/s
Q_LIMIT_DATASHEET = float(np.deg2rad(120.0))
OMEGA_FLOOR = 0.5               # rad/s: sotto, la durata del segmento (1.5*2pi/omega_min) esplode
AMPLITUDE_FLOOR = 0.05          # rad/s
DEFAULT_OUTPUT = "../Datasets/trainingA_custom.csv"   # nome diverso: non sovrascrive trainingA.csv


# ----------------------------------------------------------------------
# SIMULATORE
# ----------------------------------------------------------------------
class Sim:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        m.opt.timestep = SIM_DT

        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME)
        self.qpos_adr = int(m.jnt_qposadr[jid])
        self.qvel_adr = int(m.jnt_dofadr[jid])
        self.act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, TARGET_ACTUATOR_NAME)
        self.q_lo, self.q_hi = (float(x) for x in m.jnt_range[jid])
        self.range_matches_datasheet = abs(self.q_hi - Q_LIMIT_DATASHEET) < 1e-3

        # attuatore: controllo in coppia diretta
        a = self.act_id
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a, 0] = 1.0
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[a, :] = 0.0
        m.actuator_ctrllimited[a] = 1
        m.actuator_ctrlrange[a] = [-TORQUE_LIMIT, TORQUE_LIMIT]

        # limite di accelerazione: minimo tra MAX_ACCEL e stima da coppia a q_home
        self.reset()
        self.inertia_home = self._joint_inertia()
        friction_peak = FS + FV * V_LIMIT
        self.acc_from_torque = (TORQUE_LIMIT - friction_peak) / self.inertia_home
        self.acc_limit = float(min(MAX_ACCEL, self.acc_from_torque))

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0.0
        for act_id in range(m.nu):
            if act_id != self.act_id:
                d.ctrl[act_id] = 0.0           # q_home = 0 per tutti i giunti
        mujoco.mj_forward(m, d)

    def _joint_inertia(self):
        e = np.zeros(self.model.nv)
        e[self.qvel_adr] = 1.0
        out = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, out, e)
        return float(out[self.qvel_adr])

    def limit_near_joint_range(self, v_des, qpos):
        lo, hi = self.q_lo, self.q_hi
        if v_des > 0 and qpos > hi - JOINT_MARGIN:
            return v_des * max(0.0, (hi - qpos) / JOINT_MARGIN)
        if v_des < 0 and qpos < lo + JOINT_MARGIN:
            return v_des * max(0.0, (qpos - lo) / JOINT_MARGIN)
        return v_des


# ----------------------------------------------------------------------
# PARAMETRI E PROFILO DI RIFERIMENTO
# ----------------------------------------------------------------------
def apply_limits(a_max, w_min, w_max, acc_limit):
    """Riporta i parametri scelti dentro i limiti del giunto 2."""
    a = float(np.clip(a_max, AMPLITUDE_FLOOR, V_LIMIT))
    w_cap = acc_limit / a                      # A*omega <= acc_limit
    wmax = float(np.clip(w_max, OMEGA_FLOOR, w_cap))
    wmin = float(np.clip(w_min, OMEGA_FLOOR, wmax))
    return a, wmin, wmax


def segment_duration(w_min):
    return 1.5 * (2 * np.pi / w_min)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (x * (x * 6 - 15) + 10)


def sample_segments(rng, a_max, w_min, w_max, acc_limit):
    n = SEGMENTS_PER_TRAJ
    omegas = rng.uniform(w_min, w_max, size=n)
    cap = np.minimum(a_max, acc_limit / omegas)
    amps = rng.uniform(-1, 1, size=n) * cap

    n_slow = max(1, n // 2)                     # segmenti a bassa ampiezza
    amps[:n_slow] = rng.uniform(-1, 1, size=n_slow) * min(SLOW_AMPLITUDE, a_max)
    rng.shuffle(amps)

    # il tetto va ricalcolato DOPO il mescolamento: ogni ampiezza ha cambiato omega associata
    cap = np.minimum(a_max, acc_limit / omegas)
    return np.clip(amps, -cap, cap), omegas


def reference_velocity(t, amps, omegas, T_seg):
    idx = min(int(t // T_seg), len(amps) - 1)
    t_in = t % T_seg
    v_curr = amps[idx] * np.sin(omegas[idx] * t_in)
    if idx > 0 and t_in < RAMP_TIME:
        v_prev = amps[idx - 1] * np.sin(omegas[idx - 1] * T_seg)
        alpha = smootherstep(t_in / RAMP_TIME)
        return (1 - alpha) * v_prev + alpha * v_curr
    return v_curr


# ----------------------------------------------------------------------
# SIMULAZIONE DI UNA TRAIETTORIA
# ----------------------------------------------------------------------
def simulate_trajectory(sim, a_max, w_min, w_max, rng, traj_id, cancel=None, frac_cb=None):
    m, d = sim.model, sim.data
    qv, qp, act = sim.qvel_adr, sim.qpos_adr, sim.act_id

    amps, omegas = sample_segments(rng, a_max, w_min, w_max, sim.acc_limit)
    T_seg = segment_duration(w_min)
    n_steps = int(T_seg * SEGMENTS_PER_TRAJ / SIM_DT)

    sim.reset()
    integral_error, z = 0.0, 0.0
    tau_inertia = np.zeros(m.nv)
    rows = []

    for step in range(n_steps):
        if step % 1000 == 0:
            if cancel is not None and cancel.is_set():
                return None
            if frac_cb is not None:
                frac_cb(step / n_steps)

        t = step * SIM_DT
        v_raw = reference_velocity(t, amps, omegas, T_seg)
        v_des = sim.limit_near_joint_range(v_raw, d.qpos[qp])

        v_meas = d.qvel[qv]
        err = v_des - v_meas

        tau_cmd = KP_VEL * err + KI_VEL * integral_error + d.qfrc_bias[qv]
        if abs(tau_cmd) < TORQUE_LIMIT or err * tau_cmd < 0:      # anti wind-up
            integral_error += err * SIM_DT

        # attrito LuGre
        qdot = v_meas
        g = FC + (FS - FC) * np.exp(-(qdot / VS) ** 2)
        r = np.abs(qdot) / (g / SIGMA_0)
        z_new = (z + SIM_DT * qdot) / (1 + SIM_DT * r)
        zdot = qdot - r * z_new
        signal = SIGMA_0 * z_new + SIGMA_1 * zdot + FV * qdot
        tau_friction = signal + rng.normal(0.0, 0.02 * np.abs(signal))
        z = z_new

        d.ctrl[act] = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)
        d.qfrc_applied[qv] = -tau_friction
        mujoco.mj_forward(m, d)
        mujoco.mj_mulM(m, d, tau_inertia, d.qacc)

        rows.append({
            "traj_id": traj_id,
            "t": d.time,
            "velocity_des_rad_s": v_des,
            "position_rad": d.qpos[qp],
            "velocity_rad_s": v_meas,
            "torque_actuator_Nm": d.qfrc_actuator[qv],
            "torque_bias_Nm": d.qfrc_bias[qv],
            "inertia": tau_inertia[qv],
            "constraint": d.qfrc_constraint[qv],
            "torque_friction_true_Nm": tau_friction,
        })
        mujoco.mj_step(m, d)

    df = pd.DataFrame(rows)
    df["torque_friction_estimated_Nm"] = (
        df["inertia"] + df["torque_bias_Nm"] - df["torque_actuator_Nm"] - df["constraint"]
    )
    return df


def generate_dataset(sim, a_max, w_min, w_max, n_traj, seed, regenerate, output,
                     log=print, progress=None, cancel=None):
    """Genera il dataset, lo salva in CSV (+ file .json con i parametri) e restituisce un riepilogo."""
    a_max, w_min, w_max = apply_limits(a_max, w_min, w_max, sim.acc_limit)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 32))
    rng = np.random.default_rng(seed)

    dfs, discarded, attempts = [], 0, 0
    max_attempts = 3 * n_traj if regenerate else n_traj
    log(f"Parametri: A_max={a_max:.3f} rad/s, omega in [{w_min:.3f}, {w_max:.3f}] rad/s, "
        f"{n_traj} traiettorie, seed={seed}")

    while attempts < max_attempts and not (regenerate and len(dfs) >= n_traj):
        if cancel is not None and cancel.is_set():
            break
        ok = len(dfs)
        frac_cb = (lambda f, ok=ok: progress((ok + f) / n_traj)) if progress else None
        df = simulate_trajectory(sim, a_max, w_min, w_max, rng, ok, cancel, frac_cb)
        if df is None:                                   # annullata
            break
        attempts += 1
        if (df["constraint"] == 0).all():
            dfs.append(df)
            log(f"traiettoria {len(dfs)}/{n_traj}: ok")
        else:
            discarded += 1
            log(f"tentativo {attempts}: scartata (vincoli attivi)")
        if progress:
            progress(len(dfs) / n_traj if regenerate else attempts / n_traj)

    cancelled = cancel is not None and cancel.is_set()
    summary = {"accepted": len(dfs), "discarded": discarded, "attempts": attempts,
               "seed": seed, "cancelled": cancelled, "output": None}
    if not dfs:
        log("Nessuna traiettoria valida: nessun file scritto.")
        return summary

    summary["output"] = output
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    df_total = pd.concat(dfs, ignore_index=True)
    df_total.to_csv(output, index=False)
    with open(os.path.splitext(output)[0] + ".params.json", "w") as f:
        json.dump({
            "a_max_rad_s": a_max, "omega_min_rad_s": w_min, "omega_max_rad_s": w_max,
            "segments_per_traj": SEGMENTS_PER_TRAJ, "segment_duration_s": segment_duration(w_min),
            "n_traj_requested": n_traj, "regenerate_discarded": regenerate, **summary,
            "limits": {"velocity_rad_s": V_LIMIT, "position_rad": sim.q_hi,
                       "acc_limit_rad_s2": sim.acc_limit,
                       "acc_from_torque_rad_s2": sim.acc_from_torque,
                       "inertia_home_kg_m2": sim.inertia_home},
        }, f, indent=2)
    log(f"Dataset salvato in {output} ({len(df_total)} campioni, {len(dfs)} traiettorie, "
        f"{discarded} scartate)")
    return summary


# ----------------------------------------------------------------------
# INTERFACCIA GRAFICA
# ----------------------------------------------------------------------
class App:
    def __init__(self, root, sim):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.root, self.sim = root, sim
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.preview_seed = 0
        self._busy = False

        root.title("Generatore Training Dataset A - KUKA LBR iiwa 14, giunto 2")
        left = ttk.Frame(root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        acc = sim.acc_limit
        self.var_a = tk.DoubleVar(value=1.2)
        self.var_wmin = tk.DoubleVar(value=1.0)
        self.var_wmax = tk.DoubleVar(value=4.0)
        self.var_n = tk.IntVar(value=40)
        self.var_seed = tk.StringVar(value="")
        self.var_regen = tk.BooleanVar(value=True)
        self.var_out = tk.StringVar(value=DEFAULT_OUTPUT)

        self.sc_a = self._scale(left, 0, "Ampiezza massima A [rad/s]", self.var_a,
                                AMPLITUDE_FLOOR, V_LIMIT)
        self.sc_wmin = self._scale(left, 1, "Pulsazione minima ω [rad/s]", self.var_wmin,
                                   OMEGA_FLOOR, acc / 1.2)
        self.sc_wmax = self._scale(left, 2, "Pulsazione massima ω [rad/s]", self.var_wmax,
                                   OMEGA_FLOOR, acc / 1.2)

        ttk.Label(left, text="Numero di traiettorie").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(left, from_=1, to=500, textvariable=self.var_n, width=8).grid(
            row=4, column=0, sticky="w")
        ttk.Label(left, text="Seed (vuoto = casuale)").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(left, textvariable=self.var_seed, width=12).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(left, text="Rigenera le traiettorie scartate",
                        variable=self.var_regen).grid(row=7, column=0, sticky="w", pady=(8, 0))

        ttk.Label(left, text="File di output").grid(row=8, column=0, sticky="w", pady=(8, 0))
        out_row = ttk.Frame(left)
        out_row.grid(row=9, column=0, sticky="we")
        ttk.Entry(out_row, textvariable=self.var_out, width=30).pack(side="left")
        ttk.Button(out_row, text="...", width=3, command=self._browse).pack(side="left", padx=2)

        btns = ttk.Frame(left)
        btns.grid(row=10, column=0, sticky="we", pady=10)
        self.btn_gen = ttk.Button(btns, text="Genera", command=self._start)
        self.btn_gen.pack(side="left")
        self.btn_cancel = ttk.Button(btns, text="Annulla", command=self.cancel.set, state="disabled")
        self.btn_cancel.pack(side="left", padx=4)
        ttk.Button(btns, text="Nuovo campione", command=self._new_preview).pack(side="left")

        self.info = ttk.Label(left, justify="left", wraplength=340)
        self.info.grid(row=11, column=0, sticky="w")

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_v = self.fig.add_subplot(211)
        self.ax_q = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        bottom = ttk.Frame(root, padding=8)
        bottom.grid(row=1, column=0, columnspan=2, sticky="we")
        self.bar = ttk.Progressbar(bottom, maximum=1.0)
        self.bar.pack(fill="x")
        self.log_box = tk.Text(bottom, height=7, state="disabled")
        self.log_box.pack(fill="x", pady=(6, 0))

        self._log(f"Modello caricato. Inerzia M22 a q_home: {sim.inertia_home:.3f} kg m².")
        if not sim.range_matches_datasheet:
            self._log(f"ATTENZIONE: il range del modello (±{sim.q_hi:.4f} rad) non coincide "
                      f"con il datasheet (±{Q_LIMIT_DATASHEET:.4f} rad).")
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._on_change()
        self._poll()

    # --- widget helper ---------------------------------------------------
    def _scale(self, parent, row, text, var, lo, hi):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="we", pady=2)
        ttk.Label(f, text=text).pack(anchor="w")
        sc = tk.Scale(f, from_=lo, to=hi, resolution=0.01, orient="horizontal", length=320,
                      variable=var, command=self._on_change)
        sc.pack()
        return sc

    def _browse(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                         initialfile=os.path.basename(self.var_out.get()))
        if p:
            self.var_out.set(p)

    # --- limiti e anteprima ----------------------------------------------
    def _clamped(self):
        return apply_limits(self.var_a.get(), self.var_wmin.get(), self.var_wmax.get(),
                            self.sim.acc_limit)

    def _on_change(self, _=None):
        a, wmin, wmax = self._clamped()
        cap = self.sim.acc_limit / a
        self.sc_wmin.configure(to=cap)
        self.sc_wmax.configure(to=cap)
        if abs(self.var_a.get() - a) > 1e-9: self.var_a.set(a)
        if abs(self.var_wmin.get() - wmin) > 1e-9: self.var_wmin.set(wmin)
        if abs(self.var_wmax.get() - wmax) > 1e-9: self.var_wmax.set(wmax)
        self._update_info(a, wmin, wmax)
        self._draw_preview(a, wmin, wmax)

    def _update_info(self, a, wmin, wmax):
        s = self.sim
        T = segment_duration(wmin)
        total = T * SEGMENTS_PER_TRAJ
        exc = 2 * a / wmin
        avail = s.q_hi - JOINT_MARGIN
        lines = [
            f"Limite velocità (datasheet, {V_LIMIT_DEG:.0f}°/s): {V_LIMIT:.3f} rad/s",
            f"Limite posizione: ±{s.q_hi:.3f} rad (±120°)",
            f"Limite accelerazione usato: {s.acc_limit:.1f} rad/s²  "
            f"(stima da coppia a q_home: {s.acc_from_torque:.0f})",
            f"ω massima con A={a:.2f}: {s.acc_limit / a:.2f} rad/s",
            f"Picco accelerazione A·ω_max: {a * wmax:.2f} rad/s²",
            f"Segmento: {T:.2f} s | traiettoria: {total:.1f} s | {int(total / SIM_DT)} campioni",
            f"Escursione massima di un segmento 2A/ω_min: {exc:.2f} rad",
        ]
        if exc > avail:
            lines.append(f"! Supera il margine da q=0 ({avail:.2f} rad): interverrà il limitatore "
                         "di escursione e alcune traiettorie potrebbero essere scartate.")
        self.info.configure(text="\n".join(lines))

    def _new_preview(self):
        self.preview_seed += 1
        self._on_change()

    def _draw_preview(self, a, wmin, wmax):
        rng = np.random.default_rng(self.preview_seed)
        amps, omegas = sample_segments(rng, a, wmin, wmax, self.sim.acc_limit)
        T = segment_duration(wmin)
        dt = 0.01
        t = np.arange(0.0, T * SEGMENTS_PER_TRAJ, dt)
        v = np.array([reference_velocity(x, amps, omegas, T) for x in t])
        q = np.cumsum(v) * dt
        qh = self.sim.q_hi

        self.ax_v.clear()
        self.ax_q.clear()
        self.ax_v.plot(t, v, color="#1a1a1a", lw=1)
        self.ax_v.axhline(V_LIMIT, color="r", ls="--", lw=1)
        self.ax_v.axhline(-V_LIMIT, color="r", ls="--", lw=1)
        self.ax_v.set_ylabel("v rif. [rad/s]")
        self.ax_v.set_title("Anteprima del riferimento (senza limitatore di escursione)", fontsize=10)
        self.ax_q.plot(t, q, color="#1d3557", lw=1)
        for lim in (qh, -qh):
            self.ax_q.axhline(lim, color="r", ls="--", lw=1)
        for lim in (qh - JOINT_MARGIN, -(qh - JOINT_MARGIN)):
            self.ax_q.axhline(lim, color="orange", ls=":", lw=1)
        self.ax_q.set_ylim(-1.2 * qh, 1.2 * qh)
        self.ax_q.set_ylabel("q [rad]")
        self.ax_q.set_xlabel("t [s]")
        for ax in (self.ax_v, self.ax_q):
            ax.grid(True, ls=":", alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # --- esecuzione ------------------------------------------------------
    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _start(self):
        if self._busy:
            return
        seed_txt = self.var_seed.get().strip()
        try:
            seed = int(seed_txt) if seed_txt else None
            n = int(self.var_n.get())
            if n < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Parametri non validi", "Numero di traiettorie e seed devono essere interi.")
            return
        a, wmin, wmax = self._clamped()
        out = self.var_out.get().strip()
        if not out:
            messagebox.showerror("Parametri non validi", "Indicare il file di output.")
            return
        if os.path.exists(out) and not messagebox.askyesno("File esistente", f"{out} esiste: sovrascrivere?"):
            return

        self.cancel.clear()
        self._busy = True
        self.btn_gen.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.bar["value"] = 0
        self.worker = threading.Thread(
            target=self._work, args=(a, wmin, wmax, n, seed, self.var_regen.get(), out), daemon=True)
        self.worker.start()

    def _work(self, a, wmin, wmax, n, seed, regen, out):
        try:
            res = generate_dataset(
                self.sim, a, wmin, wmax, n, seed, regen, out,
                log=lambda m: self.q.put(("log", m)),
                progress=lambda f: self.q.put(("progress", f)),
                cancel=self.cancel)
            self.q.put(("done", res))
        except Exception as e:                       # noqa: BLE001
            self.q.put(("error", repr(e)))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "progress":
                    self.bar["value"] = min(1.0, payload)
                elif kind in ("done", "error"):
                    self._busy = False
                    self.btn_gen.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    if kind == "error":
                        self._log(f"ERRORE: {payload}")
                    elif payload["cancelled"]:
                        self._log("Generazione annullata.")
                    else:
                        self.bar["value"] = 1.0
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _close(self):
        self.cancel.set()
        self.root.destroy()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="senza interfaccia grafica")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--a-max", type=float, default=1.2)
    ap.add_argument("--w-min", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--no-regenerate", action="store_true")
    args = ap.parse_args()

    sim = Sim(args.model)
    if args.headless:
        a, wmin, wmax = apply_limits(args.a_max, args.w_min, args.w_max, sim.acc_limit)
        if (a, wmin, wmax) != (args.a_max, args.w_min, args.w_max):
            print(f"Parametri riportati nei limiti: A_max={a:.3f}, omega in [{wmin:.3f}, {wmax:.3f}]")
        generate_dataset(sim, a, wmin, wmax, args.n, args.seed, not args.no_regenerate, args.out)
        return

    if tk is None:
        raise SystemExit("tkinter non disponibile: installa python3-tk oppure usa --headless.")
    root = tk.Tk()
    App(root, sim)
    root.mainloop()


if __name__ == "__main__":
    main()