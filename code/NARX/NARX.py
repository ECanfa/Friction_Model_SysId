#System Identification ARX
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
import numpy as np
import pandas as pd
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.parameter_estimation import LeastSquares
import matplotlib.pyplot as plt
from sysidentpy.utils import build_lagged_matrix
from sysidentpy.narmax_base import RegressorDictionary
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
from sklearn.metrics import root_mean_squared_error
import math
import pandas

na = 2
nb = 1

SIM_DT = 0.001
DURATION_PER_STEP = 0.4
INPUT_A_CSV = "../Datasets/trainingA.csv"
INPUT_B_CSV = "../Datasets/trainingB.csv"
TEST_CSV = "../Datasets/testC.csv"
NUM_VELOCITIES = 10

dataA = pandas.read_csv(INPUT_A_CSV, sep=',')
dataB = pandas.read_csv(INPUT_B_CSV, sep=',')

# --- Costruiamo LISTE di traiettorie a lunghezza variabile (niente np.array rettangolare) ---
x_train_data_A = []
y_train_data_A = []
for traj_id, traj_df in dataA.groupby("traj_id", sort=True):
    x_train_data_A.append(traj_df["velocity_rad_s"].to_numpy(dtype=np.float32))
    y_train_data_A.append(traj_df["torque_friction_estimated_Nm"].to_numpy(dtype=np.float32))

x_train_data_B = []
y_train_data_B = []
for traj_id, traj_df in dataB.groupby("traj_id", sort=True):
    x_train_data_B.append(traj_df["velocity_rad_s"].to_numpy(dtype=np.float32))
    y_train_data_B.append(traj_df["torque_friction_estimated_Nm"].to_numpy(dtype=np.float32))

num_traj_A = len(y_train_data_A)
num_traj_B = len(y_train_data_B)
num_traj = num_traj_A + num_traj_B

x_train_list = x_train_data_A + x_train_data_B
y_train_list = y_train_data_A + y_train_data_B

max_lag = max(na, nb)  
basis_function = Polynomial(degree=3)

phi = []
y_targets = []

#training
for x_traj, y_traj in zip(x_train_list, y_train_list):
    x_traj = x_traj.reshape(-1, 1)
    y_traj = y_traj.reshape(-1, 1)

    regressor_matrix = build_lagged_matrix(
        x=x_traj, y=y_traj, xlag=nb, ylag=na, model_type="NARMAX"
    )

    psi = basis_function.fit(
        regressor_matrix, max_lag=max_lag, xlag=nb, ylag=na
    )

    phi.append(psi)
    y_targets.append(y_traj[max_lag:])   


PHI = np.vstack(phi)
Y = np.vstack(y_targets)

theta = LeastSquares().optimize(PHI, Y)

reg_dict = RegressorDictionary(
    xlag=nb,
    ylag=na,
    basis_function=basis_function, 
    model_type="NARMAX",
)
regressor_code = reg_dict.regressor_space(n_inputs=1) 

#testing
def NARX(theta, reg_code, x, y, k):
    model = 0
    for i in range(len(reg_code)):
        prod = 1
        for reg in reg_code[i]:         
            lag = reg % 1000
            if 1000 < reg < 2000:
                prod = prod * y[k-lag]
                if not math.isfinite(prod):
                    print(prod)
                    print(y)
                    print(k)
                    print(lag)
                if math.isnan(prod):
                    print(k)
                    print(lag)
                    print(y)
                    exit(1)    
            elif reg > 2000:
                prod = prod * x[k-lag]
       
        model += prod * theta[i][0]

    return model
    

data = pandas.read_csv(TEST_CSV, sep = ',')

friction = data["torque_friction_estimated_Nm"]
velocities = data["velocity_rad_s"]

x_test_data = []
y_test_data = []

for traj_id,traj_df in data.groupby("traj_id", sort=True):
    x_test_data.append(traj_df["velocity_rad_s"].to_numpy())
    y_test_data.append(traj_df["torque_friction_estimated_Nm"].to_numpy())

num_traj = len(y_test_data)
print(num_traj)

# in array numpy per efficienza 
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)

model_points = []

simulazioni = []   

for x,y in zip(x_test_data, y_test_data):
    model_points = []           

    for j in range(max_lag):
        model_points.append(y[j])

    points_traj = len(y)
    for k in range(max(na, nb), points_traj ):
        model_points.append(NARX(theta, regressor_code, x, model_points, k))

    simulazioni.append(model_points)
    

fig, axes = plt.subplots(6,3, figsize=(15, 12), sharex=True, sharey=True)

axes = axes.flatten()
i= 0
rmse_all = []
for y_target, y_estimated in zip(y_test_data, simulazioni):

    rmse = root_mean_squared_error(y_target, y_estimated)
    rmse_all.append(rmse)

    ax = axes[i]
        
    ax.plot(y_target, label="Reale", color="#1a1a1a", linewidth=1.5, alpha=0.7)
    ax.plot(y_estimated, label="NARX", color="#e63946", linestyle="--", linewidth=1.5)
        
    ax.set_title(f"Traiettoria {i+1} (RMSE: {rmse:.2f}Nm)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.5)
    
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")
    i+=1


fig.supxlabel("Campioni Temporali ($k$)", fontsize=12, fontweight='bold')
fig.supylabel("Uscita del Sistema", fontsize=12, fontweight='bold')
fig.suptitle("Test del modello", fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show() 


'''ee = compute_residues_autocorrelation(y_test_data, np.array(simulazioni))
plot_residues_correlation(data=ee, title="Autocorrelation", ylabel="$e^2$")

x1e = compute_cross_correlation(y_test_data, np.array(simulazioni), x_test_data)
plot_residues_correlation(data=x1e, title="Crosscorrelation", ylabel="$x_1e$")
'''

print(f"RMSE medio: {np.mean(rmse_all):.2f}Nm ± {np.std(rmse_all):.2f}Nm")
print(f"RMSE min: {np.min(rmse_all):.2f}Nm  |  RMSE max: {np.max(rmse_all):.2f}Nm")


