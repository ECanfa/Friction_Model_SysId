import pandas
import matplotlib.pyplot as plt
import numpy as np

SIM_DT = 0.001
RAMP_DURATION = 0.3
INPUT_CSV = "output.csv"
NUM_VELOCITIES = 10

if INPUT_CSV == "output.csv":
    DURATION_PER_STEP = 0.5
else:
    DURATION_PER_STEP = 0.6

points_per_traj = int(DURATION_PER_STEP*NUM_VELOCITIES/SIM_DT)
data = pandas.read_csv(INPUT_CSV, sep = ',')

print(points_per_traj)

friction_estimated = data["torque_friction_estimated_Nm"]
velocities = data["velocity_rad_s"]
velocities = velocities[:points_per_traj]
time = data["t"]
velocities_sorted = np.argsort(velocities)


t_traj_1 = time[61*points_per_traj:62*points_per_traj] %5 
friction_traj_1 = friction_estimated[61*points_per_traj:62*points_per_traj] 


plt.plot(t_traj_1, friction_traj_1, color='black', label='Attrito stimato')
plt.title('Attrito')
plt.xlabel('tempo [s]')
plt.ylabel('Attrito [Nm]')
plt.legend()
plt.show()

plt.scatter(data["velocity_rad_s"], data["torque_friction_estimated_Nm"], s=1)
plt.show()