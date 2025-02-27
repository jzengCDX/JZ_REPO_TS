import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Generate noisy measurements of a hypothetical rejection patient

# stable part
pos_start = 0.12    #need to verify with literature
velocity_start = 0
velocity_rej = 0.1
num_measurements1 = 50
num_measurements2 = 20
measurement_noise_std = 0.05    #need to verify analytical/biological variance
dt = 1  # time step

np.random.seed(0)

velocities_known = [velocity_start] * num_measurements1 + [velocity_rej] * num_measurements2
positions_known = pos_start + np.concatenate((velocity_start * np.arange(num_measurements1), velocity_rej * np.arange(num_measurements2)))
measurements = positions_known + measurement_noise_std * np.random.randn(num_measurements1+num_measurements2)


# Initialize Kalman filter
kf1 = KalmanFilter(dim_x=2, dim_z=1)
kf1.x = np.array([[pos_start], [velocity_start]])  # initial state (position and velocity)
kf1.F = np.array([[1, dt], [0, 1]])  # state transition matrix
kf1.H = np.array([[1, 0]])  # measurement function
kf1.P = np.array([[1, 0], [0, 1]])  # covariance matrix
kf1.R = np.array([[measurement_noise_std**2]])  # measurement noise
kf1.Q = np.array([[1e-5, 0], [0, 1e-5]])  # process noise

kf2 = KalmanFilter(dim_x=2, dim_z=1)
kf2.x = np.array([[pos_start], [velocity_start]])  # initial state (position and velocity)
kf2.F = np.array([[1, dt], [0, 1]])  # state transition matrix
kf2.H = np.array([[1, 0]])  # measurement function
kf2.P = np.array([[1, 0], [0, 1]])  # covariance matrix
kf2.R = np.array([[measurement_noise_std**2]])  # measurement noise
kf2.Q = np.array([[1e-5, 0], [0, 1e-4]])  # process noise

kf3 = KalmanFilter(dim_x=2, dim_z=1)
kf3.x = np.array([[pos_start], [velocity_start]])  # initial state (position and velocity)
kf3.F = np.array([[1, dt], [0, 1]])  # state transition matrix
kf3.H = np.array([[1, 0]])  # measurement function
kf3.P = np.array([[1, 0], [0, 1]])  # covariance matrix
kf3.R = np.array([[measurement_noise_std**2]])  # measurement noise
kf3.Q = np.array([[1e-5, 0], [0, 1e-3]])  # process noise

kf4 = KalmanFilter(dim_x=2, dim_z=1)
kf4.x = np.array([[pos_start], [velocity_start]])  # initial state (position and velocity)
kf4.F = np.array([[1, dt], [0, 1]])  # state transition matrix
kf4.H = np.array([[1, 0]])  # measurement function
kf4.P = np.array([[1, 0], [0, 1]])  # covariance matrix
kf4.R = np.array([[measurement_noise_std**2]])  # measurement noise
kf4.Q = np.array([[1e-5, 0], [0, 1e-2]])  # process noise

kfs = [kf1, kf2, kf3, kf4]

# Arrays to store the results
estimated_positions = [[],[],[],[]]
estimated_velocities = [[],[],[],[]]
kalman_gains = [[],[],[],[]]
residuals = [[],[],[],[]]
covariances = [[],[],[],[]]

for measurement in measurements:
    for x_i in range(len(kfs)):
        kf = kfs[x_i]

        kf.predict()
        kf.update(measurement)
        estimated_positions[x_i].append(kf.x[0, 0])
        estimated_velocities[x_i].append(kf.x[1, 0])
        kalman_gains[x_i].append(kf.K[0, 0])
        residuals[x_i].append(measurement - kf.x[0, 0])
        covariances[x_i].append(kf.P[0, 0])

# Plot the results
plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.plot(positions_known, 'k-', label='Known Truth')
plt.plot(measurements, 'k+', label='Noisy measurements')
for x_i in range(len(kfs)):
    plt.plot(estimated_positions[x_i], '-', label=f'Kalman filter {x_i} estimate')

plt.legend()
plt.xlabel('Measurement number')
plt.ylabel('cfDNA Position')
plt.title('cfDNA Position: Estimates vs Known')

plt.subplot(3, 2, 2)
plt.plot(velocities_known, 'k-', label='Known Velocity')
for x_i in range(len(kfs)):
    plt.plot(estimated_velocities[x_i], '-', label=f'Kalman filter {x_i} estimate')
plt.legend()
plt.xlabel('Measurement number')
plt.ylabel('cfDNA Velocity')
plt.title('cfDNA Velocity: Estimates vs Known')

plt.subplot(3, 2, 3)
ix_s = num_measurements1-15
ix_e = num_measurements1+10
plt.plot(positions_known[ix_s:ix_e], 'k-', label='Known Truth')
plt.plot(measurements[ix_s:ix_e], 'k+', label='Noisy measurements')
for x_i in range(len(kfs)):
    plt.plot(estimated_positions[x_i][ix_s:ix_e], '-', label=f'Kalman filter {x_i} estimate')

plt.legend()
plt.xlabel('Measurement number')
plt.ylabel('cfDNA Position')
plt.title('cfDNA Position: Estimates vs Known')

plt.subplot(3, 2, 4)
plt.plot(velocities_known[ix_s:ix_e], 'k-', label='Known Velocity')
for x_i in range(len(kfs)):
    plt.plot(estimated_velocities[x_i][ix_s:ix_e], '-', label=f'Kalman filter {x_i} estimate')
plt.legend()
plt.xlabel('Measurement number')
plt.ylabel('cfDNA Velocity')
plt.title('cfDNA Velocity: Estimates vs Known')

plt.subplot(3, 2, 5)
for x_i in range(len(kfs)):
    plt.plot(kalman_gains[x_i], '-', label=f'Kalman Gain {x_i}')
plt.legend()
plt.xlabel('Measurement number')
plt.ylabel('Kalman Gain')
plt.title('Kalman Gain over Time')

plt.subplot(3, 2, 6)
for x_i in range(len(kfs)):
    plt.plot(covariances[x_i], '-', label=f'Estimation Error Covariance {x_i}')
plt.legend()
plt.xlabel('Measurement number')
plt.ylabel('Covariance')
plt.title('Estimation Error Covariance over Time')

plt.tight_layout()

plt.savefig('../figs/240821_lineplots_rej_model_process_noise.png')

plt.show()