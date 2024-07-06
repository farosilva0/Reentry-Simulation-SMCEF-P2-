import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
M = 5.972e24  # mass of the Earth (kg)
R_earth = 6371000  # radius of the Earth (m)
C_d = 1.0  # drag coefficient (dimensionless)
C_L = 0.3  # lift coefficient (dimensionless)
m = 1000  # mass of the spacecraft (kg)
A = 1.0  # cross-sectional area of the spacecraft (m^2)
rho_0 = 1.225  # sea-level atmospheric density (kg/m^3)
H = 8500  # scale height of the atmosphere (m)

# Target horizontal distance range (in meters)
min_distance = 50000  # minimum horizontal distance (m)
max_distance = 200000  # maximum horizontal distance (m)

# Ranges of initial velocities (m/s) and angles (degrees)
velocity_range = np.linspace(2000, 10000, 10)  # example range of initial velocities
angle_range = np.linspace(-75, -5, 10)  # example range of entry angles

# Define the system of ODEs
def spacecraft_ode(t, y):
    r, x, vr, vx = y
    v = np.sqrt(vr**2 + vx**2)
    rho = rho_0 * np.exp(-(r - R_earth) / H)
    g = G * M / r**2
    L = 0.5 * rho * v**2 * C_L * A
    alpha = np.arctan2(vr, vx)
    
    drdt = vr
    dxdt = vx * (R_earth / (R_earth + r))  # adjusted for curvature
    dvrdt = -g + (1 / (2 * m)) * rho * C_d * A * v * vr + (L / m) * np.sin(alpha)
    dvxdt = -(1 / (2 * m)) * rho * C_d * A * v * vx + (L / m) * np.cos(alpha)
    
    return [drdt, dxdt, dvrdt, dvxdt]

# Find valid pairs of (angle, initial velocity)
valid_pairs = []

for v0 in velocity_range:
    for angle in angle_range:
        v0_r = v0 * np.cos(np.radians(angle))
        v0_x = v0 * np.sin(np.radians(angle))
        y0 = [R_earth + 130000, 0, v0_r, v0_x]  # initial position and velocities
        
        # Step 1: Quick Estimate of Touchdown Time
        t_span_quick = (0, 2000)  # rough estimate time span
        solution_quick = solve_ivp(spacecraft_ode, t_span_quick, y0, max_step=10.0)
        
        # Check if the spacecraft hits the ground (altitude <= Earth's radius)
        ground_index_quick = np.where(solution_quick.y[0] <= R_earth)[0]
        if ground_index_quick.size > 0:
            estimated_touchdown_time = solution_quick.t[ground_index_quick[0]]
        else:
            estimated_touchdown_time = 5000  # use the maximum time if not hit in the estimate
        
        # Step 2: Refined Simulation
        t_span_refined = (0, estimated_touchdown_time * 1.1)  # add 10% margin
        t_eval = np.linspace(0, estimated_touchdown_time * 1.1, 1000)
        solution = solve_ivp(spacecraft_ode, t_span_refined, y0, t_eval=t_eval)
        
        # Check if the spacecraft hits the ground (altitude <= Earth's radius)
        ground_index = np.where(solution.y[0] <= R_earth)[0]
        if ground_index.size > 0:
            landing_index = ground_index[0]
            horizontal_distance = solution.y[1, landing_index]  # horizontal distance
            if min_distance <= horizontal_distance <= max_distance:
                valid_pairs.append((angle, v0))

# Print valid pairs
for angle, velocity in valid_pairs:
    print(f"Angle: {angle} degrees, Initial Velocity: {velocity} m/s")

# Optionally, plot one of the valid trajectories
if valid_pairs:
    angle, velocity = valid_pairs[0]
    v0_r = velocity * np.cos(np.radians(angle))
    v0_x = velocity * np.sin(np.radians(angle))
    y0 = [R_earth + 130000, 0, v0_r, v0_x]
    t_span_refined = (0, estimated_touchdown_time * 1.1)
    t_eval = np.linspace(0, estimated_touchdown_time * 1.1, 1000)
    solution = solve_ivp(spacecraft_ode, t_span_refined, y0, t_eval=t_eval)
    
    r = solution.y[0]
    x = solution.y[1]
    
    plt.figure(figsize=(10, 5))
    plt.plot(x / 1000, (r - R_earth) / 1000)  # convert meters to kilometers, and adjust for Earth's radius
    plt.xlabel('Horizontal position (km)')
    plt.ylabel('Altitude (km)')
    plt.title('Spacecraft reentry trajectory with gravity, drag, and lift (2D model)')
    plt.grid(True)
    plt.ylim(bottom=0)  # Ensure the vertical position doesn't go below zero
    plt.show()
