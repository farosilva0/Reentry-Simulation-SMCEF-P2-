import numpy as np
import pandas as pd
import plot_functions as plot
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp



# SIMULATION PARAMETERS
dt = 0.01                                                         # time steps (s)
altitude_0 = 130_000                                              # Initial altitude (m)
initial_angles = np.arange(start=0, stop=4.1, step=1)            # Possible entry angles (degrees)
initial_velocities = np.arange(start=0, stop=4_100, step=1_000)  # Possible Initial velocities (m/s)

# Capsule parameters
CAPSULE_MASS = 12_000               # Mass of the capsule (kg)
CAPSULE_SURFACE_AREA = 4 * np.pi    # surface considered for drag coefficient (m^2)
CAPSULE_DRAG_COEFFICIENT = 1.2      # drag coefficient
CAPSULE_LIFT_COEFFICIENT = 1.0      # lift coefficient

# Parachute parameters
PARACHUTE_SURFACE_AREA = 301              # surface considered for parachute's drag coefficient (m^2)
PARACHUTE_DRAG_COEFFICIENT = 1.0          # parachute's drag coefficient
ALTITUDE_BOUNDARY = 10**3                 # boundary altitude for deployment of the parachutes (m)
VELOCITY_BOUNDARY = 10**2                 # boundary velocity for deployment of the parachutes (m/s)

# Parameter boundaries
LOWER_HORIZONTAL_FLAT_BOUNDARY = 5.0*10**5      # lower boundary for horizontal distance in a flat heart (m)
HIGHER_HORIZONTAL_FLAT_BOUNDARY = 2.5*10**6     # higher boundary for horizontal distance in a flat heart (m)
LOWER_HORIZONTAL_BOUNDARY = 2.5*10**6           # lower boundary for horizontal distance (m)
HIGHER_HORIZONTAL_BOUNDARY = 4.5*10**6          # higher boundary for horizontal distance (m)
VELOCITY_FINAL_BOUNDARY = 25                    # final boundary velocity for deployment (m/s)
ACCELERATION_BOUNDARY = 150                     # acceleration boundary for the vessel and crew (m/s^2)

# Density parameters
DENSITY_CSV = pd.read_csv('air_density.csv')                # Air density table
ALTITUDE = DENSITY_CSV['altitude']                          # Altitude values
AIR_DENSITY = DENSITY_CSV['air_density']                    # Air density values
f = CubicSpline(ALTITUDE, AIR_DENSITY, bc_type='natural')   # Cubic spline interpolation for air density

# Physical constants
G_M = 6.67430e-11 * 5.972e24    # G (Gravitational constant) * Earth mass, m^3 kg^-1 s^-2
RADIUS_EARTH = 6.371e6          # Earth radius (m)


def get_air_density_cubic_spline(y):
    altitude = y - RADIUS_EARTH
    result = f(altitude)
    return result if result > 0.0 else 0.0


def get_tot_acceleration(x, y, vx, vy):
    '''calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.'''
    v = np.sqrt(vx**2 + vy**2) # velocity in the movement's direction

    air_density = get_air_density_cubic_spline(y)

    # Drag acceleration
    F_air = -0.5 * CAPSULE_DRAG_COEFFICIENT * CAPSULE_SURFACE_AREA * air_density * v # we do v**2 and then (vx/v) so can be simplified to v
    ax = F_air * (vx / CAPSULE_MASS)
    ay = F_air * (vy / CAPSULE_MASS)

    if (y - RADIUS_EARTH) <= ALTITUDE_BOUNDARY and v <= VELOCITY_BOUNDARY:      # case where the parachutes are deployed and there is no lift resistance
        F_air_parachute = -0.5 * PARACHUTE_DRAG_COEFFICIENT * PARACHUTE_SURFACE_AREA * air_density * v
        ax += F_air_parachute * (vx / CAPSULE_MASS)
        ay += F_air_parachute * (vy / CAPSULE_MASS)
    else:                                                                       # case where the parachutes are not deployed
        F_lift = -0.5 * CAPSULE_LIFT_COEFFICIENT * CAPSULE_SURFACE_AREA * air_density * v
        ax += F_lift * (vx / CAPSULE_MASS)
        ay += F_lift * (vy / CAPSULE_MASS)

    return ax, ay


def solve_ivp_reentry_simulation(t, array):
    x, vx, y, vy = array
    ax, ay = get_tot_acceleration(x, y, vx, vy)
    return [vx, -np.sqrt(vx**2 + vy**2)*vx - ax * dt, vy, -1-np.sqrt(vx**2 + vy**2)*vy - ay * dt]


def main():

    angle_0 = 45 * np.pi / 180 # 45, valor para obter os resultados do gráfico
    v_0 = 2 # 2, 4, 6, 8, 10, valores para obter os resultados do gráfico

    sol1 = solve_ivp(solve_ivp_reentry_simulation, [0, 60], y0=[0, v_0*np.cos(angle_0), 0, v_0*np.sin(angle_0)], t_eval=np.linspace(0, 60, 100000))
    plot.plot_sim(sol1, angle_0)


if __name__ == "__main__":
    main()
