import numpy as np
import pandas as pd
import plot_functions as plot
from scipy.interpolate import CubicSpline



# SIMULATION PARAMETERS
dt = 0.01                                                         # time steps (s)
altitude_0 = 130_000                                              # Initial altitude (m)
initial_angles = np.arange(start=0, stop=15.1, step=1)            # Possible entry angles (degrees)
initial_velocities = np.arange(start=0, stop=15_100, step=1_000)  # Possible Initial velocities (m/s)

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

    # Gravity
    r = np.sqrt(x**2 + y**2)    # distance from the center of the Earth
    g = -G_M / r**2             # gravity acceleration pointing to the center of the Earth

    ay += g

    return ax, ay


def run_entry_simulation(altitude_0, angle, v_0):
    '''runs a simulation of the capsule reentry'''

    # initial velocity
    tetha = np.radians(-angle)
    vx_0 = v_0 * np.cos(tetha)
    vy_0 = v_0 * np.sin(tetha)

    # accumulator variables for the simulation
    x = 0                           # x position
    y = RADIUS_EARTH + altitude_0   # y position
    vx = vx_0                       # x velocity
    vy = vy_0                       # y velocity
    
    path_x = [x]
    path_y = [y - RADIUS_EARTH]
    time = 0

    surpassed_boundary = False
    
    while y > RADIUS_EARTH:
        ''' reentry simulation using Euler's method'''

        # update time
        time += dt

        # total acceleration
        ax, ay = get_tot_acceleration(x, y, vx, vy)
        a = np.sqrt(ax**2 + ay**2)

        if(a > ACCELERATION_BOUNDARY):  # total acceleration was higher tham the limit for a successful reentry
            surpassed_boundary = True

        # velocity
        vx += ax * dt
        vy += ay * dt

        # positions
        x += vx * dt
        y += vy * dt
        
        # store positions
        path_x.append(x)
        path_y.append(y - RADIUS_EARTH)  # Altitude em relação ao nível do mar
    
    v_final = np.sqrt(vx**2 + vy**2)

    print(f"Horizontal distance at the Earth surface: {RADIUS_EARTH * angle}")
    print(f"Time to execute the simulation: {time} seconds")
               
    return path_x, path_y, surpassed_boundary, v_final, x


def main():

    successful_angles = []

    for angle_0 in initial_angles:
        successful_velocities = []
        for v_0 in initial_velocities:
            path_x, path_y, surpassed_boundary, v, horizontal_x = run_entry_simulation(altitude_0, angle_0, v_0)
            # plot.plot_path(path_x, path_y)      # para debug, pois o tempo tomado para encerrar o separador do plot conta para o tempo total da execução
            if v < VELOCITY_FINAL_BOUNDARY and not surpassed_boundary: # and LOWER_HORIZONTAL_FLAT_BOUNDARY <= horizontal_x <= HIGHER_HORIZONTAL_FLAT_BOUNDARY:
                successful_velocities.append(v_0)
        successful_velocities = np.array(successful_velocities, dtype="int")
        angle = np.full(successful_velocities.size, angle_0, dtype=float)
        successful_angles.append((angle, successful_velocities))
                
    successful_angles = np.array(successful_angles, dtype=object)

    plot.plot_reentry_parameters(successful_angles)


if __name__ == "__main__":
    main()
