from scipy.interpolate import CubicSpline
import time
import numpy as np
import pandas as pd
import plot_functions as plot



# SIMULATION PARAMETERS
dt = 0.01                                                         # time steps (s)
g = -10.0                                                         # gravity acceleration (m/s^2)
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
VELOCITY_FINAL_BOUNDARY = 25                    # final boundary velocity for deployment (m/s)
LOWER_HORIZONTAL_FLAT_BOUNDARY = 5.0*10**5      # lower boundary for horizontal distance in a flat heart (m)
HIGHER_HORIZONTAL_FLAT_BOUNDARY = 2.5*10**6     # higher boundary for horizontal distance in a flat heart (m)
LOWER_HORIZONTAL_BOUNDARY = 2.5*10**6           # lower boundary for horizontal distance (m)
HIGHER_HORIZONTAL_BOUNDARY = 4.5*10**6          # higher boundary for horizontal distance (m)
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

    return 0.0 if result < 0.0 else result



def drag_acceleration(vx, vy, v, y, A = CAPSULE_SURFACE_AREA, Cd = CAPSULE_DRAG_COEFFICIENT):
    '''calculates the drag acceleration on the object, given its velocity and altitude.
        By default, the drag force is calculated for the capsule.
        Drag force always opposes the movement direction, so it is negative.'''
    Fd = -0.5 * Cd * A * get_air_density_cubic_spline(y) * v**2
    ax = Fd * (vx / (v * CAPSULE_MASS))
    ay = Fd * (vy / (v * CAPSULE_MASS))
    return ax, ay


def drag_acceleration_parachute(vx, vy, v, y, A = PARACHUTE_SURFACE_AREA, Cd = PARACHUTE_DRAG_COEFFICIENT):
    '''calculates the drag acceleration on the object after the parachutes are deployed, given its velocity and altitude.
        The mass of the parachutes is despised.
        Drag force always opposes the movement direction, so it is negative.'''
    Fd = -0.5 * Cd * A * get_air_density_cubic_spline(y) * v**2
    ax = Fd * (vx / (v * CAPSULE_MASS))
    ay = Fd * (vy / (v * CAPSULE_MASS))
    return ax, ay


def lift_acceleration(vx, vy, v, y, A = CAPSULE_SURFACE_AREA, Cl = CAPSULE_LIFT_COEFFICIENT):
    '''calculates the lift acceleration on the object, given its velocity and altitude.
        By default, the lift force is calculated for the capsule.
        Lift force is perpendicular to the movement direction, so it is calculated using the velocity components.'''
    Fl = -0.5 * Cl * A * get_air_density_cubic_spline(y) * v**2
    ax = Fl * (vx / (v * CAPSULE_MASS))
    ay = Fl * (vy / (v * CAPSULE_MASS))
    return ax, ay


def gravity_acceleration(x, y):
    '''calculates the acceleration due to gravity at a given position (x, y)
        returns the acceleration components gx and gy. '''
    altitude = y - RADIUS_EARTH
    r = np.sqrt(x**2 + altitude**2)    # distance from the center of the Earth
    g = - G_M / r**2            # gravity acceleration pointing to the center of the Earth (negative because it goes down)       
    g *= (altitude / r)


def total_acceleration(x, y, vx, vy):
    '''calculates the total acceleration on the capsule'''

    v = np.sqrt(vx**2 + vy**2) # velocity in the movement's direction
    dx, dy = drag_acceleration(vx, vy, v, y)

    # print(f"drag_acceleration x: {dx}")   # prints para debug
    # print(f"drag_acceleration y: {dy}")
    if (y - RADIUS_EARTH) <= ALTITUDE_BOUNDARY and v <= VELOCITY_BOUNDARY: # case where the parachutes are deployed and there is no lift resistance
        resistance_x, resistance_y = drag_acceleration_parachute(vx, vy, v, y)
        # print(f"drag_acceleration_parachute x: {resistance_x}")
        # print(f"drag_acceleration_parachute y: {resistance_y}")
    else: 
        resistance_x, resistance_y = lift_acceleration(vx, vy, v, y)
        # print(f"lift_acceleration x: {resistance_x}")
        # print(f"lift_acceleration y: {resistance_y}")

    total_ax = dx + resistance_x
    total_ay = dy + g + resistance_y

    gravity_acceleration(x, y)

    return total_ax, total_ay


def run_entry_simulation(altitude_0, angle_0, v_0):
    '''runs a simulation of the capsule reentry'''

    # initial velocity
    tetha = np.radians(-angle_0)
    vx_0 = v_0 * np.cos(tetha)
    vy_0 = v_0 * np.sin(tetha)

    # accumulator variables for the simulation
    x = 0                           # x position
    y = RADIUS_EARTH + altitude_0   # y position
    vx = vx_0                       # x velocity
    vy = vy_0                       # y velocity
    
    path_x = [x]
    path_y = [y - RADIUS_EARTH]

    surpassed_boundary = False
    
    while y > RADIUS_EARTH:
        ''' reentry simulation using Euler's method'''
        '''TODO: Runge-Kutta methods for better accuracy'''
        '''TODO: analitic methods...'''
        # print("-------------------------------------------------------------------------------")
        # total acceleration
        ax, ay = total_acceleration(x, y, vx, vy)
        a = np.sqrt(ax**2 + ay**2)

        # print(f"aceleração total: {a}")
        if(a > ACCELERATION_BOUNDARY):  # total acceleration was higher tham the limit for a successful reentry
            surpassed_boundary = True

        # velocity 
        vx += ax * dt
        vy += ay * dt

        # print(f"velocidade x: {vx}")      # prints para debug
        # print(f"velocidade y: {vy}")
        # print(f"velocidade: {np.sqrt(vx**2 + vy**2)}")
        
        # positions
        x += vx * dt
        y += vy * dt
        
        # store positions
        path_x.append(x)
        path_y.append(y - RADIUS_EARTH)  # Altitude em relação ao nível do mar
    
    v_final = np.sqrt(vx**2 + vy**2)
               
    return path_x, path_y, surpassed_boundary, v_final, x


def main():

    successful_angles = []
    time_start = time.time()

    for angle_0 in initial_angles:
        successful_velocities = []
        for v_0 in initial_velocities:
            path_x, path_y, surpassed_boundary, v, horizontal_x = run_entry_simulation(altitude_0, angle_0, v_0)
            plot.plot_path(path_x, path_y)      # para debug, pois o tempo tomado para encerrar o separador do plot conta para o tempo total da execução
            if v < VELOCITY_FINAL_BOUNDARY and not surpassed_boundary: # and LOWER_HORIZONTAL_FLAT_BOUNDARY <= horizontal_x <= HIGHER_HORIZONTAL_FLAT_BOUNDARY:
                successful_velocities.append(v_0)
        successful_velocities = np.array(successful_velocities, dtype="int")
        angle = np.full(successful_velocities.size, angle_0, dtype=float)
        successful_angles.append((angle, successful_velocities))
                
    successful_angles = np.array(successful_angles, dtype=object)
    time_end = time.time()

    print(f"Time to run the simulation: {time_end - time_start} seconds")

    plot.plot_reentry_parameters(successful_angles)


if __name__ == "__main__":
    main()
