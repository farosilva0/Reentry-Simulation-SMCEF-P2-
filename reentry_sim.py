import numpy as np
import pandas as pd
import plot_functions as plot
from scipy.interpolate import CubicSpline


'''SIMULATION OPTIONS'''
SHOW_DETAILS = True
RUN_OTHER_SIM = 2 # 0: CAPSULE, 1: PROJECTILE, 2: FREE FALL
USE_CONSTANT_FORCES = False
ROUND_EARTH = False


# Physical constants
G_M = 6.67430e-11 * 5.972e24    # G (Gravitational constant) * Earth mass, m^3 kg^-1 s^-2
RADIUS_EARTH = 6.371e6          # Earth radius (m)




# REENTRY SIMULATION WITH GRAVITY, DRAG and LIFT
dt = 0.01                                                         # time steps (s)
altitude_0 = 130_000                                              # Initial altitude (m)
V_ORBITAL = np.sqrt(G_M / (RADIUS_EARTH + altitude_0))      # velocities to test round earth formulas (y must stay at the same altitude)
V_ESCAPE = np.sqrt(2 * G_M / (RADIUS_EARTH + altitude_0))   # y must go to infinity
initial_angles = np.negative (np.arange(start=0, stop=20, step=40)) # np.arange(start=0, stop=15, step=1) Possible entry angles (degrees) - negative because the angle is measured from the horizon
initial_velocities = [V_ESCAPE] #np.arange(start=0, stop=15_000, step=1_000)  # Possible Initial velocities (m/s)

# Capsule parameters
CAPSULE_MASS = 12_000               # Mass of the capsule (kg)
CAPSULE_SURFACE_AREA = 4 * np.pi    # surface considered for drag coefficient (m^2)
CAPSULE_DRAG_COEFFICIENT = 1.2      # drag coefficient
CAPSULE_LIFT_COEFFICIENT = 1     # lift coefficient

# Parachute parameters
PARACHUTE_SURFACE_AREA = 301              # surface considered for parachute's drag coefficient (m^2)
PARACHUTE_DRAG_COEFFICIENT = 1.0          # parachute's drag coefficient
ALTITUDE_BOUNDARY = 1_000                 # boundary altitude for deployment of the parachutes (m)
VELOCITY_BOUNDARY = 1_00                 # boundary velocity for deployment of the parachutes (m/s)

# Parameter boundaries
# LOWER_HORIZONTAL_FLAT_BOUNDARY = 5.0*10**5      # lower boundary for horizontal distance in a flat heart (m)
# HIGHER_HORIZONTAL_FLAT_BOUNDARY = 2.5*10**6     # higher boundary for horizontal distance in a flat heart (m)
LOWER_HORIZONTAL_BOUNDARY = 2_500_000           # lower boundary for horizontal distance (m)
HIGHER_HORIZONTAL_BOUNDARY = 4_500_000          # higher boundary for horizontal distance (m)
VELOCITY_FINAL_BOUNDARY = 25                    # final boundary velocity for deployment (m/s)
ACCELERATION_BOUNDARY = 150                     # acceleration boundary for the vessel and crew (m/s^2)

# Density parameters
DENSITY_CSV = pd.read_csv('air_density.csv')                # Air density table
ALTITUDE = DENSITY_CSV['altitude']                          # Altitude values
AIR_DENSITY = DENSITY_CSV['air_density']                    # Air density values
f = CubicSpline(ALTITUDE, AIR_DENSITY, bc_type='natural')   # Cubic spline interpolation for air density
    




# PROJECTILE SIM
if RUN_OTHER_SIM == 1:
    print("Projectile simulation")
    altitude_0 = 0                                            # Initial altitude (m)
    initial_angles = [10, 20, 30, 45, 60, 75, 90] 
    initial_velocities = [100] 
    CAPSULE_MASS = 0.8 
    CAPSULE_SURFACE_AREA = 0.02
    CAPSULE_DRAG_COEFFICIENT = 1      # drag coefficient
    CAPSULE_LIFT_COEFFICIENT = 10      # lift coefficient


# FREE FALL SIM
elif RUN_OTHER_SIM == 2:
    print("Free fall simulation")
    altitude_0 = 1_000  # Initial altitude (m)
    initial_angles = [0] 
    initial_velocities = np.array([25, 50, 100, 500]) # com vel mt grandes é quase como se tivessemos um wing suit e conseguissemos vencer a gravidade, até o drag fazer velocidade reduzir 
    CAPSULE_MASS = 80 
    CAPSULE_SURFACE_AREA = 0.5
    CAPSULE_DRAG_COEFFICIENT = 1      # drag coefficient
    CAPSULE_LIFT_COEFFICIENT = 0.6     # lift coefficient

else:
    print("Capsule reentry simulation")


def get_air_density_cubic_spline(y):
    altitude = y - RADIUS_EARTH
    result = f(altitude)
    return result if result > 0.0 else 0.0


def get_tot_acceleration(x, y, vx, vy):
    '''calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.'''
    v = np.sqrt(vx**2 + vy**2) # velocity in the movement's direction

    air_density = get_air_density_cubic_spline(y)
    if USE_CONSTANT_FORCES:
        air_density = 1.225 # constant air density 

    # Common air components for drag and lift
    F_air = - 0.5 * CAPSULE_SURFACE_AREA * air_density * v / CAPSULE_MASS # we do v**2 and then (vx/v) so can be simplified to v; 
    
    # Drag: 
    ax = F_air * CAPSULE_DRAG_COEFFICIENT * vx # "-" because it's a resistance force
    ay = F_air * CAPSULE_DRAG_COEFFICIENT * vy 
    # F_drag = -0.5 * CAPSULE_DRAG_COEFFICIENT * CAPSULE_SURFACE_AREA * air_density * v # we do v**2 and then (vx/v) so can be simplified to v
    # ax = F_drag * (vx / CAPSULE_MASS)
    # ay = F_drag * (vy / CAPSULE_MASS)

    # print(RUN_OTHER_SIM == True, ((y - RADIUS_EARTH) >= ALTITUDE_BOUNDARY), (v >= VELOCITY_BOUNDARY))
    if RUN_OTHER_SIM or (y - RADIUS_EARTH) >= ALTITUDE_BOUNDARY or v >= VELOCITY_BOUNDARY: # case where the parachutes are not deployed
        ax += F_air * CAPSULE_LIFT_COEFFICIENT * vx 
        ay += F_air * CAPSULE_LIFT_COEFFICIENT * vy 
        # F_lift = -0.5 * CAPSULE_LIFT_COEFFICIENT * CAPSULE_SURFACE_AREA * air_density * v
        # ax += F_lift * (vx / CAPSULE_MASS)
        # ay += F_lift * (vy / CAPSULE_MASS)

    else: # if (y - RADIUS_EARTH) <= ALTITUDE_BOUNDARY and v <= VELOCITY_BOUNDARY:      # case where the parachutes are deployed and there is no lift resistance
        F_drag_parachute = -0.5 * PARACHUTE_DRAG_COEFFICIENT * PARACHUTE_SURFACE_AREA * air_density * v / CAPSULE_MASS
        ax += F_drag_parachute * vx 
        ay += F_drag_parachute * vy 

    # Gravity
    r = np.sqrt(x**2 + y**2)    # distance from the center of the Earth
    g = G_M / r**2             # gravity acceleration pointing to the center of the Earth
    if USE_CONSTANT_FORCES:
        g = 9.81   # constant gravity acceleration
    
    if ROUND_EARTH:
        ay -= g * (y / r) # we reduce the amount of g by multiplying with the division of cathetus y by the hypotenuse r, which is almost 1 but not exactly
        ax -= g * (x / r) # we reduce the x aceleration accounting for the curvature of the Earth  
    else:
        ay -= g

    return ax, ay


def run_entry_simulation(altitude_0, path_angle, v_0):
    '''runs a simulation of the capsule reentry'''

    # initial velocity
    path_angle = np.radians(path_angle)
    vx_0 = v_0 * np.cos(path_angle)
    vy_0 = v_0 * np.sin(path_angle)

    # accumulator variables for the simulation
    earth_angle = 0                 # angle at center of the Earth made by the movement (radians)
    x = 0                           # x position
    y = RADIUS_EARTH + altitude_0   # y position
    v = v_0
    vx = vx_0                       # x velocity
    vy = vy_0                       # y velocity
    
    time = 0
    times = []
    path_x = [] # we discard initial values for simplicity (because we don't know initial values of other metrics like acceleration)
    path_y = []
    velocities = []
    accelerations = [] 
    
    surpassed_g_boundary = False

    steps = 0
    while y >= RADIUS_EARTH and steps < 1_000_000: # pra se meter vel mt grande e ele fica a orbitar e nunca aterra
        steps += 1
        ''' reentry simulation using Euler's method'''
        # update time
        time += dt

        # total acceleration
        ax, ay = get_tot_acceleration(x, y, vx, vy)
        
        a = np.sqrt(ax**2 + ay**2)
        if(a > ACCELERATION_BOUNDARY):  # total acceleration was higher tham the limit for a successful reentry
            surpassed_g_boundary = True

        # velocity
        vx += ax * dt
        vy += ay * dt

        # previous x positions
        previous_x = x

        # positions
        x += vx * dt
        y += vy * dt

        # angle update for round earth
        earth_angle = (x - previous_x)/y
         
        # store positions
        path_x.append(x)
        path_y.append(y - RADIUS_EARTH)  # Altitude em relação ao nível do mar

        # store velocities
        v = np.sqrt(vx**2 + vy**2)
        velocities.append(v)
        accelerations.append(a)
        times.append(time)

    # formula do enunciado (mas isto tem de ser feito a cada passo prós plots... ??)
    x_on_round_earth = RADIUS_EARTH * earth_angle

    sim_results = {
        plot.INIT_ANGLE: path_angle,
        plot.INIT_VELOCITY: v_0,
        plot.PATH_X: path_x,
        plot.PATH_Y: path_y,
    }
    if SHOW_DETAILS:
        print("final x position: ", x)
        print("final time: ", time)
        print(f"earth_angle: 0 -> ", earth_angle)
        print(f"final velocity: {v}")


        sim_results[plot.VELOCITIES] = velocities
        sim_results[plot.ACCELERATIONS] = accelerations
        sim_results[plot.TIMES] = times

    return sim_results, surpassed_g_boundary, v, x_on_round_earth


def main():
   
    successful_pairs = []   
    tot_sims_metrics = []

    for angle_0 in initial_angles:
        for v_0 in initial_velocities:
            print(f"\nRunning simulation for angle: {angle_0} and initial velocity: {v_0}")
            sim_metrics, surpassed_g_boundary, v_final, x_on_round_earth = run_entry_simulation(altitude_0, angle_0, v_0)
            if v_final < VELOCITY_FINAL_BOUNDARY and not surpassed_g_boundary: # and LOWER_HORIZONTAL_FLAT_BOUNDARY <= horizontal_x <= HIGHER_HORIZONTAL_FLAT_BOUNDARY:
                successful_pairs.append((angle_0, v_0))
            if SHOW_DETAILS:
                tot_sims_metrics.append(sim_metrics)

    # plot.plot_reentry_parameters(successful_pairs)
    if SHOW_DETAILS:
        plot.plot_sims_metrics(tot_sims_metrics)



if __name__ == "__main__":
    main()
