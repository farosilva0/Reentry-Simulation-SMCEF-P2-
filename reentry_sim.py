import numpy as np
import pandas as pd
import plot_functions as plot
from scipy.interpolate import CubicSpline


'''CHOOSE SIMULATION OPTIONS'''

# 1. Choose the simulation to run from options below:
SIM_TO_RUN = 2
#------------------------
REENTRY_SIM = 1
PROJECTILE_SIM = 2          # simulation of a projectile being launched with different angles and velocities
#------------------------        


# 2. Choose type of simulation from options below:
SIM_TYPE = 1
#------------------------
NORMAL_SIM = 1
HORIZONTAL_SIM = 2              # we'll start the simulation with some velocity and angle 0, and no forces, so the altitude will remain the same even in round earth
VERTICAL_SIM = 3                # we'll start the simulation without velocity, so with forces object will move vertically
                                # For vertical simulation, make sure LIFT = 0, because if not there will be horizontal movement; try with lift = 0 and = 1 to see the lift effect
#------------------------


# 3. Choose more options:
ROUND_EARTH = False                  # if True we'll simulate the reentry in a round Earth

DRAG_COEFFICIENT = 0      # drag coefficient to use in the simulation
LIFT_COEFFICIENT = 0        # lift coefficient to use in the simulation

CONSTANT_GRAVITY = True            # if True we'll use constant values for gravity
CONSTANT_AIR_DENSITY = True        # if True we'll use constant values for air density

SIM_WITH_PARACHUTE = False          # if True we'll simulate the reentry with deployment of the parachutes after some conditions are met

SHOW_DETAILS = True



'''RUN THIS SCRIPT TO SIMULATE THE CHOOSEN OPTIONS AND YOU DON'T NEED TO CHANGE ANYTHING ELSE BELLOW'''






############################################################################################################
#                                   CONSTANTS AND PARAMETERS
############################################################################################################

# Physical constants
G_M = 6.67430e-11 * 5.972e24    # G (Gravitational constant) * Earth mass, m^3 kg^-1 s^-2
CONSTANT_G = 9.81               # constant gravity (m/s^2)
RADIUS_EARTH = 6.371e6          # Earth radius (m)

# Simulation precision
dt = 0.01                       # time steps (s)


''' Reentry Simulation Parameters - to try different initial angles and velocities and find the best combinations '''

X_0 = 0                                                          # Initial x position (m)
ALTITUDE_0 = 130_000                                                # "interface" == Initial altitude (m)
INIT_VELOCITIES = np.arange(start=0, stop=15_000, step=5_000)    # Possible Initial velocities (m/s)
INIT_ANGLES = np.negative (np.arange(start=0, stop=15, step=5))  # Angles in degrees --> we negate them because the path angle is measured down from the horizon
if SIM_TYPE == HORIZONTAL_SIM:
    INIT_VELOCITIES = [10]  
    INIT_ANGLES = [0]        # so, with now forces, and some initial velocity with initial angle 0 -> the altitude will remain the same even in round earth 
if SIM_TYPE == VERTICAL_SIM:
    INIT_VELOCITIES = [0]  
    INIT_ANGLES = [0]        # so, with now forces, and some initial velocity with initial angle 90 -> the altitude will remain the same even in round earth 
    X_0 = 10_000             # We start with the object at coord (10_000, altitude_0) in the flat earth, and so we'll check if in the round earth the "coming down" is well computed, meaning it will always keep the same vertical position (and it won't be 10_000, but a little less because 10_000 is x value at 130_000m, but on the surface is a little closer to the origin) 

# Capsule parameters
CAPSULE_MASS = 12_000               # Mass of the capsule (kg)
CAPSULE_SURFACE_AREA = 4 * np.pi    # surface considered for drag coefficient (m^2)
CAPSULE_DRAG_COEFFICIENT = DRAG_COEFFICIENT
CAPSULE_LIFT_COEFFICIENT = LIFT_COEFFICIENT

# Parachute parameters
PARACHUTE_SURFACE_AREA = 301                # surface considered for parachute's drag coefficient (m^2)
PARACHUTE_DRAG_COEFFICIENT = 1.0            # parachute's drag coefficient
PARACHUTE_MAX_OPEN_ALTITUDE = 1_000                   # boundary altitude for deployment of the parachutes (m)
PARACHUTE_MAX_OPEN_VELOCITY = 1_00                    # boundary velocity for deployment of the parachutes (m/s)

# Parameter boundaries
MIN_HORIZONTAL_DISTANCE = 2_500_000         # lower boundary for horizontal distance (m)
MAX_HORIZONTAL_DISTANCE = 4_500_000         # higher boundary for horizontal distance (m)
MAX_LANDING_VELOCITY = 25                   # final boundary velocity for deployment (m/s)
MAX_ACCELERATION = 150                      # acceleration boundary for the vessel and crew (m/s^2)

# TODO: dep apagar se não for preciso 
TIME_TO_TRAVEL_90_DEGREES = np.pi / 2 * (RADIUS_EARTH + ALTITUDE_0) / (min(INIT_VELOCITIES) if min(INIT_VELOCITIES) != 0 else 1) # time to cross the equator in a round Earth (s)


''' PROJECTILE SIMULATION '''
if SIM_TO_RUN == PROJECTILE_SIM:
    print("\n" * 20 ,"Projectile simulation")
    ALTITUDE_0 = 0
    X_0 = 0                          
    INIT_VELOCITIES = [500] 
    INIT_ANGLES = [10, 30, 45, 60, 90] 
    CAPSULE_MASS = 1                      # we use same names for the variables for code simplicity
    CAPSULE_SURFACE_AREA = 0.5
    CAPSULE_DRAG_COEFFICIENT = DRAG_COEFFICIENT
    CAPSULE_LIFT_COEFFICIENT = LIFT_COEFFICIENT
    if SIM_TYPE == HORIZONTAL_SIM:
        ALTITUDE_0 = 1_000
        INIT_VELOCITIES = [10]
        INIT_ANGLES = [0]        # so, with now forces, and some initial velocity with initial angle 0 -> the altitude will remain the same even in round earth 
    if SIM_TYPE == VERTICAL_SIM:
        INIT_ANGLES = [90]        # so, with now forces, and some initial velocity with initial angle 90 -> the altitude will remain the same even in round earth 
        INIT_VELOCITIES = [100]
        X_0 = 1_000
else:
    print("\n" * 20 ,"Capsule reentry simulation")


''' Metrics Names to store in the simulation results '''
INIT_ANGLE = 'init_angle'
INIT_VELOCITY = 'init_velocity' 
TIMES = 'times'
PATH_X = 'path_x'
PATH_Y = 'path_y'
VELOCITIES = 'velocities'
ACCELERATIONS = 'accelerations'





############################################################################################################
#                                   AIR DENSITY 
############################################################################################################
DENSITY_CSV = pd.read_csv('air_density.csv')                # Air density table
ALTITUDE = DENSITY_CSV['altitude']                          # Altitude values
AIR_DENSITY = DENSITY_CSV['air_density']                    # Air density values
# @Pre: density file must be ordered by altitude, and have the value for first altitude with 0 air density, and also a much bigger altitude with 0 air density so the interpolation will never be negative (this avoids checking for negative values in the code)
f = CubicSpline(ALTITUDE, AIR_DENSITY, bc_type='natural')   # Cubic spline interpolation for air density


############################################################################################################
#                                   SIMULATION 
############################################################################################################

def make_round_earth (x, y, x_step, y_step, earth_angle):
    ''' given x and y variables (e.g. position or velocities, or accelaration...),
        and the step in each direction in the flat earth, and the earth angle (angle in origin from y axis to current position), 
        converts flat steps to round earth steps and adds it to the x, y variables. '''
    x += y_step * np.sin(earth_angle) + x_step * np.cos(earth_angle)  
    y += y_step * np.cos(earth_angle) - x_step * np.sin(earth_angle)  # y_flat is decreasing because we are going down
    return x, y



def get_tot_acceleration(x, y, vx, vy):
    '''calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.'''
    
    # Common air components for drag and lift
    v_abs = np.sqrt(vx**2 + vy**2)
    air_density =  1.225 if CONSTANT_AIR_DENSITY else f(y - RADIUS_EARTH)
    F_air = 0.5 * CAPSULE_SURFACE_AREA * air_density * v_abs**2 / CAPSULE_MASS # TODO:we do v**2 and then (vx/v) so can be simplified to v;  
   
    # Drag and Lift: 
    ax = - F_air * CAPSULE_DRAG_COEFFICIENT * vx # "-" because it's a resistance force on opposite direction of the velocity
    ay = - F_air * CAPSULE_DRAG_COEFFICIENT * vy 
    if SIM_WITH_PARACHUTE and v_abs <= PARACHUTE_MAX_OPEN_VELOCITY and (y - RADIUS_EARTH) <= PARACHUTE_MAX_OPEN_ALTITUDE:
        # if we open the parachutes, we'll add its drag force in the opposite direction of the velocity
        F_drag_parachute = 0.5 * PARACHUTE_DRAG_COEFFICIENT * PARACHUTE_SURFACE_AREA * air_density * v_abs**2 / CAPSULE_MASS
        ax -= F_drag_parachute * (vx /v_abs) # TODO:we do v**2 and then (vx/v) so can be simplified to v;
        ay -= F_drag_parachute * (vy /v_abs)
    # TODO: implementar lift perpendicular ao vetor velocidade
    # else:
        # lift_force = F_air * CAPSULE_LIFT_COEFFICIENT # * v_abs**2 ??
        # ax += lift_force * (-vy / v_abs)  # Perpendicular component
        # ay += lift_force * (vx / v_abs)   # Perpendicular component
    #     print("3. ax: ", ax, " ay: ", ay)

    # Gravity
    g = CONSTANT_G if CONSTANT_GRAVITY else G_M / (x**2 + y**2) # G_M / r**2 # simplified from: (np.sqrt(x**2 + y**2)**2)
        # TODO: podemos simplificar e tirar sqrt e **2 ??
    ay -= g
    return ax, ay



def run_entry_simulation(angle_0, v_0, altitude_0 = ALTITUDE_0, x_0 = X_0):
    '''runs a simulation of the capsule reentry'''
    if SHOW_DETAILS:
        print("\nsimulation:  angle: ", angle_0, "   init velocity: ", v_0, "   altitude: ", altitude_0, "   x_0: ", x_0)
    # metrics to store
    times = []
    path_x = [] # we discard initial values for simplicity (because we don't know initial values of other metrics like acceleration)
    path_y = []
    velocities = []
    accelerations = [] 
    passed_max_g_limit = False
    passed_max_horizontal_dist = False

    # accumulator variables for the simulation
    time = 0
    x = x_0                         
    y = RADIUS_EARTH + altitude_0  
    earth_angle = np.arctan(x / y)
    
    # initial velocity on flat earth
    angle_0_rad = np.radians(angle_0)
    vx = v_0 * np.cos(angle_0_rad)
    vy = v_0 * np.sin(angle_0_rad)
    if ROUND_EARTH:
        vx, vy = make_round_earth(vx, vy, vx, vy, earth_angle)
        if SHOW_DETAILS:
            print("Changing Velocity to Round Earth")
    if SHOW_DETAILS:
        print( "Starting loops with: x: ", x, "   y: ", y, " (R = ", RADIUS_EARTH,")   vx: ", vx, "   vy: ", vy)
    
    while y >= RADIUS_EARTH:  # more stop conditions are inside the loop so we can store those circumstances
        if (SIM_TYPE == HORIZONTAL_SIM and x > 10_000) or (SIM_TYPE == VERTICAL_SIM and y > RADIUS_EARTH + 10_000): # we'll stop the simulation after a while 
            break

        if x > MAX_HORIZONTAL_DISTANCE:
            print("Max horizontal distance surpassed. Exiting simulation for angle: ", angle_0, "   init velocity: ", v_0)
            print(" -> x: ", x, " y: ", y, " vx: ", vx, " vy: ", vy, " ax: ", ax, " ay: ", ay)
            passed_max_horizontal_dist = True
            break
            
        # time
        time += dt

        # acceleration
        ax, ay = (0, 0) if SIM_TYPE == HORIZONTAL_SIM else get_tot_acceleration(x, y, vx, vy)

        a = np.sqrt(ax**2 + ay**2)
        if(a > MAX_ACCELERATION):  
            print("Max acceleration surpassed. But continuing simulation for angle: ", angle_0, "   init velocity: ", v_0)
            print(" -> x: ", x, " y: ", y, " vx: ", vx, " vy: ", vy, " ax: ", ax, " ay: ", ay)
            passed_max_g_limit = True
            # don't break. continue simulation to store the metrics

        # velocity
        vx_step = ax * dt
        vy_step = ay * dt
        if ROUND_EARTH:
            vx, vy = make_round_earth(vx, vy, vx_step, vy_step, earth_angle)
        else:
            vx += vx_step
            vy += vy_step

        # positions
        x += vx * dt 
        y += vy * dt

        # print("x: ", x, " y: ", y, " vx: ", vx, " vy: ", vy, " ax: ", ax, " ay: ", ay)
        # update earth angle for next step
        earth_angle = np.arctan(x/y) 
        
        if SHOW_DETAILS:
            # store metrics
            times.append(time)
            path_x.append(x)
            path_y.append(y - RADIUS_EARTH)   # altitude above the Earth's surface 
            velocities.append(np.sqrt(vx**2 + vy**2))
            accelerations.append(a)

    sim_results = {
        INIT_ANGLE: angle_0,
        INIT_VELOCITY: v_0,
        PATH_X: path_x,
        PATH_Y: path_y,
        VELOCITIES: velocities,
        ACCELERATIONS: accelerations,
        TIMES: times
    }


    if SHOW_DETAILS:
        print("x:   min: ", min(path_x), "     max: ", max(path_x), "   diff: ", max(path_x) - min(path_x))
        print("y:   min: ", min(path_y), "     max: ", max(path_y), "   diff: ", max(path_y) - min(path_y))
        print("velocities:   min: ", min(velocities), "     max: ", max(velocities))
        print("accelerations:   min: ", min(accelerations), "     max: ", max(accelerations))
        print("times:   min: ", min(times), "     max: ", max(times))
    
    landed_before_min_horizontal_distance = path_x[-1] < MIN_HORIZONTAL_DISTANCE
    passed_max_landing_velocity = velocities[-1] > MAX_LANDING_VELOCITY

    successfull_landing = not passed_max_g_limit and not passed_max_horizontal_dist and not passed_max_landing_velocity and not landed_before_min_horizontal_distance

    return sim_results, successfull_landing



def main():   
    successful_pairs = []   
    tot_sims_metrics = []

    for angle_0 in INIT_ANGLES:
        for v_0 in INIT_VELOCITIES:
            sim_metrics, successfull_landing = run_entry_simulation(angle_0, v_0)
            if successfull_landing:
                successful_pairs.append((angle_0, v_0))
            if SHOW_DETAILS:
                tot_sims_metrics.append(sim_metrics)
    # plot.plot_reentry_parameters(successful_pairs)
    if SHOW_DETAILS:
        plot.plot_sims_metrics(tot_sims_metrics, reentry_sim = (SIM_TO_RUN == REENTRY_SIM))



if __name__ == "__main__":
    main()
