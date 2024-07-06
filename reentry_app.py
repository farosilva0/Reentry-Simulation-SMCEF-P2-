import numpy as np
import pandas as pd
import plot_functions as plot
from scipy.interpolate import CubicSpline


'''CHOOSE SIMULATION OPTIONS'''

# 1. Choose the simulation to run from options below:
SIM_TO_RUN = 1
#------------------------
REENTRY_SIM = 1
PROJECTILE_SIM = 2          # simulation of a projectile being launched with different angles and velocities
#------------------------        


# 2. Choose type of simulation from options below:
SIM_TYPE = 1
#------------------------
NORMAL_SIM = 1                  # we'll start simulation for several angles and velocities
HORIZONTAL_SIM = 2              # we'll start the simulation with some velocity and angle 0, and no forces, so the altitude will remain the same even in round earth
VERTICAL_SIM = 3                # we'll start the simulation without velocity, so with forces object will move vertically
                                # For vertical simulation, make sure LIFT = 0, because if not there will be horizontal movement; try with lift = 0 and = 1 to see the lift effect
ORBITAL_VEL_SIM = 4             # we'll start the simulation with the orbital velocity, so the object will keep the same altitude and will move horizontally
ESCAPE_VEL_SIM = 5              # we'll start the simulation with the escape velocity, so the object will keep the same altitude and will move horizontally
#------------------------


# 3. Choose more options:
ROUND_EARTH = True                  # if True we'll simulate the reentry in a round Earth

DRAG_COEFFICIENT = 1.2      # drag coefficient to use in the simulation
LIFT_COEFFICIENT = 1        # lift coefficient to use in the simulation

LIFT_PERPENDICULAR_TO_VELOCITY = True  # if False, all lift force will be added to y component regardless of velocity direction
                                        # if True, lift force will be perpendicular to velocity direction, and always pointing up
                                        
CONSTANT_GRAVITY = False            # if True we'll use constant values for gravity
CONSTANT_AIR_DENSITY = False        # if True we'll use constant values for air density

SIM_WITH_PARACHUTE = True          # if True we'll simulate the reentry with deployment of the parachutes after some conditions are met

SHOW_DETAILS = True
# TODO: correr com SHOW_DETAILS = False e ver se os resultados são os mesmos, e se não são, ver o que está a ser mostrado que não devia ser mostrado

dt = 0.001                        # time steps (s)
SIM_MAX_TIME = 60 * 15            # max time for the simulation (s)

SIMS_TO_SHOW_IN_PLOT_METRICS = 10 # number of simulations to show in the plot metrics (we don't show all of them to not clutter the plot)

'''RUN THIS SCRIPT TO SIMULATE THE CHOOSEN OPTIONS AND YOU DON'T NEED TO CHANGE ANYTHING ELSE BELLOW'''






############################################################################################################
#                                   CONSTANTS AND PARAMETERS
############################################################################################################

# Physical constants
G_M = 6.67430e-11 * 5.972e24    # G (Gravitational constant) * Earth mass, m^3 kg^-1 s^-2
CONSTANT_G = 9.81               # constant gravity (m/s^2)
RADIUS_EARTH = 6.371e6          # Earth radius (m)


''' Reentry Simulation Parameters - to try different initial angles and velocities and find the best combinations '''

X_0 = 0                                                          # Initial x position (m)
ALTITUDE_0 = 130_000                                               # "interface" == Initial altitude (m)
INIT_ANGLES = [-0, -2, -4, -8] #np.negative(np.arange(start=0, stop=15.1, step=2))  # Angles in degrees --> we negate them because the path angle is measured down from the horizon
INIT_VELOCITIES = [0, 2_000, 4_000, 8_000] #np.arange(start=0, stop=15_001, step=2_000)    # Possible Initial velocities (m/s)
if ROUND_EARTH:
    dt = 0.001
    INIT_ANGLES = [0, 2, 4] # TODO: ver na round earth se angulo inicial é negativo... começa a subir???
    INIT_VELOCITIES = [0, 2_000]
if SIM_TYPE == HORIZONTAL_SIM:
    SIM_MAX_TIME = 20
    INIT_VELOCITIES = [10]  
    INIT_ANGLES = [0]        # so, with now forces, and some initial velocity with initial angle 0 -> the altitude will remain the same even in round earth 
if SIM_TYPE == VERTICAL_SIM:
    INIT_VELOCITIES = [0]  
    INIT_ANGLES = [0]        # so, with now forces, and some initial velocity with initial angle 90 -> the altitude will remain the same even in round earth 
    ALTITUDE_0 = RADIUS_EARTH       # in simulation loop will be added to another                                                # "interface" == Initial altitude (m)
    X_0 = RADIUS_EARTH * 2             # We start with the object at coord (10_000, altitude_0) in the flat earth, and so we'll check if in the round earth the "coming down" is well computed, meaning it will always keep the same vertical position (and it won't be 10_000, but a little less because 10_000 is x value at 130_000m, but on the surface is a little closer to the origin) 
if SIM_TYPE == ORBITAL_VEL_SIM:
    SIM_MAX_TIME = 20
    INIT_VELOCITIES = [np.sqrt(G_M / (RADIUS_EARTH + ALTITUDE_0))]  
    INIT_ANGLES = [0] 
if SIM_TYPE == ESCAPE_VEL_SIM:
    SIM_MAX_TIME = 20
    INIT_VELOCITIES = [np.sqrt(2 * G_M / (RADIUS_EARTH + ALTITUDE_0))]  
    INIT_ANGLES = [0]

# Capsule parameters
CAPSULE_MASS = 12_000               # Mass of the capsule (kg)
CAPSULE_SURFACE_AREA = 4 * np.pi    # surface considered for drag coefficient (m^2)
CAPSULE_DRAG_COEFFICIENT = DRAG_COEFFICIENT
CAPSULE_LIFT_COEFFICIENT = LIFT_COEFFICIENT

# Parachute parameters
PARACHUTE_SURFACE_AREA = 301                # surface considered for parachute's drag coefficient (m^2)
PARACHUTE_DRAG_COEFFICIENT = 1.0            # parachute's drag coefficient
PARACHUTE_MAX_OPEN_ALTITUDE = 1_000         # boundary altitude for deployment of the parachutes (m)
PARACHUTE_MAX_OPEN_VELOCITY = 100           # boundary velocity for deployment of the parachutes (m/s)

# Parameter boundaries
MIN_HORIZONTAL_DISTANCE = 2_500_000         # lower boundary for horizontal distance (m)
MAX_HORIZONTAL_DISTANCE = 4_500_000         # higher boundary for horizontal distance (m)
MAX_LANDING_VELOCITY = 25                   # final boundary velocity for deployment (m/s)
MAX_ACCELERATION = 150                      # acceleration boundary for the vessel and crew (m/s^2)


''' PROJECTILE SIMULATION '''
if SIM_TO_RUN == PROJECTILE_SIM:
    print("\n" * 20 ,"Projectile simulation")
    ALTITUDE_0 = 0
    X_0 = 0                          
    INIT_VELOCITIES = [100] 
    INIT_ANGLES = [10, 30, 45, 60, 90] 
    CAPSULE_MASS = 10                      # we use same names for the variables for code simplicity
    CAPSULE_SURFACE_AREA = 2     
    CAPSULE_DRAG_COEFFICIENT = DRAG_COEFFICIENT
    CAPSULE_LIFT_COEFFICIENT = LIFT_COEFFICIENT
    if SIM_TYPE == HORIZONTAL_SIM:
        ALTITUDE_0 = 1_000
        INIT_ANGLES = [0]        # so, with now forces, and some initial velocity with initial angle 0 -> the altitude will remain the same even in round earth 
    if SIM_TYPE == VERTICAL_SIM:
        INIT_VELOCITIES = [1_000]
        INIT_ANGLES = [90]        # so, with now forces, and some initial velocity with initial angle 90 -> the altitude will remain the same even in round earth 
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


def get_air_density_cubic_spline(y):
    altitude = y - RADIUS_EARTH
    result = f(altitude)
    return result if result > 0.0 else 0.0


def get_total_acceleration(y, vx, vy):
    '''calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.'''
    # Common air components for drag and lift
    v_abs = np.sqrt(vx**2 + vy**2)
    air_density = 1.225 if CONSTANT_AIR_DENSITY else get_air_density_cubic_spline(y)

    F_air_drag = -0.5 * CAPSULE_SURFACE_AREA * air_density * CAPSULE_DRAG_COEFFICIENT * (v_abs)**2
    ax = F_air_drag * vx / (v_abs * CAPSULE_MASS)
    ay = F_air_drag * vy / (v_abs * CAPSULE_MASS)

    if SIM_TO_RUN == REENTRY_SIM and SIM_WITH_PARACHUTE and v_abs <= PARACHUTE_MAX_OPEN_VELOCITY and (y - RADIUS_EARTH) <= PARACHUTE_MAX_OPEN_ALTITUDE:
        # if we open the parachutes, we'll add its drag force in the opposite direction of the velocity
        F_air_drag_parachute = -0.5 * PARACHUTE_DRAG_COEFFICIENT * PARACHUTE_SURFACE_AREA * air_density * (v_abs)**2
        F_air_drag += F_air_drag_parachute
        ax += F_air_drag_parachute * vx / (v_abs * CAPSULE_MASS)
        ay += F_air_drag_parachute * vy / (v_abs * CAPSULE_MASS)
        F_lift = 0
    else:
        F_lift = -0.5 * CAPSULE_SURFACE_AREA * air_density * CAPSULE_LIFT_COEFFICIENT * (v_abs)**2
        ay -= (F_lift / CAPSULE_MASS)

    a = (F_lift - F_air_drag) / CAPSULE_MASS
    
    # Gravity
    g = G_M / y**2
    ay -= g

    return ax, ay, a


def run_entry_simulation(angle_0, v_0, altitude_0 = ALTITUDE_0, x_0 = X_0):
    '''runs a simulation of the capsule reentry'''
    if SHOW_DETAILS:
        print("\nsimulation:  path_angle: ", angle_0, "   init velocity: ", v_0, "   altitude: ", altitude_0, "   x_0: ", x_0)
    # metrics to st
    # ore
    times = []
    path_x = [] # we discard initial values for simplicity (because we don't know initial values of other metrics like acceleration)
    path_y = []
    velocities = []
    accelerations = []
    passed_max_g_limit = False

    # accumulator variables for the simulation
    time = 0
    x = x_0                         
    y = RADIUS_EARTH + altitude_0  
    earth_angle = np.radians(90) - np.arctan2(y, x) # angle in origin from y axis to current position to convert flat earth steps to round earth steps
    x_round = RADIUS_EARTH * earth_angle if ROUND_EARTH else x
    y_round = (np.sqrt(x**2 + y**2) - RADIUS_EARTH) if ROUND_EARTH else y # altitude above the Earth's surface
    accumulated_horizontal_distance = 0             # accumulated horizontal distance in round earth to double check if the horizontal distance is well computed
    
    # initial velocity
    angle_0_rad = np.radians(angle_0)
    vx = v_0 * np.cos(angle_0_rad)
    vy = v_0 * np.sin(angle_0_rad)
    if SHOW_DETAILS:
        print( "Starting loops with: x: ", x, "   y: ", y, " (R = ", RADIUS_EARTH,")   vx: ", vx, "   vy: ", vy, "   earth_angle: ", np.degrees(earth_angle))
    steps = 0
    while y > RADIUS_EARTH:  # more stop conditions are inside the loop so we can store those circumstances
        show_details = steps % (200_000) == 0 # we'll show details at this intervals 
        steps += 1
        if show_details: 
            if SHOW_DETAILS:
                print("t ",round(time,1)," - init:  x: ", round(x, 0), " y: ", round(y, 0), " vx: ", round(vx, 0), " vy: ", round(vy, 0), "     -->> sim: ang: ", angle_0, "    vel: ", v_0)
        if time > SIM_MAX_TIME:
            print("Max time surpassed. Exiting simulation for angle: ", angle_0, "   init velocity: ", v_0)
            break
                   
        # time
        time += dt

        # acceleration
        ax, ay, a = (0, 0) if SIM_TYPE == HORIZONTAL_SIM else get_total_acceleration(y, vx, vy)

        if(a > MAX_ACCELERATION): # we check if the acceleration is too high, without counting the gravity = 10 m/s^2
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
        
        v = np.sqrt(vx**2 + vy**2)
        if show_details:
            print("t ",round(time,1)," - vel update:    vx_step: ", round(vx_step, 4), "   vy_step: ", round(vy_step, 4), "  angle: ", round(np.degrees(earth_angle),1), "   vx: ", round(vx,2), "   vy: ", round(vy,2) )
        
        # previous x positions
        previous_x = x # to update accumulated_horizontal_distance

        # positions
        x += vx * dt 
        y += vy * dt
        if show_details:
            print("t ",round(time,1)," - pos update:    x_step: ", round(vx * dt, 2), "   y_step: ", round(vy * dt, 2) ,"     x: ", round(x,2), "   y: ", round(y,2))

        # update earth angle for next step
        earth_angle = np.radians(90) - np.arctan2(y, x) # angle in origin from y axis to current position
        accumulated_horizontal_distance += (x - previous_x) / y # get accumulated horizontal distance in round earth to double check if the horizontal distance is well computed
        
        if SHOW_DETAILS: # and steps % 1_000 == 0: # we don't need all steps for the plots
            # store metrics
            times.append(time)
            x_round = RADIUS_EARTH * earth_angle if ROUND_EARTH else x
            y_round = (np.sqrt(x**2 + y**2) - RADIUS_EARTH) if ROUND_EARTH else (y - RADIUS_EARTH) # altitude above the Earth's surface
            if show_details and ROUND_EARTH:
                print("t ",round(time,1)," - (x,y) in round earth:    x_round: ", round(x_round,2), "   y_round: ", round(y_round,2))
            path_x.append(x_round)
            path_y.append(y_round)    
            velocities.append(v)
            accelerations.append(a)

    accumulated_horizontal_distance *= RADIUS_EARTH

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
        print("end t ",round(time,1)," - (x,y) in round earth:    x_round: ", round(x_round,2), "   y_round: ", round(y_round,2), "    vx: ", round(vx,2), "   vy: ", round(vy,2))
        print("x_min:  ", round(min(path_x),2), "     x_max: ", round(max(path_x),2), "   x_horizontal_distance: ", round(max(path_x) - min(path_x),2))
        if not ROUND_EARTH:
            print("accumulated round earth horizontal distance after flat earth calculations: ", round(accumulated_horizontal_distance,2))
        print("y_min: ", round(min(path_y),2), "     y_max: ", round(max(path_y),2), "   y_vertical_distance: ", round(max(path_y) - min(path_y),2))
        print("vel_min: ", round(min(velocities),2), "     vel_max: ", round(max(velocities),2), "  vel_final: ", round(velocities[-1],2) )
        print("acc_min: ", round(min(accelerations),2), "     acc_max: ", round(max(accelerations),2), "  acc_final: ", round(accelerations[-1],2))
        print("time duration:  " , round(max(times) - min(times),2) , "s")
    
    landed_after_min_horizontal_distance = x >= MIN_HORIZONTAL_DISTANCE
    landed_before_max_horizontal_distance = x <= MAX_HORIZONTAL_DISTANCE
    landed_below_max_landing_velocity = v <= MAX_LANDING_VELOCITY

    successfull_landing = not passed_max_g_limit and landed_below_max_landing_velocity and landed_after_min_horizontal_distance and landed_before_max_horizontal_distance

    return sim_results, successfull_landing, passed_max_g_limit, landed_below_max_landing_velocity, landed_after_min_horizontal_distance, landed_before_max_horizontal_distance, 



def main():

    successful_pairs = []
    landed_before = []
    landed_after = []
    acceleration_pairs = []
    velocity_pairs = []

    if SHOW_DETAILS:
        sims_to_show = min(SIMS_TO_SHOW_IN_PLOT_METRICS, len(INIT_ANGLES) * len(INIT_VELOCITIES))
        axs = plot.start_sims_metrics_plot(SIM_TO_RUN == REENTRY_SIM, sims_to_show)
        random_sim_to_show = np.random.choice(len(INIT_ANGLES) * len(INIT_VELOCITIES), size=sims_to_show, replace=False)
    sim_number = 0
    for angle_0 in INIT_ANGLES:
        for v_0 in INIT_VELOCITIES:
            sim_metrics, successfull_landing, g_limit, velocity_limit, horizontal_min, horizontal_max = run_entry_simulation(angle_0, v_0)
            if successfull_landing:
                successful_pairs.append((angle_0, v_0))
            elif g_limit:
                acceleration_pairs.append((angle_0, v_0))
            elif not velocity_limit:
                velocity_pairs.append((angle_0, v_0))
            elif horizontal_min:
                landed_before.append((angle_0, v_0))
            elif horizontal_max:
                landed_after.append((angle_0, v_0))
            
            # if g_limit:
            #     acceleration_pairs.append((angle_0, v_0))
            # if velocity_limit:
            #     velocity_pairs.append((angle_0, v_0))
            # if horizontal_landing_limit:
            #     distance_pairs.append((angle_0, v_0))
            if SHOW_DETAILS:
                if sim_number in random_sim_to_show:
                    plot.plot_sim_metrics(axs, sim_metrics, SIM_TO_RUN == REENTRY_SIM)
    if SHOW_DETAILS:
        plot.end_sims_metrics_plot()
    if SIM_TO_RUN == REENTRY_SIM:
        plot.plot_reentry_parameters(successful_pairs)
    plot.plot_reentry_parameters(successful_pairs)
    plot.plot_all_reentrys(successful_pairs, acceleration_pairs, velocity_pairs, landed_before, landed_after)
    


if __name__ == "__main__":
    main()
    print()
