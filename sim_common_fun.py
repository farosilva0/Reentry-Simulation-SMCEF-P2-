import numpy as np
from scipy.interpolate import CubicSpline

import sim_plots as plot

from sim_params import *



''' option for, when printing the numpy array values, to show only 2 decimal places, and not in scientific notation. '''
np.set_printoptions(suppress=True, precision=3)


############################################################################################################
#                                   AIR DENSITY 
############################################################################################################

# air density interpolation
# @Pre: the source document has a table with the air density values for each altitude until further distances in both directions, so we don't incur in errrs on the edges of the table
air_dens_f = CubicSpline(ALTITUDE, AIR_DENSITY, bc_type='natural')   # Cubic spline interpolation for air density




############################################################################################################
#                                   SIMULATION 
############################################################################################################

def make_round_earth (x_flat_step, y_flat_step, earth_angle):
    ''' given the step in each direction in the flat earth, and the earth angle (angle in origin from y axis to current position), 
        converts flat steps to round earth steps. '''
    x_round_step = y_flat_step * np.sin(earth_angle) + x_flat_step * np.cos(earth_angle)  
    y_round_step = y_flat_step * np.cos(earth_angle) - x_flat_step * np.sin(earth_angle)  # y_flat is decreasing because we are going down
    return x_round_step, y_round_step

def lift_perpendicular_to_velocity(vx, vy, a_lift, p: Params):
    # get vel angle
    vel_angle = np.arctan2(vy, vx)
    # get lift angle perpendicular to velocity
    lift_angle = vel_angle + np.pi/2
    # to guarantee lift is always up
    if lift_angle > np.pi:     
        lift_angle -= np.pi 
    # use attack angle to increase lift over y component as much as possible
    if lift_angle > np.pi/2 - p.max_angle_of_attack and lift_angle < np.pi/2 + p.max_angle_of_attack:
        lift_angle = np.pi/2                        # if with max angle of attack we can make the lift up, so we do it 
    elif lift_angle < np.pi/2 - p.max_angle_of_attack:
        lift_angle += p.max_angle_of_attack         # if lift is much smaller than 90º, we increase it to the max angle of attack
    else:
        lift_angle -= p.max_angle_of_attack         # if lift is much bigger than 90º, we decrease it to the max angle of attack
    # calculate lift acceleration components   
    a_lift_x = a_lift * np.cos(lift_angle)
    a_lift_y = a_lift * np.sin(lift_angle)
    return a_lift_x, a_lift_y


def get_acceleration(Sk, Mk, p: Params): 
    ''' Calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.
        Updates metrics (M) with the total acceleration in the current state, without counting gravity aceleration.'''
    
    x, y, vx, vy = Sk

    v, a, acc_angle, chute_open = Mk
       
    # variables commonly used in the calculations
    y = 1e-10 if y == 0 else y
    v = 1e-10 if v == 0 else v
    v_mass = v * p.capsule_mass 
    vx_v_mass = vx / v_mass
    vy_v_mass = vy / v_mass

    air_density = air_dens_f((np.sqrt(x**2 + y**2) if p.sim_round_earth else y) - RADIUS_EARTH)

    # Air drag
    F_air_drag = 0.5 * p.capsule_surface_area * air_density * p.capsule_drag_coefficient * v**2
    ax = - F_air_drag * vx_v_mass
    ay = - F_air_drag * vy_v_mass

    if v <= p.parachute_max_open_velocity and (np.sqrt(x**2 + y**2) if p.sim_round_earth else y) - RADIUS_EARTH <= p.parachute_max_open_altitude and p.is_reentry_sim and p.sim_with_parachute:
        # Parachute drag
        F_air_drag_parachute = 0.5 * p.parachute_drag_coefficient * p.parachute_surface_area * air_density * v**2
        ax -= F_air_drag_parachute * vx_v_mass
        ay -= F_air_drag_parachute * vy_v_mass
        Mk[A] = (F_air_drag + F_air_drag_parachute) / p.capsule_mass
        Mk[CHUTE_OPEN] = 1
    else:
        # Lift 
        F_lift = 0.5 * p.capsule_surface_area * air_density * p.capsule_lift_coefficient * v**2
        a_lift = F_lift / p.capsule_mass
        if p.lift_perpendicular_to_velocity:
            a_lift_x, a_lift_y = lift_perpendicular_to_velocity(vx, vy, a_lift, p)
            ax += a_lift_x
            ay += a_lift_y
        else:            
            ay += a_lift # lift is always up, and that is guaranteed by v**2 wich makes lift force positive
        Mk[A] = (F_air_drag - F_lift) / p.capsule_mass

    # Gravity
    g = G_M / y**2
    ay -= g

    return ax, ay, Mk

    

def reentry_slope(Sk, Mk, p:Params):
    ''' given the previous state (Sk), returns the derivatives (ODEs) for each variable of the System [x, y, vx, vy]
        and updates the metrics for the current state with the value of total acceleration without gravity. 
        returns Sk1 as a new vector by value, not altering the original Sk. 
        Mk is updated by reference, so it is changed in the original variable.'''
    
    slopes = np.zeros(4, dtype=float)
    # vel derivatives == aceleeration in the current state (given considering current velocity and altitude)
    ax, ay, Mk = (0,0,Mk) if p.is_horizontal_sim else get_acceleration(Sk, Mk, p)
    if p.sim_round_earth:
        ax, ay = make_round_earth(ax, ay, Mk[EARTH_ANGLE])

    slopes[VX] = ax
    slopes[VY] = ay
        
    # x and y derivatives == velocity in the current state (given considering current velocity)
    slopes[X] = Sk[VX]
    slopes[Y] = Sk[VY]

    return slopes, Mk



def run_one_simulation(S0, M0, p: Params, method_f):
    def print_state(S, M, str, end="\r"):
        print(f"{str}:   x:{S[X]:_.2f}    y:{(S[Y] - RADIUS_EARTH):_.2f}" + (f"    alt:{np.sqrt(S[X]**2 + S[Y]**2) - RADIUS_EARTH:_.2f}" if p.sim_round_earth else "") +      f"    v:{M[V]:_.2f}    a:{M[A]:_.2f}    acc_angle:{M[EARTH_ANGLE]:_.2f}    chute_open?:{M[CHUTE_OPEN] == True}", end=end)
    
    size = int(p.sim_max_time / p.dt + 1)
    S = np.zeros((size, S0.shape[0]), dtype=float) # System variables: [x, y, vx, vy]
    M = np.zeros((size, M0.shape[0]), dtype=float) # Other metrics: [v, a, acc_angle, chute_open]
    t = np.array([i * p.dt for i in range(size)])
    
    S[0] = S0
    M[0] = M0

    for i in range(1, size):
        S[i], M[i] = method_f(S[i-1], M[i-1], p, reentry_slope)
        if (np.sqrt(S[i][X]**2 + S[i][Y]**2) if p.sim_round_earth else S[i][Y]) < RADIUS_EARTH:
            print_state(S[i], M[i], "Landed", end="\n")
            return S[:i+1], M[:i+1], t[:i+1]
        if i % 5_000 == 0:       
            print_state(S[i], M[i], "i: " + str(i))
            if S[i][X] > plot.MAX_HORIZONTAL_DISTANCE_TO_PLOT:
                break
    print_state(S[size-1], M[size-1], "Time out", end="\n")
    return S, M, t



def run_all_simulations(method_f, run_with_solver_ivp=False):
    
    # Lists to store the results of the simulations
    successful_pairs = []
    acceleration_pairs = []
    velocity_pairs = []
    landed_before = []
    landed_after = []

    p = get_params()
    print("\n"*20, "Running simulations with parameters: \n", p)

    # prepare plots to be done: if too many simulations, we will show only a few of them
    total_sims = len(p.init_angles) * len(p.init_velocities)
    total_sims_to_show = min(p.sims_to_show_in_plot_metrics, total_sims)
    fig, axs = plot.start_sims_metrics_plot(p, total_sims_to_show)
    sims_to_show = np.random.choice(total_sims, size=total_sims_to_show, replace=False)
    
    # Run all simulations
    sim_number = 0
    for angle_0 in p.init_angles:
        for v_0 in p.init_velocities:
            print(f"---> sim {sim_number + 1} of {total_sims} - angle (", angle_0, ")    velocity (", v_0, ")")
            
            # Initial state
            angle_0_rad = np.radians(angle_0)
            vx = v_0 * np.cos(angle_0_rad)
            vy = v_0 * np.sin(angle_0_rad)
            S0 = np.array([p.x_0, p.altitude_0 + RADIUS_EARTH, vx, vy]) # S b= [X, Y, VX, VY]
            M0 = np.array([v_0, 0, 0, 0]) # M = [V, A, ACC_EARTH_ANGLE, CHUTE_OPEN] 
            
            # Run the simulation
            if run_with_solver_ivp: 
                S, M, t = method_f(S0, M0, p, get_acceleration)
            else:
                S, M, t = run_one_simulation(S0, M0, p, method_f)
            
            # Update Y positions before we plot them
            if p.sim_round_earth: 
                S[:, Y] = np.sqrt(S[:, X]**2 + (S[:, Y])**2)    # in flat earth y is a cathetus (vertical distance from x axis); in round earth y is hipotenuse (distance from origin); so we use pythagoras to convert it
            
            # Update X positions to round earth before we plot them (position x is the arc length of round earth: x = R * angle)
            S[:, X] = np.array(M[:, EARTH_ANGLE] * RADIUS_EARTH)  # one method, using the angle accumulated in the simulation
            final_x = S[:, X][-1]
            
            # Check success conditions of the simulation
            if np.any(M[:,A] > p.max_acceleration):
                acceleration_pairs.append((angle_0, v_0))
            elif M[-1][V] > p.max_landing_velocity:
                velocity_pairs.append((angle_0, v_0))
            elif final_x < p.min_horizontal_distance:
                landed_before.append((angle_0, v_0))
            elif final_x > p.max_horizontal_distance:
                landed_after.append((angle_0, v_0))
            else:
                successful_pairs.append((angle_0, v_0))
            
            # Plot this simulation metrics (to avoid storing them all in then ploting them all at once)
            if sim_number in sims_to_show:
                sim_metrics = {
                    PATH_X: S[1:, X],
                    PATH_Y: S[1:, Y] - RADIUS_EARTH,
                    ABS_VELOCITIES: M[1:, V],
                    Y_VELOCITIES: S[1:, VY],
                    ACCELERATIONS: M[1:, A],
                    CHUTE_OPENING: M[1:, CHUTE_OPEN],
                    TIMES: t[1:]
                }
                plot.plot_sim_metrics(axs, sim_metrics, angle_0, v_0, p.is_reentry_sim, p)
            sim_number += 1

    plot.end_sims_metrics_plot(fig, axs, p)
    if p.is_reentry_sim:
        plot.plot_all_reentrys(successful_pairs, acceleration_pairs, velocity_pairs, landed_before, landed_after, p)
    



