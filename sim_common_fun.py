import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import sim_plots as plot

from sim_params import *
from run_various_sims import *



''' option for, when printing the numpy array values, to show only 2 decimal places, and not in scientific notation. '''
np.set_printoptions(suppress=True, precision=3)


############################################################################################################
#                                   AIR DENSITY 
############################################################################################################
DENSITY_CSV = pd.read_csv('air_density.csv')                # Air density table
ALTITUDE = DENSITY_CSV['altitude']                          # Altitude values
AIR_DENSITY = DENSITY_CSV['air_density']                    # Air density values
f = CubicSpline(ALTITUDE, AIR_DENSITY, bc_type='natural')   # Cubic spline interpolation for air density

def get_air_density_cubic_spline(altitude):
    result = f(altitude)
    return result if result > 0.0 else 0.0



############################################################################################################
#                                   SIMULATION 
############################################################################################################

def make_round_earth (x, y, x_step, y_step, earth_angle):
    ''' given x and y variables (e.g. velocities, or positions...),
        and the step in each direction in the flat earth, and the earth angle (angle in origin from y axis to current position), 
        converts flat steps to round earth steps and adds it to the x, y variables. '''
    x += y_step * np.sin(earth_angle) + x_step * np.cos(earth_angle)  
    y += y_step * np.cos(earth_angle) - x_step * np.sin(earth_angle)  # y_flat is decreasing because we are going down
    return x, y


def get_acceleration(Sk, Mk, p: Params): 
    ''' Calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.
        Updates metrics (M) with the total acceleration in the current state, without counting gravity aceleration.'''
    
    x, y, vx, vy = Sk
    v, a, acc_horiz_dist, chute_open = Mk   
    
    # variables commonly used in the calculations
    y = 1e-20 if y == 0 else y
    v = 1e-20 if v == 0 else v
    v_mass = v * p.capsule_mass 
    vx_v_mass = vx / v_mass
    vy_v_mass = vy / v_mass

    air_density = get_air_density_cubic_spline(y - RADIUS_EARTH)

    # Air drag
    F_air_drag = 0.5 * p.capsule_surface_area * air_density * p.capsule_drag_coefficient * v**2
    ax = - F_air_drag * vx_v_mass
    ay = - F_air_drag * vy_v_mass
    # print("init values in acc: x: ", round(x, 2), "  y:", round(y, 2), "  vx:", round(vx, 2), "  vy:", round(vy, 2), "  v:", round(v, 2), " v_mass:", round(v_mass, 2))

    if v <= p.parachute_max_open_velocity and (np.sqrt(x**2 + y**2) if p.sim_round_earth else y) - RADIUS_EARTH <= p.parachute_max_open_altitude and p.is_reentry_sim and p.sim_with_parachute:
        # Parachute drag
        F_air_drag_parachute = 0.5 * p.parachute_drag_coefficient * p.parachute_surface_area * air_density * v**2
        ax -= F_air_drag_parachute * vx_v_mass
        ay -= F_air_drag_parachute * vy_v_mass
        Mk[A] = (F_air_drag + F_air_drag_parachute) / p.capsule_mass
        Mk[CHUTE_OPEN] = 1
        # print("F_drag_parachute: ", round(F_air_drag_parachute, 2), "\t-->> ax:", round(ax, 2), " ay:", round(ay, 2))
    else:
        # Lift
        F_lift = 0.5 * p.capsule_surface_area * air_density * p.capsule_lift_coefficient * v**2
        ay += (F_lift / p.capsule_mass)
        Mk[A] = (F_air_drag - F_lift) / p.capsule_mass
        # print("F_lift:           ", round(F_lift, 2), "\t-->> ax:", round(ax, 2), " ay:", round(ay, 2))

    # Gravity
    g = G_M / y**2
    ay -= g
    # print("F_g:               ", round(g, 2), "\t-->> ax:", round(ax, 2), " ay:", round(ay, 2), "  a:", round(np.sqrt(ax**2 + ay**2), 2), "  a_without_G:", round(Mk[A], 2))
    return ax, ay, Mk

    

def reentry_slope(Sk, Mk, p:Params):
    ''' given the previous state (Sk), returns the derivatives (ODEs) for each variable of the System [x, y, vx, vy]
        and updates the metrics for the current state with the value of total acceleration without gravity. 
        returns Sk1 as a new vector by value, not altering the original Sk. 
        Mk is updated by reference, so it is changed in the original variable.'''
    
    slopes = np.zeros(4, dtype=float)
    # vel derivatives == aceleeration in the current state (given considering current velocity and altitude)
    ax, ay, Mk = get_acceleration(Sk, Mk, p)
    slopes[VX] = ax
    slopes[VY] = ay

    # x and y derivatives == velocity in the current state (given considering current velocity)
    slopes[X] = Sk[VX]
    slopes[Y] = Sk[VY]

    return slopes, Mk



    
def run_one_simulation(S0, M0, p: Params, method_f):
    size = int(p.sim_max_time / p.dt + 1)
    S = np.zeros((size, S0.shape[0]), dtype=float) # System variables: [x, y, vx, vy]
    M = np.zeros((size, M0.shape[0]), dtype=float) # Other metrics: [v, a, accumulated_horizontal_distance]
    t = np.array([i * p.dt for i in range(size)])
    
    S[0] = S0
    M[0] = M0

    for i in range(1, size):
        S[i], M[i] = method_f(S[i-1], M[i-1], p, reentry_slope)
        if (np.sqrt(S[i][X]**2 + S[i][Y]**2) if p.sim_round_earth else S[i][Y]) < RADIUS_EARTH:
            print(f"Landed:  M: ", M[i])
            return S[:i+1], M[:i+1], t[:i+1]
    print(f"Time out:  M: ", M[i])
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
    axs = plot.start_sims_metrics_plot(p, total_sims_to_show)
    sims_to_show = np.random.choice(total_sims, size=total_sims_to_show, replace=False)
    
    # Run all simulations
    sim_number = 0
    for angle_0 in p.init_angles:
        for v_0 in p.init_velocities:
            print(f"------------------------> sim {sim_number + 1} of {total_sims} - angle: ", angle_0, "    velocity: ", v_0)
            
            # Initial state
            angle_0_rad = np.radians(angle_0)
            vx = v_0 * np.cos(angle_0_rad)
            vy = v_0 * np.sin(angle_0_rad)
            S0 = np.array([p.x_0, p.altitude_0 + RADIUS_EARTH, vx, vy]) # S b= [X, Y, VX, VY]
            M0 = np.array([v_0, 0, 0, 0]) # M = [V, A, ACC_EARTH_ANGLE] 
            
            # Run the simulation
            if run_with_solver_ivp: 
                S, M, t = method_f(S0, M0, p, get_acceleration)
            else:
                S, M, t = run_one_simulation(S0, M0, p, method_f)
            
            # Update Y positions before we plot them
            if p.sim_round_earth: 
                assert p.sim_round_earth, "fghj"                              # if ROUND_EARTH is True, we convert y positions from flat earth to round earth (this needs to be done before updating x positions, but subtracting RADIUS_EARTH only after adapting x positions to round earth)
                S[:, Y] = np.sqrt(S[:, X]**2 + (S[:, Y])**2)    # in flat earth y is a cathetus (vertical distance from x axis); in round earth y is hipotenuse (distance from origin); so we use pythagoras to convert it
            
            # Update X positions to round earth before we plot them (position x is the arc length of round earth: x = R * angle)
            S[:, X] = np.array(M[:, EARTH_ANGLE] * RADIUS_EARTH)  # one method, using the angle accumulated in the simulation
            x_tg = RADIUS_EARTH * (np.radians(90) - np.arctan2(S[:, Y], S[:, X] - p.x_0)) # another method, using positions x and y to calculate the total angle directly
            final_x = S[:, X][-1]
            # print("final_x with arctg: ", round(final_x, 2), "       with acc angle: ", round(x_acc[-1], 2), "    diff: ", round(final_x - x_acc[-1], 2))
            
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

    plot.end_sims_metrics_plot(axs, p)
    if p.is_reentry_sim:
        plot.plot_all_reentrys(successful_pairs, acceleration_pairs, velocity_pairs, landed_before, landed_after)
    



