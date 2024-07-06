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

def get_acceleration(Sk, Mk, p: Params): 
    ''' Calculates the total acceleration on the capsule, which depends if the parachutes are deployed or not.
        Updates metrics (M) with the total acceleration in the current state, without counting gravity aceleration.'''
    
    x, y, vx, vy = Sk
    v, a, acc_horiz_dist = Mk
    
    if v == 0:
        v = 0.00000000000001 # TODO: simplificar formulas pra não se dividir por v
    
    air_density = get_air_density_cubic_spline(y - RADIUS_EARTH)
    
    F_air_drag = -0.5 * p.capsule_surface_area * air_density * p.capsule_drag_coefficient * v**2
    v_mass = v * p.capsule_mass
    # print("init values in acc: x: ", round(x, 2), "  y:", round(y, 2), "  vx:", round(vx, 2), "  vy:", round(vy, 2), "  v:", round(v, 2), "  air_density:", round(air_density, 2), " v_mass:", round(v_mass, 2))
    
    ax = F_air_drag * vx / v_mass
    ay = F_air_drag * vy / v_mass
    # print("F_air_drag:        ", round(F_air_drag, 2), "\t-->> ax:", round(ax, 2), " ay:", round(ay, 2))

    if v <= p._parachute_max_open_velocity and (y - RADIUS_EARTH) <= p._parachute_max_open_altitude:
        F_air_drag_parachute = -0.5 * p._parachute_drag_coefficient * p._parachute_surface_area * air_density * v**2
        F_air_drag += F_air_drag_parachute
        ax += F_air_drag_parachute * vx / v_mass
        ay += F_air_drag_parachute * vy / v_mass
        F_lift = 0
        # print("F_drag_parachute: ", round(F_air_drag_parachute, 2), "\t-->> ax:", round(ax, 2), " ay:", round(ay, 2))
    else:
        F_lift = - 0.5 * p.capsule_surface_area * air_density * p.capsule_lift_coefficient * v**2
        ay -= (F_lift / p.capsule_mass)
        # print("F_lift:           ", round(F_lift, 2), "\t-->> ax:", round(ax, 2), " ay:", round(ay, 2))
    
    Mk[A] = (F_lift - F_air_drag) / p.capsule_mass
    
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



    
def run_one_simulation(S0, M0, p: Params, method_f, slope_f):
    size = int(p.sim_max_time / p.dt + 1)
    
    S = np.zeros((size, S0.shape[0]), dtype=float) # System variables: [x, y, vx, vy]
    M = np.zeros((size, M0.shape[0]), dtype=float) # Other metrics: [v, a, accumulated_horizontal_distance]
    t = np.array([i * p.dt for i in range(size)])
    
    S[0] = S0
    M[0] = M0

    for i in range(1, size):
        S[i], M[i] = method_f(S[i-1], M[i-1], p, slope_f)
        if S[i][Y] < RADIUS_EARTH:
            print("Landed at step ", i)
            return S[:i+1], M[:i+1], t[:i+1]
    
    return S, M, t



def run_all_simulations(method_f):

    successful_pairs = []
    acceleration_pairs = []
    velocity_pairs = []
    landed_before = []
    landed_after = []

    p = get_params()
    print("\n"*20, "Running simulations with parameters: \n", p)

    if p.show_details:
        sims_to_show = min(p.sims_to_show_in_plot_metrics, len(p.init_angles) * len(p.init_velocities))
        axs = plot.start_sims_metrics_plot(p.is_reentry_sim, sims_to_show)
        random_sim_to_show = np.random.choice(len(p.init_angles) * len(p.init_velocities), size=sims_to_show, replace=False)
    sim_number = 0
    for angle_0 in p.init_angles:
        for v_0 in p.init_velocities:
            
            angle_0_rad = np.radians(angle_0)
            vx = v_0 * np.cos(angle_0_rad)
            vy = v_0 * np.sin(angle_0_rad)

            # S b= [X, Y, VX, VY]
            S0 = np.array([p.x_0, p.altitude_0 + RADIUS_EARTH, vx, vy])
            # M = [V, A, ACC_HORIZ_DIST]
            M0 = np.array([v_0, 0, 0])  

            S, M, t = run_one_simulation(S0, M0, p, method_f, reentry_slope)
            passed_max_acceleration = False
            for i in range(0, len(M)):
                if(M[i][A] > p._max_acceleration):
                    acceleration_pairs.append((angle_0, v_0))
                    break
            # if np.any(M[A] > p.max_acceleration):
            #     acceleration_pairs.append((angle_0, v_0))
            if M[-1][V] > p._max_landing_velocity:
                velocity_pairs.append((angle_0, v_0))
            elif M[-1][ACC_HORIZ_DIST] < p._min_horizontal_distance:
                landed_before.append((angle_0, v_0))
            elif M[-1][ACC_HORIZ_DIST] > p._max_horizontal_distance:
                landed_after.append((angle_0, v_0))
            else:
                successful_pairs.append((angle_0, v_0))
            if p.show_details:
                if sim_number in random_sim_to_show:
                    sim_metrics = {
                        INIT_ANGLE: angle_0,
                        INIT_VELOCITY: v_0,
                        PATH_X: S[1:, X],
                        PATH_Y: S[1:, Y] - RADIUS_EARTH,
                        VELOCITIES: M[1:, V],
                        ACCELERATIONS: M[1:, A],
                        TIMES: t[1:]
                    }
                    plot.plot_sim_metrics(axs, sim_metrics, SIM_TO_RUN == REENTRY_SIM)
                sim_number += 1
    if p.show_details:
        plot.end_sims_metrics_plot()
    # if SIM_TO_RUN == REENTRY_SIM:
    #     plot.plot_reentry_parameters(successful_pairs)
    # plot.plot_reentry_parameters(successful_pairs)
    # plot.plot_all_reentrys(successful_pairs, acceleration_pairs, velocity_pairs, landed_before, landed_after)
    



