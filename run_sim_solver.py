import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sim_common_fun import *

params = None
acceleration_f = None

def solve_ivp_sim(t, Y0):
    ''' function that defines the system of ODEs to be solved by the solver_ivp function.
        args: Y0 = initial state vector [x, y, vx, vy] '''
    x, y, vx, vy, = Y0
    v = np.sqrt(vx**2 + vy**2)
    Mk = np.array([v, 0, 0, 0])  # [v, a, acc_horiz_dist]
    ax, ay, Mk = acceleration_f(Y0, Mk, params)
    return [vx, vy, ax, ay]

def event_conditions(t, Yk):
    ''' function that defines the stop condition for the solver_ivp function.
        args: Y = state vector [x, y, vx, vy]. 
        return 0 when we should stop the simulation. '''
    return Yk[Y] - RADIUS_EARTH


def solver_ivp (Sk, Mk, p: Params, get_acceleration_f):
    global params, acceleration_f
    params = p
    acceleration_f = get_acceleration_f
    
    size = int(p.sim_max_time / p.dt + 1)
    t = np.array([i * p.dt for i in range(size)])

    # Set stop condition for the solver_ivp function
    event_conditions.terminal = True  # Stop the integration when this event occurs
    event_conditions.direction = -1   # Look for a decreasing y position

    sol = solve_ivp(
                solve_ivp_sim,                      # function that defines the system of ODEs
                [0, size],                          # time span
                y0=[Sk[X], Sk[Y], Sk[VX], Sk[VY]],  # initial state vector 
                t_eval=t,                           # time points to evaluate the solution
                events=event_conditions,            # stop condition
                dense_output=True)  
    t = sol.t
    S = sol.y.T
    M = np.zeros((len(t), Mk.shape[0]), dtype=float)
    # TODO: fazer V e ACC vectorized pra ser mais eficiente com numpy do que nos loops 
    M[:, V] = np.sqrt(S[:, VX]**2 + S[:, VY]**2)
    # TODO: vectorize this (it would be better to store the acceleration during the simulation, but it's complex with the solver_ivp)
    for i in range(len(t)):
        ax, ay, Mk = get_acceleration_f(S[i], M[i], p)
        M[i, A] = Mk[A]
    for i in range(1, len(t)):
        M[i, ACC_EARTH_ANGLE] = M[i-1, ACC_EARTH_ANGLE] + (S[i, X] - S[i-1, X]) / S[i, Y]

    return S, M, t


if __name__ == "__main__":
    run_all_simulations(solver_ivp, True)
