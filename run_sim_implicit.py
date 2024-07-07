
import numpy as np

from sim_common_fun import *


def get_residual_vector(Sk, Sk1, Mk, p: Params, slope_f):
    ''' returns the residual vector, given by the formula: 
        (Sk1-Sk)/dt - f(Sk1,tk1).'''
    slopes, Mk = slope_f(Sk1, Mk, p) # f(Sk1,tk1)
    # TODO: o slope devia ser calculado com (t+dt), não com t, mas na prática nem usamos o t no slope então como é???
    res_vector = (Sk1 - Sk)/p.dt - slopes
    return res_vector, Mk



def get_jacobian_matrix(Sk, p: Params):
    ''' returns the jacobian matrix.'''

    return np.array([[0, 0, 1, 0],  # dx/dt
                     [0, 0, 0, 1],  # dy/dt
                     [0, 0, 0, 0],  # dvx/dt
                     [0, 0, 0, -9.8]]) # dvy/dt
     


def newton_backward_euler_step(Sk, Mk, p: Params, slope_f):
    ''' Finds the roots of the system of equations using Newton's method:
        I/dt-J(f(v,tk1))∆v = - ((v-Sk)/dt - f(v,tk1)) '''
    
    max_iter = p.max_iter
    epsilon = p.epsilon

    Sk1 = np.array(Sk) # initial estimative for Sk1 value, which we set to Sk, and then we'll update in the loop
    Mk1 = np.array(Mk) 
    
    Idt = np.eye(Sk.shape[0]) / p.dt # identity matrix divided by dt, to form system matrix (I/dt - J) 
    
    for _ in range(max_iter):
        res_vector, Mk1 = get_residual_vector(Sk, Sk1, Mk, p, slope_f)
        if np.all(np.abs(Sk1) < epsilon):
           break
        system_matrix = Idt - get_jacobian_matrix(Sk, p) # left side of the equation: I/dt - J
        delta = np.linalg.solve(system_matrix, -Sk1)
        Sk1 += delta

    Mk1[V] = np.sqrt(Sk1[VX]**2 + Sk1[VY]**2)
    Mk1[ACC_EARTH_ANGLE] = Mk[ACC_EARTH_ANGLE] + (Sk1[X] - Sk[X]) / Sk1[Y]
    print("Sk:", Sk, "\nSk1:", Sk1, "\nMk:", Mk, "\nMk1:", Mk1)
    print("Newton method finished")
    return Sk1, Mk1



if __name__ == '__main__':
    run_all_simulations(newton_backward_euler_step)
