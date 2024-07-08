
import numpy as np

from sim_common_fun import *


def get_residual_vector(Sk, Sk1, Mk, p: Params, slope_f):
    ''' returns the residual vector, given by the formula: 
        (Sk1-Sk)/dt - f(Sk1,tk1).'''
    slopes, Mk = slope_f(Sk1, Mk, p) 
    res_vector = (Sk1 - Sk)/p.dt - slopes
    return res_vector, Mk



def get_jacobian_matrix(Sk, Mk, p: Params):
    ''' returns the jacobian matrix.'''
    x, y, vx, vy = Sk
    v = Mk[V]

    # jacobian matrix for system of 4 equations with 4 variables
    J = np.zeros((4, 4))

    # vx partial derivatives 
    J[0][2] = 1
    
    # vy partial derivatives
    J[1][3] = 1


    # air density
    rho = air_dens_f(y - RADIUS_EARTH)
    d_rho = air_dens_f.derivative()(y - RADIUS_EARTH)

    # air drag and lift constants
    drag_const = -0.5 * p.capsule_surface_area * p.capsule_drag_coefficient / p.capsule_mass
    lift_const =  0.5 * p.capsule_surface_area * p.capsule_lift_coefficient / p.capsule_mass

    #ax partial derivatives
    J[2][1] = (drag_const * d_rho * vx * v)
    J[2][2] = (drag_const * rho * (2 * vx**2 + vy**2)) / v
    J[2][3] = (drag_const * rho * vx * vy) / v

    # ay partial derivatives
    J[3][1] = (drag_const * d_rho * vy * v)                 + (lift_const * d_rho * (vx**2 + vy**2))   +   (2 * G_M * p.capsule_mass) / (y)**3
    J[3][2] = (drag_const * rho * vx * vy / v)              + (2 * lift_const * rho * vx)
    J[3][3] = (drag_const * rho * (2 * vy**2 + vx**2) / v)  + (2 * lift_const * rho * vy)

    return J



     

def newton_backward_euler_step(Sk, Mk, p: Params, slope_f):
    ''' Finds the roots of the system of equations using Newton's method:
        I/dt-J(f(v,tk1))∆v = - ((v-Sk)/dt - f(v,tk1)) '''
    
    max_iter = p.max_iter
    epsilon = p.epsilon

    Sk1 = np.array(Sk) # initial estimative for Sk1 value, which we set to Sk, and then we'll update in the loop
    Mk1 = np.array(Mk) 
    
    Idt = np.eye(Sk.shape[0]) / p.dt # identity matrix divided by dt, to form system matrix (I/dt - J) 
    
    for i in range(max_iter):
        res_vector, Mk1 = get_residual_vector(Sk, Sk1, Mk, p, slope_f)
        if np.linalg.norm(res_vector) < epsilon:  # Verifica a convergência com o vetor residual
        # if np.all(np.abs(Sk1) < epsilon): #TODO: ver que método é mais rapido
           break
        system_matrix = Idt - get_jacobian_matrix(Sk, Mk, p) # left side of the equation: I/dt - J
        delta = np.linalg.solve(system_matrix, -res_vector)
        Sk1 += delta

    Mk1[V] = np.sqrt(Sk1[VX]**2 + Sk1[VY]**2)
    Mk1[EARTH_ANGLE] = Mk[EARTH_ANGLE] + (Sk1[X] - Sk[X]) / Sk1[Y]
    return Sk1, Mk1



if __name__ == '__main__':
    run_all_simulations(newton_backward_euler_step)
