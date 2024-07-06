
import numpy as np

from sim_common_fun import *


def forward_euler_step(Sk, Mk, p: Params, slope_f): 
    ''' Forward Euler step for the capsule reentry simulation. 
        Given previous state (Sk) and metrics (Mk), calculates the current state (Sk + 1) and metrics (Mk +1).'''
    slopes, Mk1 = slope_f(np.array(Sk), np.array(Mk), p)
    Sk1 = Sk + slopes * p._dt
    Mk1[V] = np.sqrt(Sk1[VX]**2 + Sk1[VY]**2)
    Mk1[ACC_HORIZ_DIST] = Mk[ACC_HORIZ_DIST] + (Sk1[X] - Sk[X]) / Sk1[Y]
    return Sk1, Mk1


if __name__ == '__main__':
    run_all_simulations(forward_euler_step)
