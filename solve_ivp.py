import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

from reentry_sim import *

# TODO: isto vai buscar o get_tot_acceleration() do reentry_sim
# mas temos de usar o g = CONSTANT - comentar a seguir porque dá infinitos... ver porquê depois

def solve_ivp_reentry_simulation(t, array):
    x, vx, y, vy = array
    ax, ay = get_air_acceleration(x, y, vx, vy)
    return [vx, 
            -np.sqrt(vx**2 + vy**2)*vx - ax * dt, 
            vy, 
            -1-np.sqrt(vx**2 + vy**2)*vy - ay * dt]


def main():

    angle_0 = np.degrees(-4)
    v_0 = 2 
    sol1 = solve_ivp(solve_ivp_reentry_simulation, [0, 60], 
                     y0=[0, 
                         v_0*np.cos(angle_0), 
                         0, 
                         v_0*np.sin(angle_0)], 
                    t_eval=np.linspace(0, 60, 1000))
    
    plt.plot(sol1.y[0], sol1.y[2])
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Reentry Simulation')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
