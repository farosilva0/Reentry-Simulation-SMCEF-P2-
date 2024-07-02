import numpy as np

# encontrar angulo dados 2 vetores:

vx = [100, 100,   1, -100]
vy = [100,   1, 100,  100]

for i in range(len(vx)):
    print(f'\nvx: {vx[i]}, vy: {vy[i]}')
    v_angle = np.arctan2(vy[i], vx[i])
    print("v_angle: ",np.degrees(v_angle))
    # encontrar angulo do lift, perpendicular à velocidade e sempre a apontar para cima: 
    l_angle = v_angle + np.pi/2
    if l_angle > np.pi:
        l_angle -= np.pi
    print("l_angle: ",np.degrees(l_angle))
