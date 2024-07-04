import numpy as np
print("\n"*20)
# encontrar angulo dados 2 vetores:

y = [100,   100,  100,    100,     10,     0]
x = [-10,   0,    10,     100,   100,    100]

for i in range(len(x)):
    print(f'\nvx: {x[i]}, vy: {y[i]}')
    v_angle = np.radians(90) - np.arctan2(y[i], x[i])

    print("v_angle: ",np.degrees(v_angle))
    # encontrar angulo do lift, perpendicular à velocidade e sempre a apontar para cima: 
    l_angle = v_angle + np.pi/2
    if l_angle > np.pi:
        l_angle -= np.pi
    print("l_angle: ",np.degrees(l_angle))

