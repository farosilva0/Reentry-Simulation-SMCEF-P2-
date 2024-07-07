import numpy as np
import matplotlib.pyplot as plt

print("\n"*20)
# encontrar angulo dados 2 vetores:

# y = [100,   100,  100,    100,     10,     0]
# x = [-10,   0,    10,     100,   100,    100]

# for i in range(len(x)):
#     print(f'\nvx: {x[i]}, vy: {y[i]}')
#     v_angle = np.radians(90) - np.arctan2(y[i], x[i])

#     print("v_angle: ",np.degrees(v_angle))
#     # encontrar angulo do lift, perpendicular à velocidade e sempre a apontar para cima: 
#     l_angle = v_angle + np.pi/2
#     if l_angle > np.pi:
#         l_angle -= np.pi
#     print("l_angle: ",np.degrees(l_angle))

# Example data setup with simulated projectile motion
num_points = 1000

# Initial conditions
x0 = 0  # Initial horizontal position
y0 = 0  # Initial vertical position
v0 = 50  # Initial velocity magnitude
angle = np.radians(45)  # Launch angle (45 degrees)
vx0 = v0 * np.cos(angle)  # Initial horizontal velocity
vy0 = v0 * np.sin(angle)  # Initial vertical velocity

# Time step and time array
dt = 0.01  # Time step
t = np.arange(0, num_points * dt, dt)  # Time array

# Simulation of projectile motion
x = x0 + vx0 * t  # Horizontal position
y = y0 + vy0 * t - 0.5 * 9.81 * t**2  # Vertical position, with gravity

# Create state array S
S = np.column_stack((x, y, vx0 * np.ones_like(t), vy0 - 9.81 * t))

crossing_index = np.argmax(y < 0)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(S[:, 0], S[:, 1], label='Projectile Trajectory')
plt.scatter(S[crossing_index, 0], S[crossing_index, 1], color='red', marker='o', label='Crossing Point (y = 0)')

plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Altitude (m)')
plt.title('Projectile Motion Simulation')
plt.grid(True)
plt.legend()
plt.show()

