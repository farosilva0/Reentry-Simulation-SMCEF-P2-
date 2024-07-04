import math
import matplotlib.pyplot as plt


R = 6.371e6
y_step = -10 # ex gravidade
x_step = 10  # ex vel inicial ...

ground_x = []
altitude = []


x = 0
y = R + 100_000

x_ground = 0
y_altitude = y

theta = math.atan(x / y)
print("theta: ", theta)
x_flat = x_step

while y_altitude >= 0 and x_ground < 1_000_000: 
    x_flat += x_step

    ''' convertemos coordenadas flat earth para round earth, consoante o theta em que estamos'''
    # só movimento pra baixo: y diminui -> x mantem igual
    x += y_step * math.sin(theta) 
    y += y_step * math.cos(theta) 

    # só movimento prá esquerda: x diminui -> y mantem igual
    # x += x_step * math.cos(theta)
    # y += - x_step * math.sin(theta) 

    x_ground = R * theta
    y_altitude = math.sqrt(x**2 + y**2) - R
    print("x_flat: ", x_flat ,"x: ", x, "x: ",x_ground ," y: ", y, " theta: ", theta)
    # if y_altitude != y:
    #     print("y_altitude: ", y_altitude, " y: ", y)
    ground_x.append(x_ground)
    altitude.append(y_altitude)
    
    '''Atualizamos theta para a posição atual'''
    theta = math.atan(x / y)
    
print("R: ",R)
print("perimetro/4: ", R * math.pi / 2)

print("\nmin ground_x: ", min(ground_x))
print("max ground_x: ", max(ground_x))
print("error: ", max(ground_x) - min(ground_x))
print("\nmin altitude: ", min(altitude))
print("max altitude: ", max(altitude))
print("error: ", max(altitude) - min(altitude))

# Plotar trajetoria no mesmo plot
plt.plot(ground_x, altitude)
plt.xlabel("ground distance x")
plt.ylabel("altitude y above sea level")
plt.xlim(min(ground_x), max(ground_x))
plt.ylim(min(altitude), max(altitude))
plt.grid(True)
plt.show()




