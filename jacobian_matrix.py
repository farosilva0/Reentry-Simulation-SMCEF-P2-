from sympy import symbols, Function, sqrt, Matrix

# variables used in the formulas
x, y, v, vx, vy, Cd, Cl, A, m = symbols('x y vx vy v Cd Cl A m')

# functions used in the formulas
rho = Function('rho')(y)  # Air density as a function of altitude
g = Function('g')(y)      # Gravity as a function of altitude

# Define the acceleration components
# ax = -(0.5 * Cd * rho * A * v * vx) / m
# ay = -(0.5 * Cd * rho * A * v * vy) / m + (0.5 * Cl * rho * A * v * vx) / m - g

dx = vx   
dy = vy
dvx = 0
dvy = 9.81

# Define the state variables
state_vars = [x, y, vx, vy]

# Compute the Jacobian matrix
jacobian_matrix = Matrix([dvx, dvy]).jacobian(state_vars)

print(jacobian_matrix)
