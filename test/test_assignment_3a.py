import numpy as np

# Define the function for dy/dt = t - y^2
def f(t, y):
    return t - y**2

# Euler Method Implementation
def euler_method(f, t0, y0, t_end, steps):
    t_values = np.linspace(t0, t_end, steps + 1)
    y_values = np.zeros(steps + 1)
    y_values[0] = y0
    
    dt = (t_end - t0) / steps  # Step size
    
    for i in range(steps):
        y_values[i + 1] = y_values[i] + dt * f(t_values[i], y_values[i])
    
    return t_values, y_values

# Runge-Kutta Method Implementation
def runge_kutta(f, t0, y0, t_end, steps):
    t_values = np.linspace(t0, t_end, steps + 1)
    y_values = np.zeros(steps + 1)
    y_values[0] = y0
    
    dt = (t_end - t0) / steps  # Step size
    
    for i in range(steps):
        t = t_values[i]
        y = y_values[i]
        
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, y + k1 / 2)
        k3 = dt * f(t + dt / 2, y + k2 / 2)
        k4 = dt * f(t + dt, y + k3)
        
        y_values[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return t_values, y_values

# Initial conditions and parameters
t0 = 0    # Initial time
y0 = 1    # Initial value of y
t_end = 2 # End time
steps = 10 # Number of steps/iterations

# Apply Euler Method
t_euler, y_euler = euler_method(f, t0, y0, t_end, steps)

# Apply Runge-Kutta Method
t_rk, y_rk = runge_kutta(f, t0, y0, t_end, steps)
