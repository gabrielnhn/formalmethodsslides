import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the sigmoid and tanh functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Define the derivatives
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def get_paper_params(l, u, activation_function):
    g = activation_function
    if g == sigmoid:
        g_prime = sigmoid_prime
    elif g == tanh:
        g_prime = tanh_prime
    else:
        raise ValueError("Unsupported activation function")

    g_l = g(l)
    g_u = g(u)
    
    # Paper's definitions for lambda and lambda'
    lambda_conn = (g_u - g_l) / (u - l) if (u - l) != 0 else 0
    lambda_prime = min(g_prime(l), g_prime(u))
    
    # Conditional rules for the lower bound a_i'<= (x)
    if l >= 0:
        lambda_lower = lambda_conn
    else:
        lambda_lower = lambda_prime
    
    mu_lower = g_l - lambda_lower * l
        
    # Conditional rules for the upper bound a_i'>= (x)
    if u <= 0:
        lambda_upper = lambda_conn
    else:
        lambda_upper = lambda_prime
    
    mu_upper = g_u - lambda_upper * u
    
    x_approx = np.linspace(l, u, 100)
    lower_bound_y_values = lambda_lower * x_approx + mu_lower
    upper_bound_y_values = lambda_upper * x_approx + mu_upper
    
    return lower_bound_y_values, upper_bound_y_values
    
# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_title('Sigmoid Function with Two Linear Bounds (Conditional Logic)')
ax.grid(True)
ax.set_ylim([-0.2, 1.2])

# Choose activation function to plot
activation_func = sigmoid

# Initial values for the input range
l_init = -4.0
u_init = 4.0

# Generate x-axis data for the activation function
x = np.linspace(-10, 10, 400)
y = activation_func(x)

# Plot the activation function curve
ax.plot(x, y, 'b-', label='Sigmoid Function')

# Get initial approximation parameters
lower_y, upper_y = get_paper_params(l_init, u_init, activation_func)

# Plot the linear bounds
x_approx = np.linspace(l_init, u_init, 100)
ax.plot(x_approx, upper_y, 'r--', label='Linear Upper Bound')
ax.plot(x_approx, lower_y, 'g--', label='Linear Lower Bound')

# Plot the convex hull by filling the area between the lines
ax.fill_between(x_approx, lower_y, upper_y, color='orange', alpha=0.3, label='Convex Hull')

# Plot the input interval on the x-axis
ax.axvline(l_init, color='k', linestyle=':', label='Input Range [l, u]')
ax.axvline(u_init, color='k', linestyle=':')
ax.fill_between(x_approx, -0.2, 1.2, color='gray', alpha=0.1, label='Input Interval')
ax.legend()

# Create sliders for l and u
ax_l = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
ax_u = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
s_l = Slider(ax_l, 'Lower Bound (l)', -10, 10, valinit=l_init)
s_u = Slider(ax_u, 'Upper Bound (u)', -10, 10, valinit=u_init)

def update(val):
    l = s_l.val
    u = s_u.val
    
    if u <= l:
        s_u.set_val(l + 0.1)
        u = s_u.val
    
    ax.clear()
    ax.set_title('Sigmoid Function with Two Linear Bounds (Conditional Logic)')
    ax.grid(True)
    ax.set_ylim([-0.2, 1.2])
    
    ax.plot(x, y, 'b-', label='Sigmoid Function')

    lower_y_val, upper_y_val = get_paper_params(l, u, activation_func)
    
    x_approx = np.linspace(l, u, 100)
    ax.plot(x_approx, upper_y_val, 'r--', label='Linear Upper Bound')
    ax.plot(x_approx, lower_y_val, 'g--', label='Linear Lower Bound')
    
    ax.fill_between(x_approx, lower_y_val, upper_y_val, color='orange', alpha=0.3, label='Convex Hull')
    
    ax.axvline(l, color='k', linestyle=':')
    ax.axvline(u, color='k', linestyle=':')
    
    for artist in ax.collections:
        if artist.get_label() == 'Input Interval':
            artist.remove()

    ax.fill_between(x_approx, -0.2, 1.2, color='gray', alpha=0.1, label='Input Interval')
    ax.legend()
    fig.canvas.draw_idle()
    
s_l.on_changed(update)
s_u.on_changed(update)

plt.show()