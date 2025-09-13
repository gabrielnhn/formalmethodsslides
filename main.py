import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the sigmoid and tanh functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivatives
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Corrected function to get linear approximation parameters
def get_linear_approx_params(l, u):
    g = sigmoid
    g_l = g(l)
    g_u = g(u)
    
    # Correct Upper Bound: Line connecting the endpoints (for a concave function)
    lambda_upper_conn = (g_u - g_l) / (u - l) if (u - l) != 0 else 0
    mu_upper_conn = g_l - lambda_upper_conn * l
    
    # Correct Lower Bound: The minimum slope is at the end points
    lambda_lower_prime = min(sigmoid_prime(l), sigmoid_prime(u))
    mu_lower_prime = g_l - lambda_lower_prime * l
    
    # Additional Upper Bound: Tangent line with max slope (at x=0)
    lambda_upper_tangent = sigmoid_prime(0) # max derivative is 0.25 at x=0
    mu_upper_tangent = g(0) - lambda_upper_tangent * 0
    
    return lambda_upper_conn, mu_upper_conn, lambda_lower_prime, mu_lower_prime, lambda_upper_tangent, mu_upper_tangent

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_title('Sigmoid + Linear + Derivative')
ax.grid(True)
ax.set_ylim([-0.2, 1.2])
ax.set_xlim([-5, 5])

# Initial values for the input range
l_init = -4.0
u_init = 4.0

# Generate x-axis data
x = np.linspace(-10, 10, 400)
y = sigmoid(x)

# Plot the sigmoid curve
line, = ax.plot(x, y, 'b-', label='Sigmoid Function')

# Get initial approximation parameters
lambda_uc, mu_uc, lambda_lp, mu_lp, lambda_ut, mu_ut = get_linear_approx_params(l_init, u_init)

# Plot the linear bounds
x_approx = np.linspace(l_init, u_init, 100)
line_upper_conn, = ax.plot(x_approx[:len(x_approx)//2], lambda_uc * x_approx[:len(x_approx)//2] + mu_uc, 'r--', label='Linear Approx')

line_upper_tangent, = ax.plot(x_approx[len(x_approx)//2:], lambda_ut * x_approx[len(x_approx)//2:] + mu_ut, 'm--', label='Derivative (x=0)')

line_lower_prime, = ax.plot(x_approx, [0]*len(x_approx), 'g--', label='Lower Bound (Zero)')

# line_lower_prime, = ax.plot(x_approx, lambda_lp * x_approx + mu_lp, 'g--', label='Lower Bound (Min Slope Tangent)')

# Plot the interval on the x-axis
# ax.axvline(l_init, color='k', linestyle=':', label='Input Range [l, u]')
# ax.axvline(u_init, color='k', linestyle=':')
# ax.fill_between(x_approx, -0.2, 1.2, color='gray', alpha=0.1, label='Input Interval')
ax.legend()

# Create sliders for l and u
# ax_l = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
# ax_u = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
# s_l = Slider(ax_l, 'Lower Bound (l)', -10, 10, valinit=l_init)
# s_u = Slider(ax_u, 'Upper Bound (u)', -10, 10, valinit=u_init)

def update(val):
    # l = s_l.val
    # u = s_u.val
    l = l_init
    u = u_init
    
    # if u <= l:
    #     s_u.set_val(l + 0.1)
    #     u = s_u.val
    
    lambda_uc, mu_uc, lambda_lp, mu_lp, lambda_ut, mu_ut = get_linear_approx_params(l, u)
    
    x_approx = np.linspace(l, u, 100)
    line_upper_conn.set_xdata(x_approx)
    line_upper_conn.set_ydata(lambda_uc * x_approx + mu_uc)
    line_lower_prime.set_xdata(x_approx)
    line_lower_prime.set_ydata(lambda_lp * x_approx + mu_lp)
    line_upper_tangent.set_xdata(x_approx)
    line_upper_tangent.set_ydata(lambda_ut * x_approx + mu_ut)
    
    ax.axvline(l, color='k', linestyle=':')
    ax.axvline(u, color='k', linestyle=':')
    
    for artist in ax.collections:
        artist.remove()

    ax.fill_between(x_approx, -0.2, 1.2, color='gray', alpha=0.1, label='Input Interval')
    ax.legend()
    fig.canvas.draw_idle()

# s_l.on_changed(update)
# s_u.on_changed(update)

plt.show()