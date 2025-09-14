import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def activation_func(x):
    return np.where(x > 0, x, 0)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_title('ReLU (x < 0)')
ax.grid(True)

# ax.set_ylim([-1, 5])
ax.set_ylim([-5, 5])
ax.set_xlim([-5, 5])


# Initial values for the input range
# l_init = 0.1
# u_init = 4
l_init = -4
u_init = +4
l = l_init
u = u_init

# Generate x-axis data for the activation function
x = np.linspace(-10, 10, 400)
y = activation_func(x)

# Plot the activation function curve
ax.plot(x, y, 'b-', label='ReLU')

# Plot the linear bounds
x_approx = np.linspace(l_init, u_init, 100)
# ax.plot(x_approx, activation_func(x_approx), 'r', label='Linear Lower=Upper Bound')
# ax.plot(x_approx, lower_y, 'g--', label='Linear Lower Bound')

# # Plot the convex hull by filling the area between the lines
# ax.fill_between(x_approx, lower_y, upper_y, color='orange', alpha=0.3, label='Convex Hull')

zero_base, = ax.plot(x_approx, [0]*len(x_approx), 'g--', label='Lower Bound (Zero)')
zero_upper, = ax.plot(x_approx, u*(x_approx-l)/(u-l), 'r--', label='Upper Bound (From Zero)')
linear, = ax.plot(x_approx, x_approx, 'r--', label='Upper Bound (From Zero)')




# Plot the input interval on the x-axis
ax.axvline(l_init, color='k', linestyle=':', label='Input Range [l, u]')
ax.axvline(u_init, color='k', linestyle=':')
# ax.fill_between(x_approx, ax.get_ylim()[0], ax.get_ylim()[1], color='gray', alpha=0.5, label='Input Interval')
ax.fill_between(x_approx, -0.1, 4.1, color='gray', alpha=0.25, label='Input Interval')
# ax.fill_between(x_approx, -0.1, +0.1, color='gray', alpha=0.5, label='Input Interval')
ax.legend(loc="upper left")

# Create sliders for l and u
# ax_l = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
# ax_u = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
# s_l = Slider(ax_l, 'Lower Bound (l)', -10, 10, valinit=l_init)
# s_u = Slider(ax_u, 'Upper Bound (u)', -10, 10, valinit=u_init)

plt.tight_layout()

plt.show()