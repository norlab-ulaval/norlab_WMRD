import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure and axes
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Initial vertical lines
vline_x = [1, 2, 3]  # Initial x positions for the vertical lines
y_min = [0, 0, 0]
y_max = [10, 10, 10]

# Create vertical lines
vlines = ax.vlines(vline_x, y_min, y_max, colors='b')

# Function to update the vertical lines
def update(frame):
    # Move lines to the right; wrap around after reaching 10
    new_vline_x = [(x + 0.10*frame)  for x in vline_x]

    segment = [] 
    for line, new_x_pos in zip(vlines.get_segments(), new_vline_x):
        line[0][0] = new_x_pos  # Update start x position
        line[1][0] = new_x_pos  # Update end x position
        segment.append(line)

    vlines.set_segments(segment)
    return vlines,

# Create the animation with a specified interval (e.g., 100 ms)
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True, interval=100, repeat=True)

# Show the plot
plt.title('Animating Vertical Lines')
plt.show()
