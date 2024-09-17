import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import KeyEvent
import numpy as np 

# Initialize data
data = {'x': [], 'y': []}

# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

# Set up the plot limits
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

# Initialize animation data
def init():
    line.set_data([], [])
    return line,

# Update the line data
def update(frame):
    data['x'].append(frame)
    data['y'].append(np.sin(frame))
    line.set_data(data['x'], data['y'])
    return line,

# Function to handle key presses
def on_key(event):
    if event.key == ' ':
        ani.event_source.stop()  # Stop the current animation
        ani.event_source.start()  # Restart the animation

# Connect the key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(10),
    init_func=init,
    blit=True,
    interval=1000  # Update interval in milliseconds
)

plt.show()