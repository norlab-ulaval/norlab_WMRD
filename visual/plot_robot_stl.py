import numpy as np
import math
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

## Warthog dimensions
k1 = 0
k2 = 0.5826
k3 = 0.24979
k4 = 0.457367
k5 = 0
k6 = 0.012977

# Create a new plot
figure = plt.figure()
axes = mplot3d.Axes3D(figure)

# locomotion = 'wheels.stl'
locomotion = 'tracks.stl'

# Load the STL files and add the vectors to the plot
chassis_mesh = mesh.Mesh.from_file('../stl/warthog/chassis.stl')
left_drive_mesh = mesh.Mesh.from_file('../stl/warthog/fenders.stl')
right_drive_mesh = mesh.Mesh.from_file('../stl/warthog/fenders.stl')
front_left_wheel_mesh = mesh.Mesh.from_file('../stl/warthog/' + locomotion)
rear_left_wheel_mesh = mesh.Mesh.from_file('../stl/warthog/' + locomotion)
front_right_wheel_mesh = mesh.Mesh.from_file('../stl/warthog/' + locomotion)
rear_right_wheel_mesh = mesh.Mesh.from_file('../stl/warthog/' + locomotion)

# position and rotate all meshes to fit dimensions
chassis_mesh.z -= 0.27218
left_drive_mesh.rotate([0.0, 0.0, 0.5], math.radians(90))
left_drive_mesh.y += k2
left_drive_mesh.z -= k3
right_drive_mesh.rotate([0.0, 0.0, 0.5], math.radians(90))
right_drive_mesh.y -= k2
right_drive_mesh.z -= k3
front_left_wheel_mesh.rotate([0.5, 0.0, 0.0], math.radians(-90))
front_left_wheel_mesh.x += k4
front_left_wheel_mesh.y += k2
front_left_wheel_mesh.z -= (k3 - k6)
rear_left_wheel_mesh.rotate([0.5, 0.0, 0.0], math.radians(-90))
rear_left_wheel_mesh.rotate([0.0, 0.0, 0.5], math.radians(180))
rear_left_wheel_mesh.x -= k4
rear_left_wheel_mesh.y += k2
rear_left_wheel_mesh.z -= (k3 - k6)
front_right_wheel_mesh.rotate([0.5, 0.0, 0.0], math.radians(-90))
front_right_wheel_mesh.x += k4
front_right_wheel_mesh.y -= k2
front_right_wheel_mesh.z -= (k3 - k6)
rear_right_wheel_mesh.rotate([0.5, 0.0, 0.0], math.radians(-90))
rear_right_wheel_mesh.rotate([0.0, 0.0, 0.5], math.radians(180))
rear_right_wheel_mesh.x -= k4
rear_right_wheel_mesh.y -= k2
rear_right_wheel_mesh.z -= (k3 - k6)

# Add all meshes to plot
body_alpha = 0.5
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(chassis_mesh.vectors,
                                                     alpha=body_alpha, color = 'black', ec='None'))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(left_drive_mesh.vectors,
                                                     alpha=body_alpha, color = 'yellow', ec='None'))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(right_drive_mesh.vectors,
                                                     alpha=body_alpha, color = 'yellow', ec='None'))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(front_left_wheel_mesh.vectors,
                                                     alpha=body_alpha, color = 'black', ec='None'))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(rear_left_wheel_mesh.vectors,
                                                     alpha=body_alpha, color = 'black', ec='None'))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(front_right_wheel_mesh.vectors,
                                                     alpha=body_alpha, color = 'black', ec='None'))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(rear_right_wheel_mesh.vectors,
                                                     alpha=body_alpha, color = 'black', ec='None'))



# Auto scale to the mesh size
scale = chassis_mesh.points.flatten() * 1.5
print(scale)
axes.auto_scale_xyz(scale, scale, scale)

plt.show()