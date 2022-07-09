import open3d as o3d
import numpy as np
from PIL import Image


# Dataset and camera parameters.
SCALE_FACTOR = 5000

fx = 481.20
fy = -480.00
cx = 319.50
cy = 239.50


# 1. Creating the vector of colors.
colors_image = Image.open('data/annot.png')
colors_matrix = np.array(colors_image) / 256
colors_vector = colors_matrix.reshape((307200, 3))


# 2. Creating the vector of xyz-coordinates.
# 2.0. Getting the depth matrix from the image.
depths_image = Image.open('data/depth.png')
depths_matrix = np.array(depths_image)

# 2.1. Creating the vector of points where each element is (u_coordinate, v_coordinate, depth)
M, N = depths_matrix.shape
v_index, u_index = np.ix_(np.arange(M), np.arange(N))

uvd_points_matrix = np.zeros(colors_matrix.shape)
uvd_points_matrix[:, :, 0] = u_index
uvd_points_matrix[:, :, 1] = v_index
uvd_points_matrix[:, :, 2] = depths_matrix
uvd_points_vector = uvd_points_matrix.reshape((uvd_points_matrix.shape[0] * uvd_points_matrix.shape[1], uvd_points_matrix.shape[2]))

# 2.2. Separating the uvd-coordinates and calculating xyz-coordinates.
u_coordinates = uvd_points_vector[:, 0]
v_coordinates = uvd_points_vector[:, 1]
depths = uvd_points_vector[:, 2]

z_coordinates = depths / SCALE_FACTOR
x_coordinates = (u_coordinates - cx) * z_coordinates / fx
y_coordinates = (v_coordinates - cy) * z_coordinates / fy

xyz_points_vector = np.column_stack((x_coordinates, y_coordinates, z_coordinates))


# 3. Visualizing the point cloud.
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_points_vector)
pcd.colors = o3d.utility.Vector3dVector(colors_vector)


pcd.transform([[-1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]])
o3d.visualization.draw_geometries([pcd])
