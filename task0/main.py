import open3d as o3d
import numpy as np
from PIL import Image

extended_intrinsic_matrix = np.matrix([[481.20, 0, 319.50, 0],
                                       [0, -480.00, 239.50, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

extendet_intrinsic_matrix_inverted = np.linalg.inv(extended_intrinsic_matrix)

# converting a pixel to a pointcloud point
def uvd2xyz(u, v, d):
    z = d / 5000
    xyz1 = z * extendet_intrinsic_matrix_inverted @ np.array([u, v, 1, 1 / z]).transpose()
    return xyz1[0, 0], xyz1[0, 1], xyz1[0, 2]


depths_img = Image.open('data/depth.png')
colors_img = Image.open('data/annot.png')


points = [uvd2xyz(u, v, depths_img.getpixel((u, v)))
          for u in range(depths_img.size[0]) for v in range(depths_img.size[1])]
colors = [colors_img.getpixel((u, v))
          for u in range(colors_img.size[0]) for v in range(colors_img.size[1])]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

pcd.transform([[-1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]])
o3d.visualization.draw_geometries([pcd])
