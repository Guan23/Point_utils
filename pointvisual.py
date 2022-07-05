import numpy as np
import mayavi.mlab

lidar_path = "/home/cidi/datasets/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
lidar_bus = "/home/cidi/datasets/v1.0-mini/gt_database_10sweeps_withvelo/car/52_car_36.bin"

# lidar_path换成自己的.bin文件路径
pointcloud = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
print(pointcloud.shape)
print(pointcloud[:])

x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point

r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

degr = np.degrees(np.arctan(z / d))

vals = 'height'
if vals == "height":
    col = z
else:
    col = d

fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
mayavi.mlab.points3d(x, y, z,
                     col,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     scale_factor=0.2
                     )

mayavi.mlab.show()
