from utils.VTK import *
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

file_name = 'tetBox_5000.vtk' # 'DLR-internal.vtu' # 'tetBox_0.vtk'
file_path = fr'D:\MyDesktop\hp\Desktop\DDINR\data\{file_name}'
update_reader(file_path)
points_array, points_value_array = readVTK_on_attribute(file_path, 'point', 'p')

# 创建Open3D的点云对象
pcd = o3d.geometry.PointCloud()

# 将点坐标填充到点云对象中
pcd.points = o3d.utility.Vector3dVector(points_array.reshape(-1, 3))

# 创建颜色映射函数
def colormap(value, min_val, max_val):
    # 将值映射到归一化范围内
    value_normalized = (value - min_val) / (max_val - min_val)  # 将数据归一化到[0, 1]范围内

    # 创建热度图颜色映射
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.hot

    # 将值映射到热度图颜色空间
    rgba = cmap(norm(value_normalized))

    # 返回 RGB 颜色值
    return [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]


min_val = np.min(points_value_array)
max_val = np.max(points_value_array)
colors_array = np.array([colormap(value,min_val,max_val) for value in points_value_array.flatten()]).reshape(-1, 3)
# 创建一个新的字段并将值填充到点云对象中
# colors = np.tile(points_value_array, (1, 3))  # 生成颜色数组
pcd.colors = o3d.utility.Vector3dVector(colors_array)  # 将颜色数组直接传递给Open3D的Vector3dVector



# 保存为PCD文件
o3d.io.write_point_cloud(f'pcd/{file_name[:-4]}.pcd', pcd)
