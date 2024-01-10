import vtk
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'tetBox_0.vtk'
PointOrCell = 'point'
attribute_name = 'nuTilda'
Size = 128  # eg:256   -> 256*256*256

def readVTK():
    print('read VTK file')
    print(f'Seted option is: {PointOrCell}.{attribute_name}')
    # 创建并设置 VTK 读取器
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    unstructured_grid = reader.GetOutput()
    if PointOrCell == 'point':
        if unstructured_grid.GetPointData().HasArray(attribute_name):
            attribute = unstructured_grid.GetPointData().GetArray(attribute_name)
            attribute_dimension = attribute.GetNumberOfComponents()
            print(f"The dimension of the attribute '{attribute_name}' is: {attribute_dimension}")
        else:
            sys.exit(f"Attribute '{attribute_name}' not found in the file.")
    elif PointOrCell == 'cell':
        if unstructured_grid.GetCellData().HasArray(attribute_name):
            attribute = unstructured_grid.GetCellData().GetArray(attribute_name)
            attribute_dimension = attribute.GetNumberOfComponents()
            print(f"The dimension of the attribute '{attribute_name}' is: {attribute_dimension}")
        else:
            sys.exit(f"Attribute '{attribute_name}' not found in the file.")
    else:
        sys.exit(f"Wrong opt: '{PointOrCell}'")


    channel = attribute_dimension

    # 获取点的数量
    num_points = unstructured_grid.GetNumberOfPoints()

    # 获取原始坐标的最大和最小值，用于归一化
    min_max = [unstructured_grid.GetBounds()[i] for i in range(6)] # 这个方法返回一个包含六个值的元组，这些值分别表示网格在每个坐标轴（X、Y、Z）上的最小和最大坐标值。
    min_point = np.array(min_max[::2])
    max_point = np.array(min_max[1::2])
    print(f'min_point: {min_point}')
    print(f'max_point: {max_point}')

    # 创建一个 Size * Size * Sizex * channel
    coord_space = np.zeros((Size, Size, Size, channel))
    count_space = np.zeros((Size, Size, Size))

    # 遍历每个点
    for i in range(num_points):
        # 获取原始点坐标
        point = unstructured_grid.GetPoint(i)

        # 归一化和转换坐标
        normalized_point = (np.array(point) - min_point) / (max_point - min_point)
        transformed_point = np.round(normalized_point * (Size - 1)).astype(int)

        # 更新
        x, y, z = transformed_point
        count_space[x, y, z] += 1
        for j in range(channel):
            coord_space[x, y, z, j] += attribute.GetTuple(i)[j]

    # 计算均值
    for x in range(Size):
        for y in range(Size):
            for z in range(Size):
                if count_space[x, y, z] > 0:
                    coord_space[x, y, z, :] /= count_space[x, y, z]


    print(coord_space.shape)
    # show3D(coord_space)

def show3D(coord_space, channel = 0):
    # 创建一个 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 准备数据
    x, y, z = np.indices(coord_space.shape[:3])
    c = coord_space[:, :, :, channel].flatten()  # 获取第四维度的值并扁平化

    # 绘制散点图
    sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=c, s = 0.01, cmap='viridis')  # 调整s来调整图像中节点的大小

    # 添加颜色条
    plt.colorbar(sc)

    # 设置图形的标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # 显示图形
    plt.show()
