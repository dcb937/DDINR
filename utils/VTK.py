import vtk
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


PointOrCell = 'point'
attribute_name = 'nuTilda'
origin_points = None

def parse_opt_vtk():

    pass

def sort_in_3D_axies(predict_points, predict_points_value):
    sorted_indices = np.lexsort((predict_points[:, 2], predict_points[:, 1], predict_points[:, 0]))
    predict_points_sorted = predict_points[sorted_indices]
    predict_points_value_sorted = predict_points_value[sorted_indices]
    return predict_points_sorted, predict_points_value_sorted

def restore_sorting(restore_indices, sorted_data):
    return sorted_data[restore_indices]

def get_vtk_size(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    unstructured_grid = reader.GetOutput()
    size = 0
    if attribute_name:        # 指明了属性名，即只看那个属性的大小
        if PointOrCell == 'point':
            if unstructured_grid.GetPointData().HasArray(attribute_name):
                attribute = unstructured_grid.GetPointData().GetArray(attribute_name)
                attribute_dimension = attribute.GetNumberOfComponents()
                print(f"The dimension of the attribute '{attribute_name}' is: {attribute_dimension}")
            else:
                sys.exit(f"Attribute '{attribute_name}' not found in the PointData.")
        elif PointOrCell == 'cell':
            if unstructured_grid.GetCellData().HasArray(attribute_name):
                attribute = unstructured_grid.GetCellData().GetArray(attribute_name)
                attribute_dimension = attribute.GetNumberOfComponents()
                print(f"The dimension of the attribute '{attribute_name}' is: {attribute_dimension}")
            else:
                sys.exit(f"Attribute '{attribute_name}' not found in the CellData.")
        else:
            sys.exit(f"Wrong opt: '{PointOrCell}'")

        num = attribute.GetNumberOfTuples() * attribute.GetNumberOfComponents()
    else:               # 未指明了属性名，即算所有属性的大小
        sys.exit('not finished yet')

    return num*4    # 默认每个是用float32存储的

def save_vtk(input_file_path, output_file_path, points_value):
    # 读取现有的VTK文件
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_file_path)  # 设置输入文件名
    reader.Update()

    # 获取已有的VTK数据
    vtk_data = reader.GetOutput()

    # 获取点数据
    pointData = vtk_data.GetPointData()

    # 获取或创建名为 'nuTilda' 的属性数组
    nuTildaArray = pointData.GetArray('nuTilda')
    if nuTildaArray is None:
        sys.exit('Attribute not exist')
        # 如果 'nuTilda' 属性不存在，创建一个新的属性数组
        nuTildaArray = vtk.vtkDoubleArray()
        nuTildaArray.SetName("nuTilda")  # 设置属性名称为 'nuTilda'
        pointData.AddArray(nuTildaArray)

    assert nuTildaArray.GetNumberOfTuples() == points_value.shape[0]
    # 获取恢复顺序的索引
    restore_indices = np.argsort(np.lexsort((origin_points[:, 2], origin_points[:, 1], origin_points[:, 0])))
    # 使用恢复顺序的索引来恢复原始顺序的数据
    restored_predict_points_value = restore_sorting(restore_indices, points_value)

    # 将新的属性值设置到 'nuTilda' 数组中
    for i in range(restored_predict_points_value.shape[0]):
        nuTildaArray.SetValue(i, restored_predict_points_value[i])

    # 写入修改后的 VTK 文件
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file_path)  # 设置输出文件名
    writer.SetInputData(vtk_data)
    writer.Write()


def readVTK(file_path):
    global origin_points

    print('read VTK file')
    print(f'Set option: {PointOrCell}.{attribute_name}')

    # 创建并设置 VTK 读取器
    reader = vtk.vtkUnstructuredGridReader()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    reader.SetFileName(file_path)
    reader.Update()

    unstructured_grid = reader.GetOutput()
    if PointOrCell == 'point':
        if unstructured_grid.GetPointData().HasArray(attribute_name):
            attribute = unstructured_grid.GetPointData().GetArray(attribute_name)
            attribute_dimension = attribute.GetNumberOfComponents()
            print(f"The dimension of the attribute '{attribute_name}' is: {attribute_dimension}")
        else:
            sys.exit(f"Attribute '{attribute_name}' not found in the PointData.")
    elif PointOrCell == 'cell':
        if unstructured_grid.GetCellData().HasArray(attribute_name):
            attribute = unstructured_grid.GetCellData().GetArray(attribute_name)
            attribute_dimension = attribute.GetNumberOfComponents()
            print(f"The dimension of the attribute '{attribute_name}' is: {attribute_dimension}")
        else:
            sys.exit(f"Attribute '{attribute_name}' not found in the CellData.")
    else:
        sys.exit(f"Wrong opt: '{PointOrCell}'")


    channel = attribute_dimension

    # 获取点的数量
    num_points = unstructured_grid.GetNumberOfPoints()
    points_list = []
    points_value_list = []

    # 遍历每个点
    for i in range(num_points):
        # 获取原始点坐标
        point = unstructured_grid.GetPoint(i)

        points_list.append(point)

        value = attribute.GetTuple(i)
        points_value_list.append(value)

    points_array, points_value_array = np.array(points_list), np.array(points_value_list)
    origin_points = points_array
    points_array, points_value_array = sort_in_3D_axies(points_array, points_value_array)
    return points_array, points_value_array

def show3D(coord_space, channel = 0, path=os.getcwd()):
    # 创建一个 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 准备数据
    x, y, z = np.indices(coord_space.shape[:3])
    c = coord_space[:, :, :, channel]  # 获取第四维度的值并扁平化
    x, y, z, c = x.flatten(), y.flatten(), z.flatten(), c.flatten()
    # 创建掩码，忽略值小于 0 的点
    mask = c >= 0.0
    # 应用掩码
    x, y, z, c = x[mask], y[mask], z[mask], c[mask]

    # 绘制散点图
    sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=c, s = 0.1, cmap='viridis')  # 调整s来调整图像中节点的大小

    # 添加颜色条
    plt.colorbar(sc)

    # 设置图形的标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # 显示图形
    # plt.show()
    print(f'saved in {path}')
    filename = os.path.join(path, "fig.png")
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    file_path = 'D:\MyDesktop\hp\Desktop\DDINR\data\\tetBox_0.vtk'
    readVTK(file_path)