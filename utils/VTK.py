import vtk
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import re


attribute_name = 'nuTilda'
origin_points = None
new_attribute_name_list = []
vtk_size_bytes = 0
PointOrCell = None

def parse_and_read_vtk(opt):
    global PointOrCell, new_attribute_name_list, vtk_size_bytes
    data_path = opt.Path
    PointOrCell = opt.VTK.PointOrCell
    attribute_name_list = opt.VTK.attribute

    if PointOrCell not in ['point', 'cell']:
        raise ValueError("opt.VTK.PointOrCell must be either 'point' or 'cell'.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} does not exist.")

    aggregated_value_array = None
    if attribute_name_list == []:
        attribute_name_list = get_VTK_all_attributes(data_path, PointOrCell)

    for attribute_name in attribute_name_list:
        vtk_size_bytes = vtk_size_bytes + get_vtk_attribute_size_bytes(data_path, PointOrCell, attribute_name)

    for attribute_name in attribute_name_list:
        PointOrCell_array, PointOrCell_value_array = readVTK_on_attribute(data_path, PointOrCell, attribute_name)
        # 根据通道数更新属性名列表
        num_channels = PointOrCell_value_array.shape[1]
        if num_channels == 1:
            new_attribute_name_list.append(attribute_name)
        else:
            for i in range(0, num_channels):
                new_attribute_name_list.append(f"{attribute_name}_{i}")

        # 聚合数组
        if aggregated_value_array is None:
            aggregated_value_array = PointOrCell_value_array
        else:
            aggregated_value_array = np.hstack((aggregated_value_array, PointOrCell_value_array))

    return PointOrCell_array, aggregated_value_array

def get_new_attribute_name_list():
    global new_attribute_name_list
    return new_attribute_name_list

def get_vtk_size_bytes():
    global vtk_size_bytes
    return vtk_size_bytes

def sort_in_3D_axies(predict_points, predict_points_value):
    sorted_indices = np.lexsort((predict_points[:, 2], predict_points[:, 1], predict_points[:, 0]))
    predict_points_sorted = predict_points[sorted_indices]
    predict_points_value_sorted = predict_points_value[sorted_indices]
    return predict_points_sorted, predict_points_value_sorted

def restore_sorting(restore_indices, sorted_data):
    return sorted_data[restore_indices]

def get_vtk_attribute_size_bytes(file_path, PointOrCell, attribute_name):
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
    global new_attribute_name_list, PointOrCell
    # 读取现有的VTK文件
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_file_path)  # 设置输入文件名
    reader.Update()

    # 获取已有的VTK数据
    vtk_data = reader.GetOutput()

    if PointOrCell == 'point':
        # 获取点数据
        pointData = vtk_data.GetPointData()
    elif PointOrCell == 'cell':
        pointData = vtk_data.GetCellData()

    for j in range(len(new_attribute_name_list)):
        offset, base_str = extract_offset_and_base_string(new_attribute_name_list[j])  # 末尾没有_%d或者有但是_0 返回的是0
        # 获取或创建名为 'nuTilda' 的属性数组
        if base_str is None:
            nuTildaArray = pointData.GetArray(new_attribute_name_list[j])
        else:
            nuTildaArray = pointData.GetArray(base_str)

        if nuTildaArray is None:
            sys.exit('Attribute not exist')

        assert nuTildaArray.GetNumberOfTuples() == points_value.shape[0]
        # 获取恢复顺序的索引
        restore_indices = np.argsort(np.lexsort((origin_points[:, 2], origin_points[:, 1], origin_points[:, 0])))
        # 使用恢复顺序的索引来恢复原始顺序的数据
        restored_predict_points_value = restore_sorting(restore_indices, points_value)

        # 将新的属性值设置到 'nuTilda' 数组中
        for i in range(restored_predict_points_value.shape[0]):
            nuTildaArray.SetComponent(i, offset, restored_predict_points_value[i,offset])   # 设置第 i 个元素的第 j 个分量的值


    # 写入修改后的 VTK 文件
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file_path)  # 设置输出文件名
    writer.SetInputData(vtk_data)
    writer.Write()

def get_VTK_all_attributes(data_path, PointOrCell):
    # 创建并设置VTK读取器
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(data_path)
    reader.Update()

    # 获取VTK文件的输出
    unstructured_grid = reader.GetOutput()

    # 获取所有属性
    attributes_list = []
    if PointOrCell == 'point':
        point_data = unstructured_grid.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            attributes_list.append(point_data.GetArrayName(i))
    elif PointOrCell == 'cell':
        cell_data = unstructured_grid.GetCellData()
        for i in range(cell_data.GetNumberOfArrays()):
            attributes_list.append(cell_data.GetArrayName(i))

    return attributes_list

def readVTK_on_attribute(data_path, PointOrCell, attribute_name):
    global origin_points

    print('read VTK file')
    print(f'Option: {PointOrCell}.{attribute_name}')

    # 创建并设置 VTK 读取器
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(data_path)
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

    if PointOrCell == 'point':
        # 获取点的数量
        num_points = unstructured_grid.GetNumberOfPoints()
    elif PointOrCell == 'cell':
        num_points = unstructured_grid.GetNumberOfCells()
    points_list = []
    points_value_list = []

    # 遍历每个点
    for i in range(num_points):
        # 获取原始点坐标
        if PointOrCell == 'point':
            point = unstructured_grid.GetPoint(i)
        elif PointOrCell == 'cell':
            point = unstructured_grid.GetCell(i)

        points_list.append(point)

        value = attribute.GetTuple(i)
        points_value_list.append(value)

    points_array, points_value_array = np.array(points_list), np.array(points_value_list)
    origin_points = points_array
    points_array, points_value_array = sort_in_3D_axies(points_array, points_value_array)
    return points_array, points_value_array

def extract_offset_and_base_string(s):
    # 正则表达式匹配模式：任意字符序列，后跟下划线和一个或多个数字
    pattern = r'(.*)_(\d+)$'
    match = re.search(pattern, s)
    if match:
        base_string = match.group(1) # 去掉最后的 _%d 的字符串部分
        offset = int(match.group(2)) # 字符串末尾的数字部分（偏移量）
        return offset, base_string
    else:
        return 0, None # 如果没有匹配，则返回None作为偏移量，并原样返回字符串

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
