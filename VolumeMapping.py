import vtk

# 读取非结构化网格数据
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("data/tetBox_5000.vtk")
reader.Update()

# 获取非结构化网格数据
unstructured_grid = reader.GetOutput()
unstructured_grid.GetPointData().SetActiveScalars('nuTilda')

# 创建渲染器和渲染窗口
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# 创建交互器
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 创建VolumeMapper
volume_mapper = vtk.vtkProjectedTetrahedraMapper()
volume_mapper.SetInputData(unstructured_grid)

# 创建VolumeProperty
volume_property = vtk.vtkVolumeProperty()
volume_property.ShadeOff()
# 自定义颜色传递函数
color_transfer_function = vtk.vtkColorTransferFunction()
# 添加颜色节点
color_transfer_function.AddRGBPoint(0.0, 0.23137254902000001, 0.298039215686, 0.75294117647100001)
color_transfer_function.AddRGBPoint(0.5 * 0.38, 0.86499999999999999, 0.86499999999999999, 0.86499999999999999)
color_transfer_function.AddRGBPoint(1.0 * 0.38, 0.70588235294099999, 0.015686274509800001, 0.149019607843)
# color_transfer_function.AddRGBPoint(0.183, 0.0, 0.0, 0.0)  # 数据值为0时的颜色为红色
# color_transfer_function.AddRGBPoint(0.38, 1.0, 0.0, 1.0)  # 数据值为255时的颜色为蓝色

volume_property.SetColor(color_transfer_function)

# 创建一个不透明度传输函数
opacity_transfer_function = vtk.vtkPiecewiseFunction()
# 添加不透明度节点
opacity_transfer_function.AddPoint(0.18, 0.0)  # 数据值为0时的不透明度为0
opacity_transfer_function.AddPoint(0.187, 0.567)  # 数据值为0时的不透明度为0
opacity_transfer_function.AddPoint(0.38, 1.0)  # 数据值为255时的不透明度为1
volume_property.SetScalarOpacity(opacity_transfer_function)

volume_property.SetInterpolationTypeToLinear()

# 创建Volume
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# 添加Volume到渲染器
renderer.AddVolume(volume)

# 设置渲染器背景颜色
renderer.SetBackground(55.0, 55.0, 55.0)

# # 设置相机位置和方向
# renderer.GetActiveCamera().SetPosition(0, 0, 5)
# renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
# renderer.GetActiveCamera().SetViewUp(0, 1, 0)

# 启动交互器
interactor.Initialize()
render_window.Render()
interactor.Start()
