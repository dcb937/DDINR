import sys
import vtk
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QStackedLayout, \
    QComboBox, QSlider, QFileDialog
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import pyqtgraph as pg
from utils.TransferFunctionEditor import Plot, TransferFunctionEditor

project_folder_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_folder_path, "data")
savedmodels_folder = os.path.join(project_folder_path, "outputs")
tf_folder = os.path.join(project_folder_path, "colormaps")

reader = None
unstructured_grid = None
last_read_vtk_file_path = None   # 为了避免重复加载同一个vtk
attribute_list = None

def update_reader(file_path, PointOrCell):
    global reader, unstructured_grid, last_read_vtk_file_path, attribute_list
    if file_path == last_read_vtk_file_path:
        return
    last_read_vtk_file_path = file_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    print('Update reader...')
    reader = vtk_or_vtu_reader(file_path)
    reader.SetFileName(file_path)
    reader.Update()
    unstructured_grid = reader.GetOutput()
    attribute_list = get_VTK_all_attributes(file_path, PointOrCell)

def vtk_or_vtu_reader(file_path):
    if file_path[-4:] == '.vtk':
        return vtk.vtkUnstructuredGridReader()
    elif file_path[-4:] == '.vtu':
        return vtk.vtkXMLUnstructuredGridReader()
    else:
        sys.exit(f'Unrecognized file type: {file_path[-4:]}')

def get_VTK_all_attributes(data_path, PointOrCell):
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
    else:
        sys.exit(f'Unrecognized: {PointOrCell}, must be `point` or `cell`')

    return attributes_list





class MainWindow(QtWidgets.QMainWindow):
    loading_model = False

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("VTK Renderer")
        self.resize(2000,1000)  # 必须得有。。。。
        # self.showMaximized()  # 最大化窗口
        self.frame = QtWidgets.QFrame()

        # Find all available models/colormaps
        self.available_models = os.listdir(savedmodels_folder)
        self.available_tfs = os.listdir(tf_folder)
        self.available_data = os.listdir(data_folder)

        self.PointOrCell = 'point'
        self.selected_attribute = ''
        self.selected_colormap = 'default'
        self.selected_SurfaceOrEdge = 'Edge'

        # Full screen layout
        layout = QHBoxLayout()
        # Render area
        self.render_view = QLabel()
        # Settings area
        self.settings_ui = QVBoxLayout()

        self.load_box = QHBoxLayout()
        self.load_box.addWidget(QLabel("Load from: "))
        self.load_from_dropdown = QComboBox()
        self.load_from_dropdown.addItems(["Data", "Model"])
        self.load_from_dropdown.currentTextChanged.connect(self.loadFrom_update)
        self.load_box.addWidget(self.load_from_dropdown)

        self.datamodel_box = QHBoxLayout()
        self.datamodel_box.addWidget(QLabel("Model/data: "))
        self.datamodel_dropdown = self.load_datamodel_dropdown()
        self.datamodel_dropdown.currentTextChanged.connect(self.datamodel_update)
        self.datamodel_box.addWidget(self.datamodel_dropdown)

        self.colormap_box = QHBoxLayout()
        self.colormap_box.addWidget(QLabel("Colormap: "))
        self.colormap_dropdown = self.load_colormap_dropdown()
        self.colormap_dropdown.currentTextChanged.connect(self.colormap_update)
        self.colormap_box.addWidget(self.colormap_dropdown)
        self.colormap_dropdown.setEnabled(False)

        self.PointOrCell_box = QHBoxLayout()
        self.PointOrCell_box.addWidget(QLabel("Mode: "))
        self.PointOrCell_dropdown = QComboBox()
        self.PointOrCell_dropdown.addItems(["point", "cell", "Volume"])
        self.PointOrCell_dropdown.currentTextChanged.connect(self.PointOrCell_update)
        self.PointOrCell_box.addWidget(self.PointOrCell_dropdown)

        self.SurfaceOrEdge_box = QHBoxLayout()
        self.SurfaceOrEdge_box.addWidget(QLabel("Surface or Edge: "))
        self.SurfaceOrEdge_dropdown = QComboBox()
        self.SurfaceOrEdge_dropdown.addItems(["Edge", "Surface"])
        self.SurfaceOrEdge_dropdown.currentTextChanged.connect(self.SurfaceOrEdge_update)
        self.SurfaceOrEdge_box.addWidget(self.SurfaceOrEdge_dropdown)
        self.SurfaceOrEdge_dropdown.setEnabled(self.PointOrCell == 'cell')

        self.attribute_box = QHBoxLayout()
        self.attribute_box.addWidget(QLabel("Attribute: "))
        self.attribute_dropdown = self.load_attribute_dropdown()
        self.attribute_dropdown.currentTextChanged.connect(self.attribute_update)
        self.attribute_box.addWidget(self.attribute_dropdown)

        self.button_box = QHBoxLayout()
        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self.start_rendering)  # 连接按钮点击事件到槽函数
        self.button_box.addWidget(self.render_button)  # 将按钮添加到设置面板的布局中

        # self.plot_box = QHBoxLayout()
        # self.plot_widget = TransferFunctionEditor(self)
        # self.plot_box.addWidget(self.plot_widget)
        self.transfer_function_box = QHBoxLayout()
        self.tf_editor = TransferFunctionEditor(self)
        x = np.linspace(0.0, 1.0, 4)
        pos = np.column_stack((x, x))
        win = pg.GraphicsLayoutWidget()
        view = win.addViewBox(row=0, col=1, rowspan=2, colspan=2)
        view.enableAutoRange(axis='xy', enable=False)
        view.setYRange(0, 1.0, padding=0.1, update=True)
        view.setXRange(0, 1.0, padding=0.1, update=True)
        view.setBackgroundColor([255, 255, 255, 255])
        view.setMouseEnabled(x=False, y=False)
        x_axis = pg.AxisItem("bottom", linkView=view)
        y_axis = pg.AxisItem("left", linkView=view)
        win.addItem(x_axis, row=2, col=1, colspan=2)
        win.addItem(y_axis, row=0, col=0, rowspan=2)
        view.addItem(self.tf_editor)
        self.transfer_function_box.addWidget(win)

        self.saveImg_button_box = QHBoxLayout()
        self.saveImg_button = QPushButton("Save Image")
        self.saveImg_button.clicked.connect(self.Save_Image)  # 连接按钮点击事件到槽函数
        self.saveImg_button_box.addWidget(self.saveImg_button)  # 将按钮添加到设置面板的布局中

        # VTK Renderer
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.settings_ui.addLayout(self.load_box)
        self.settings_ui.addLayout(self.datamodel_box)
        self.settings_ui.addLayout(self.PointOrCell_box)
        self.settings_ui.addLayout(self.colormap_box)
        self.settings_ui.addLayout(self.SurfaceOrEdge_box)
        # self.settings_ui.addLayout(self.button_box)
        self.settings_ui.addLayout(self.attribute_box)
        self.settings_ui.addLayout(self.transfer_function_box)
        # self.settings_ui.addLayout(self.saveImg_button_box)
        self.settings_ui.addStretch()
        self.settings_ui.insertLayout(layout.count() - 1, self.saveImg_button_box)

        # UI full layout
        layout.addWidget(self.vtkWidget, stretch=4)
        layout.addLayout(self.settings_ui, stretch=1)
        layout.setContentsMargins(0, 0, 10, 10)
        layout.setSpacing(20)
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        self.setCentralWidget(self.centralWidget)

        self.show()

    # def resizeEvent(self, event):
    #     w = self.vtkWidget.frameGeometry().width()
    #     h = self.vtkWidget.frameGeometry().height()
    #     self.vtkWidget.resize.emit(w, h)
    #     QMainWindow.resizeEvent(self, event)

    def loadFrom_update(self, s):
        self.loading_model = "Model" in s
        if(self.loading_model):
            self.datamodel_dropdown.currentTextChanged.disconnect(self.datamodel_update)
            self.datamodel_dropdown.clear()
            self.datamodel_dropdown.addItems(self.available_models)
            self.datamodel_dropdown.currentTextChanged.connect(self.datamodel_update)
        else:
            self.datamodel_dropdown.currentTextChanged.disconnect(self.datamodel_update)
            self.datamodel_dropdown.clear()
            self.datamodel_dropdown.addItems(self.available_data)
            self.datamodel_dropdown.currentTextChanged.connect(self.datamodel_update)

    def PointOrCell_update(self, s):
        self.PointOrCell = s
        self.SurfaceOrEdge_dropdown.setEnabled(self.PointOrCell == 'cell')
        self.colormap_dropdown.setEnabled(self.PointOrCell == 'Volume')
        self.render()


    def load_datamodel_dropdown(self):
        dropdown = QComboBox()
        dropdown.addItems(self.available_data)
        return dropdown

    def load_colormap_dropdown(self):
        dropdown = QComboBox()
        dropdown.addItems(self.available_tfs)
        return dropdown

    def load_attribute_dropdown(self):
        dropdown = QComboBox()
        return dropdown


    def start_rendering(self):
        print("Rendering...")
        self.render()
        # update attribute_list
        self.attribute_dropdown.currentTextChanged.disconnect(self.attribute_update)
        self.attribute_dropdown.clear()
        self.attribute_dropdown.addItems(attribute_list)
        self.attribute_dropdown.currentTextChanged.connect(self.attribute_update)


    def datamodel_update(self, s):
        if s == "":
            return
        if(self.loading_model):
            self.selected_file_path = os.path.join(savedmodels_folder, f'{s}\decompressed\decompressed_psnr_final.vtk')
        else:
            self.selected_file_path = os.path.join(data_folder, s)

        self.start_rendering()

    def colormap_update(self, s):
        if s == "":
            return
        self.selected_colormap = s
        self.loadColormap(self.selected_colormap)
        data_for_tf_editor = np.stack(
            [self.opacity_control_points,self.opacity_values],
            axis=0
        ).transpose()
        self.tf_editor.setData(pos=data_for_tf_editor)
        if self.selected_file_path is not None:
            self.render()

    def attribute_update(self, s):
        if s == "":
            return
        self.selected_attribute = s
        self.render()

    def SurfaceOrEdge_update(self, s):
        if s == "":
            return
        self.selected_SurfaceOrEdge = s
        self.render()


    def render(self):
        # Read the VTK file
        update_reader(self.selected_file_path, self.PointOrCell)


        if self.selected_attribute != '':
            if self.PointOrCell == 'cell':
                unstructured_grid.GetCellData().SetActiveScalars(self.selected_attribute)  # 设置活动标量为 nuTilda
            else:
                unstructured_grid.GetPointData().SetActiveScalars(self.selected_attribute)  # 设置活动标量为 nuTilda
        else:
            self.selected_attribute = attribute_list[0]
            if self.PointOrCell == 'cell':
                unstructured_grid.GetCellData().SetActiveScalars(self.selected_attribute)  # 设置活动标量为 nuTilda
            else:
                unstructured_grid.GetPointData().SetActiveScalars(self.selected_attribute)  # 设置活动标量为 nuTilda

        actor = vtk.vtkActor()
        if self.PointOrCell == 'point':
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(unstructured_grid)
            mapper.SetScalarModeToUsePointData()  # 确保使用点数据

            actor.GetProperty().SetRepresentationToPoints()  # 设置为点表示
            actor.GetProperty().SetPointSize(0.1)  # 设置点的大小

        else:
            if self.selected_SurfaceOrEdge == 'Edge':
                # 使用 vtkExtractEdges 提取边缘
                extractEdges = vtk.vtkExtractEdges()
                extractEdges.SetInputData(unstructured_grid)

                # 创建 vtkPolyDataMapper 来渲染提取的边缘
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputConnection(extractEdges.GetOutputPort())
            else:
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputConnection(reader.GetOutputPort())

        actor.SetMapper(mapper)

        # 创建一个新的颜色查找表
        lookupTable = vtk.vtkLookupTable()
        lookupTable.SetNumberOfTableValues(256)  # 设置颜色条的大小
        lookupTable.Build()
        # 为查找表设置颜色范围
        for i in range(256):
            # 示例：创建一个从蓝色到红色的渐变
            r = i / 255.0
            g = 0
            b = (255 - i) / 255.0
            lookupTable.SetTableValue(i, r, g, b, 1.0)  # RGBA
        # 将自定义的颜色查找表应用于映射器
        mapper.SetLookupTable(lookupTable)
        # 颜色条
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(mapper.GetLookupTable())
        # scalarBar.SetTitle("Color Scale")  # 设置色条的标题
        # titleTextProperty = scalarBar.GetTitleTextProperty()
        # titleTextProperty.SetColor(0, 0, 0)
        scalarBar.GetLabelTextProperty().SetColor(0, 0, 0)  # 设置文字颜色
        scalarBar.GetLabelTextProperty().SetFontSize(1)  # 设置字体大小
        # scalarBar.SetNumberOfLabels(5)  # 设置标签数量
        # 设置色条的大小和位置
        scalarBar.SetWidth(0.05)  # 色条宽度（相对于渲染窗口的比例）
        scalarBar.SetHeight(0.3)  # 色条高度（相对于渲染窗口的比例）
        scalarBar.SetPosition(0.9, 0.05)  # 色条在渲染窗口中的位置（x, y）
        # Update the renderer
        self.renderer.RemoveAllViewProps()  # Remove previous data
        self.renderer.AddActor(scalarBar)
        self.renderer.AddActor(actor)
        self.renderer.SetBackground(1, 1, 1)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def Save_Image(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self,
                                                  "Save Image",
                                                  "rendered_image.png",
                                                  "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
                                                  options=options)

        if filePath:  # 如果用户选择了文件

            renderWindow = self.vtkWidget.GetRenderWindow()
            # 创建vtkWindowToImageFilter以从渲染窗口捕获图像
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(renderWindow)
            windowToImageFilter.Update()

            # 根据文件扩展名选择合适的writer
            if filePath.lower().endswith('.png'):
                writer = vtk.vtkPNGWriter()
            elif filePath.lower().endswith('.jpg') or filePath.lower().endswith('.jpeg'):
                writer = vtk.vtkJPEGWriter()
            else:
                print("Unsupported file format")
                return

            # 创建一个图片写入器并保存图像
            writer.SetFileName(filePath)
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            writer.Write()

    def loadColormap(self, colormapname):
        '''
        Loads a colormap exported from Paraview. Assumes colormapname is a
        file path to the json to be loaded
        '''
        colormaps_folder = tf_folder
        file_location = os.path.join(colormaps_folder, colormapname)
        import json
        if (os.path.exists(file_location)):
            with open(file_location) as f:
                color_data = json.load(f)[0]
        else:
            print("Colormap file doesn't exist")
            exit()
            return

        # Load all RGB data
        rgb_data = color_data['RGBPoints']
        self.color_control_points = np.array(rgb_data[0::4], dtype=np.float32)
        self.color_control_points = self.color_control_points - self.color_control_points[0]
        self.color_control_points = self.color_control_points / self.color_control_points[-1]
        r = np.array(rgb_data[1::4], dtype=np.float32)
        g = np.array(rgb_data[2::4], dtype=np.float32)
        b = np.array(rgb_data[3::4], dtype=np.float32)
        self.color_values = np.stack([r, g, b], axis=1)

        # If alpha points set, load those, otherwise ramp opacity
        if ("Points" in color_data.keys()):
            a_data = color_data['Points']
            self.opacity_control_points = np.array(a_data[0::4], dtype=np.float32)
            self.opacity_control_points = self.opacity_control_points - self.opacity_control_points[0]
            self.opacity_control_points = self.opacity_control_points / self.opacity_control_points[-1]
            self.opacity_values = np.array(a_data[1::4], dtype=np.float32)

        else:
            self.opacity_control_points = np.array([0.0, 1.0], dtype=np.float32)
            self.opacity_values = np.array([0.0, 1.0], dtype=np.float32)
        print('1111')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
