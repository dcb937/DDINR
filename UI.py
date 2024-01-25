import sys
import vtk
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QStackedLayout, \
    QComboBox, QSlider, QFileDialog
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

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
    # # 创建并设置VTK读取器
    # reader = vtk_or_vtu_reader(data_path)
    # reader.SetFileName(data_path)
    # reader.Update()

    # # 获取VTK文件的输出
    # unstructured_grid = reader.GetOutput()

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
        self.resize(500,300)  # 必须得有。。。。
        self.showMaximized()  # 最大化窗口
        self.frame = QtWidgets.QFrame()

        # Find all available models/colormaps
        self.available_models = os.listdir(savedmodels_folder)
        self.available_tfs = os.listdir(tf_folder)
        self.available_data = os.listdir(data_folder)

        self.PointOrCell = 'point'
        self.selected_attribute = ''

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
        self.load_from_dropdown.currentTextChanged.connect(self.data_box_update)
        self.load_box.addWidget(self.load_from_dropdown)

        self.datamodel_box = QHBoxLayout()
        self.datamodel_box.addWidget(QLabel("Model/data: "))
        self.models_dropdown = self.load_models_dropdown()
        self.models_dropdown.currentTextChanged.connect(self.load_model)
        self.datamodel_box.addWidget(self.models_dropdown)

        self.PointOrCell_box = QHBoxLayout()
        self.PointOrCell_box.addWidget(QLabel("point or cell: "))
        self.PointOrCell_dropdown = QComboBox()
        self.PointOrCell_dropdown.addItems(["point", "cell"])
        self.PointOrCell_dropdown.currentTextChanged.connect(self.PointOrCell_update)
        self.PointOrCell_box.addWidget(self.PointOrCell_dropdown)

        self.attribute_box = QHBoxLayout()
        self.attribute_box.addWidget(QLabel("Attribute: "))
        self.attribute_dropdown = self.load_attribute_dropdown()
        self.attribute_dropdown.currentTextChanged.connect(self.attribute_update)
        self.attribute_box.addWidget(self.attribute_dropdown)

        self.button_box = QHBoxLayout()
        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self.start_rendering)  # 连接按钮点击事件到槽函数
        self.button_box.addWidget(self.render_button)  # 将按钮添加到设置面板的布局中

        # VTK Renderer
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.settings_ui.addLayout(self.load_box)
        self.settings_ui.addLayout(self.datamodel_box)
        self.settings_ui.addLayout(self.PointOrCell_box)
        self.settings_ui.addLayout(self.button_box)
        self.settings_ui.addLayout(self.attribute_box)
        self.settings_ui.addStretch()

        # UI full layout
        layout.addWidget(self.vtkWidget, stretch=4)
        layout.addLayout(self.settings_ui, stretch=1)
        layout.setContentsMargins(0, 0, 10, 10)
        layout.setSpacing(20)
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        self.setCentralWidget(self.centralWidget)

        self.show()

    def data_box_update(self, s):
        self.loading_model = "Model" in s
        if(self.loading_model):
            self.models_dropdown.clear()
            self.models_dropdown.addItems(self.available_models)
        else:
            self.models_dropdown.clear()
            self.models_dropdown.addItems(self.available_data)

    def PointOrCell_update(self, s):
        self.PointOrCell = s


    def load_models_dropdown(self):
        dropdown = QComboBox()
        dropdown.addItems(self.available_data)
        return dropdown

    def load_attribute_dropdown(self):
        dropdown = QComboBox()
        return dropdown


    def start_rendering(self):
        print("Rendering...")
        self.render()

        self.update_attribute_dropdown(attribute_list)

    def load_model(self, s):
        if s == "":
            return
        # self.status_text_update.emit(f"Loading model {s}...")
        if(self.loading_model):
            self.selected_file_path = os.path.join(savedmodels_folder, s)
            # self.render_worker.load_new_model.emit(s)
        else:
            self.selected_file_path = os.path.join(data_folder, s)
            # self.render_worker.load_new_data.emit(s)
        # self.status_text_update.emit("")

    def attribute_update(self, s):
        if s == "":
            return
        self.selected_attribute = s
        self.render()

    def update_attribute_dropdown(self, items):
        # 断开信号以避免代码触发的变更调用槽函数
        self.attribute_dropdown.currentTextChanged.disconnect(self.attribute_update)

        # 清空并更新下拉菜单项
        self.attribute_dropdown.clear()
        self.attribute_dropdown.addItems(items)

        # 重新连接信号和槽函数
        self.attribute_dropdown.currentTextChanged.connect(self.attribute_update)

    def render(self):
        # Read the VTK file
        update_reader(self.selected_file_path, self.PointOrCell)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        if self.selected_attribute != '':
            if self.PointOrCell == 'cell':
                unstructured_grid.GetCellData().SetActiveScalars(self.selected_attribute)  # 设置活动标量为 nuTilda
            else:
                unstructured_grid.GetPointData().SetActiveScalars(self.selected_attribute)  # 设置活动标量为 nuTilda

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Update the renderer
        self.renderer.RemoveAllViewProps()  # Remove previous data
        self.renderer.AddActor(actor)
        self.renderer.SetBackground(1, 1, 1)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
