import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF
import pyqtgraph as pg

class Plot(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Plot')
        # self.setGeometry(0, 0, 200, 200)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)

        self.points = [(0, 0), (1, 1)]  # 默认初始情况只有两个点

        self.drawPoints()

    def drawPoints(self):
        self.scene.clear()
        # 绘制坐标轴
        pen = QPen(Qt.black)
        self.scene.addLine(0, self.height() / 2, self.width(), self.height() / 2, pen)
        self.scene.addLine(self.width() / 2, 0, self.width() / 2, self.height(), pen)

        # 绘制已有点和连线
        pen.setColor(Qt.red)
        for i in range(len(self.points)):
            x, y = self.points[i]
            point = QPointF(x * self.width(), self.height() - y * self.height())
            self.scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10, pen)
            if i > 0:
                prev_x, prev_y = self.points[i - 1]
                prev_point = QPointF(prev_x * self.width(), self.height() - prev_y * self.height())
                self.scene.addLine(prev_point.x(), prev_point.y(), point.x(), point.y(), pen)

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        return self.view.sizeHint()

    def resizeEvent(self, event):
        self.view.setGeometry(0, 0, self.width(), self.height())
        self.drawPoints()

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            x = event.pos().x() / self.width()
            y = (self.height() - event.pos().y()) / self.height()
            new_point = (x, y)

            # 检查是否点击到已有点附近
            for point in self.points:
                px, py = point
                if abs(px - x) < 0.05 and abs(py - y) < 0.05:
                    # 删除点
                    self.points.remove(point)
                    self.drawPoints()
                    return

            # 添加新点
            self.points.append(new_point)
            self.points.sort()
            self.drawPoints()


class TransferFunctionEditor(pg.GraphItem):
    def __init__(self, parent=None):
        self.dragPoint = None
        self.dragOffset = None
        self.lastDragPointIndex = 0
        self.parent = parent
        pg.GraphItem.__init__(self)

    def setData(self, **kwds):
        '''
        Assumes kwds['pos'] is a pre-sorted lists of tuples of control point -> opacity
        sorted by control point value. I.e.
        [[0, 0], [0.5, 1.0], [1.0, 0.0]]
        is a mountain and is legal because kwds['pos'][:,0] is strictly increasing.
        '''
        self.data = kwds
        self.data['size'] = 12
        self.data['pxMode'] = True
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            # Normalize control point x to [0,1]
            self.data['pos'][:, 0] -= self.data['pos'][0, 0]
            self.data['pos'][:, 0] /= self.data['pos'][-1, 0]
            # Clip opacity between 0 and 1
            self.data['pos'][:, 1] = np.clip(self.data['pos'][:, 1], 0.0, 1.0)
            self.data['adj'] = np.column_stack((np.arange(0, npts - 1), np.arange(1, npts)))
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        if (self.parent is not None):
            if "pos" in self.data.keys():
                opacity_control_points = self.data['pos'][:, 0]
                opacity_values = self.data['pos'][:, 1]
                # if self.parent.render_worker is not None:
                #     self.parent.render_worker.change_opacity_controlpoints.emit(
                #         opacity_control_points, opacity_values
                #     )

    def deleteLastPoint(self):
        if (self.lastDragPointIndex > 0 and
                self.lastDragPointIndex < self.data['pos'].shape[0] - 1):
            new_pos = np.concatenate(
                [self.data['pos'][0:self.lastDragPointIndex],
                 self.data['pos'][self.lastDragPointIndex + 1:]],
                axis=0
            )
            self.data['pos'] = new_pos
            self.setData(**self.data)
            self.lastDragPointIndex -= 1

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.lastDragPointIndex = ind
            self.dragOffset = [
                self.data['pos'][ind][0] - pos[0],
                self.data['pos'][ind][1] - pos[1]
            ]
        elif ev.isFinish():
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]

        # Cannot move endpoints
        if (ind == 0 or ind == self.data['pos'].shape[0] - 1):
            # only move y
            self.data['pos'][ind][1] = np.clip(ev.pos()[1] + self.dragOffset[1], 0.0, 1.0)
        # Points in between cannot move past other points to maintain ordering
        else:
            # move x
            self.data['pos'][ind][0] = np.clip(ev.pos()[0] + self.dragOffset[0],
                                               self.data['pos'][ind - 1][0] + 1e-4,
                                               self.data['pos'][ind + 1][0] - 1e-4)
            # move y
            self.data['pos'][ind][1] = np.clip(ev.pos()[1] + self.dragOffset[1], 0.0, 1.0)

        self.updateGraph()
        ev.accept()

    def mouseClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        p = event.pos()
        x = np.clip(p.x(), 0.0, 1.0)
        y = np.clip(p.y(), 0.0, 1.0)

        pts = self.scatter.pointsAt(p)
        if len(pts) > 0:
            return

        if x > 0 and x < 1.0:
            ind = 0
            while x > self.data['pos'][ind][0]:
                ind += 1
            new_pos = np.concatenate(
                [self.data['pos'][0:ind],
                 [[x, y]],
                 self.data['pos'][ind:]
                 ],
                axis=0
            )
            self.data['pos'] = new_pos
            self.setData(**self.data)
            self.lastDragPointIndex = ind

