import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
import pyqtgraph as pg


class TransferFunctionEditor(pg.GraphItem):
    valueChanged = pyqtSignal(np.ndarray)

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
            self.valueChanged.emit(self.data['pos'])

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
        self.valueChanged.emit(self.data['pos'])
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
            self.valueChanged.emit(self.data['pos'])

