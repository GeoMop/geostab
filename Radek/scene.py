#!/usr/bin/env python
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
from PyQt5.QtCore import Qt
import mouse
import math


"""
TODO:

- switch buttons
- decomposition:
  wrapper classes: Point, Segment, Polygon (merge with GsPoint etc. add additional info)
  keep both diretion relation to GsPoint, are organized 
  into a dict using same decomp IDs, keep back reference to decomp objects.
   
- move lines with points
- decomposition two modes - with and without intersection tracking, with and without polygons
- add decomposition
"""

class Cursor:
    @classmethod
    def setup_cursors(cls):
        cls.point =  QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        cls.segment = QtGui.QCursor(QtGui.QCursor(QtCore.Qt.UpArrowCursor))
        cls.draw = QtGui.QCursor(QtCore.Qt.ArrowCursor)

class Region:
    _cols = ["cyan", "magenta", "red", "darkRed", "darkCyan", "darkMagenta",
             "green", "darkBlue", "yellow","blue"]
    colors = [ QtGui.QColor(col) for col in _cols]
    id_next = 0



    def __init__(self, id = None, color = None ):
        if id is None:
            id = Region.id_next
            Region.id_next += 1
        self.id = id

        if color is None:
            color = Region.colors[self.id%len(Region.colors)].name()
        self.color = color

# Special instances
Region.none =  Region(0, "grey")




class Point:
    def __init__(self, x, y, region):
        #self.gpt = None
        self.xy = np.array([x, y])
        self.region = region

    def set_xy(self, x, y):
        print("point move")
        self.xy[0] = x
        self.xy[1] = y

    def segments(self):
        """
        Generator of connected segments.
        :return:
        """

    def g_segments(self):
        """
        Generator of graphical segments.
        :return:
        """
        for seg in self.segments():
            yield seg.g_segment


class Segment:
    def __init__(self, pt_from, pt_to, region):
        #self.g_segment = None
        self.points = [pt_from, pt_to]
        self.region = region



class GsPoint(QtWidgets.QGraphicsEllipseItem):
    SIZE = 6
    STD_ZVALUE = 1
    SELECTED_ZVALUE = 0
    __pen_table={}

    no_brush = QtGui.QBrush(QtCore.Qt.NoBrush)
    no_pen = QtGui.QPen(QtCore.Qt.NoPen)
    add_brush = QtGui.QBrush(QtCore.Qt.darkGreen, QtCore.Qt.SolidPattern)

    @classmethod
    def make_pen(cls, color):
        brush = QtGui.QBrush(color, QtCore.Qt.SolidPattern)
        pen = QtGui.QPen(color, 1.4, QtCore.Qt.SolidLine)
        return (brush, pen)

    @classmethod
    def pen_table(cls, color):
        brush_pen = cls.__pen_table.setdefault(color, cls.make_pen(QtGui.QColor(color)))
        return brush_pen

    def __init__(self, pt):
        self.pt = pt
        #pt.gpt = self
        super().__init__(-self.SIZE, -self.SIZE, 2*self.SIZE, 2*self.SIZE, )
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
            # do not scale points whenzooming
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
            # Keep point shape (used for mouse interaction) having same size as the point itself.
        #self.setFlag(QtWidgets.QGraphicsItem.ItemClipsToShape, True)
        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
            # Caching: Points can move.
        self.setCursor(Cursor.point)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.RightButton)
        self.update()

    def paint(self, painter, option, widget):
        print("option: ", option.state)
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.setBrush(GsPoint.no_brush)
            painter.setPen(self.region_pen)
        else:
            painter.setBrush(self.region_brush)
            painter.setPen(GsPoint.no_pen)
        painter.drawEllipse(self.rect())

    def update(self):
        self.setPos(self.pt.xy[0], self.pt.xy[1])
        self.region_brush, self.region_pen = GsPoint.pen_table(self.pt.region.color)
        self.setZValue(self.STD_ZVALUE)
        super().update()


    def move_to(self, x, y):
        self.pt.set_xy(x, y)
        for gseg in self.pt.g_segments():
            gseg.update()
        self.update()


    def itemChange(self, change, value):
        """
        The item enables itemChange() notifications for
        ItemPositionChange, ItemPositionHasChanged, ItemMatrixChange,
        ItemTransformChange, ItemTransformHasChanged, ItemRotationChange,
        ItemRotationHasChanged, ItemScaleChange, ItemScaleHasChanged,
        ItemTransformOriginPointChange, and ItemTransformOriginPointHasChanged.
        """
        #print("change: ", change, "val: ", value)
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.pt.set_xy(value.x(), value.y())
        if change == QtWidgets.QGraphicsItem.ItemSelectedChange:
            if self.isSelected():
                self.setZValue(self.SELECTED_ZVALUE)
            else:
                self.setZValue(self.STD_ZVALUE)
        return super().itemChange(change, value)


    # def mousePressEvent(self, event):
    #     self.update()
    #     super().mousePressEvent(event)
    #
    # def mouseReleaseEvent(self, event):
    #     self.update()
    #     super().mouseReleaseEvent(event)



class GsSegment(QtWidgets.QGraphicsLineItem):
    __pen_table={}

    WIDTH = 3.0
    STD_ZVALUE = 11
    SELECTED_ZVALUE = 10
    no_pen = QtGui.QPen(QtCore.Qt.NoPen)


    @classmethod
    def make_pen(cls, color):
        pen = QtGui.QPen(color, cls.WIDTH, QtCore.Qt.SolidLine)
        pen.setCosmetic(True)
        selected_pen = QtGui.QPen(color, cls.WIDTH, QtCore.Qt.DashLine)
        selected_pen.setCosmetic(True)
        return (pen, selected_pen)

    @classmethod
    def pen_table(cls, color):
        pens = cls.__pen_table.setdefault(color, cls.make_pen(QtGui.QColor(color)))
        return pens

    def __init__(self, segment):
        self.segment = segment
        #segment.g_segment = self
        super().__init__()
        #self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        self.setCursor(Cursor.segment)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.RightButton)
        self.update()
        pass

    def update(self):
        pt_from, pt_to = self.segment.points
        self.setLine(pt_from.xy[0], pt_from.xy[1], pt_to.xy[0], pt_to.xy[1])
        self.region_pen, self.region_selected_pen  = GsSegment.pen_table(self.segment.region.color)
        super().update()

    def paint(self, painter, option, widget):
        if option.state & (QtWidgets.QStyle.State_Sunken | QtWidgets.QStyle.State_Selected):
            painter.setPen(self.region_selected_pen)
        else:
            painter.setPen(self.region_pen)
        painter.drawLine(self.line())

    def itemChange(self, change, value):
        #print("change: ", change, "val: ", value)
        #if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
        #    self.pt.set_xy(value.x(), value.y())
        if change == QtWidgets.QGraphicsItem.ItemSelectedChange:
            if self.isSelected():
                self.setZValue(self.SELECTED_ZVALUE)
            else:
                self.setZValue(self.STD_ZVALUE)
        return super().itemChange(change, value)

class TmpPoint(Point):
    def __init__(self):
        super().__init__(0, 0, Region.adding)

class TmpSegment(Segment):
    def __init__(self):
        super().__init__(0, 0, Region.adding)

class Diagram(QtWidgets.QGraphicsScene):

    def __init__(self, parent):
        rect = QtCore.QRectF(-622500, 1128600, 400, 500)
        super().__init__(rect, parent)
        self.points = []
        self.regions = []
        self.segments = []

        self.last_point = None
        self.aux_pt, self.aux_seg = self.create_aux_segment()
        self.hide_aux_line()

        pen = QtGui.QPen()
        pen.setWidthF(0)
        self.addRect(rect, pen)


    def create_aux_segment(self):
        pt_size = GsPoint.SIZE
        no_pen = QtGui.QPen(QtCore.Qt.NoPen)
        add_brush = QtGui.QBrush(QtCore.Qt.darkGreen, QtCore.Qt.SolidPattern)
        pt = self.addEllipse(-pt_size, -pt_size, 2*pt_size, 2*pt_size, no_pen, add_brush)
        pt.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        pt.setCursor(Cursor.draw)
        add_pen = QtGui.QPen(QtGui.QColor(QtCore.Qt.darkGreen), GsSegment.WIDTH)
        add_pen.setCosmetic(True)
        line = self.addLine(0,0,0,0, add_pen)
        return pt, line

    def move_aux_segment(self, tip, origin=None):
        """
        Update tip point and show aux segment and point.
        :param tip: Tip point (QPointF)
        :param origin: Origin point (QPointF)
        """
        self.aux_pt.show()
        self.aux_seg.show()
        self.aux_pt.setPos(tip)
        if origin is None:
            origin = self.aux_seg.line().p1()
        self.aux_seg.setLine(QtCore.QLineF(origin, tip))

    def hide_aux_line(self):
        self.aux_pt.hide()
        self.aux_seg.hide()



    def add_point(self, pos, gitem):
        if type(gitem) == GsPoint:
            return gitem
        else:
            #if type(gitem) == GsSegment:
            pt = Point(pos.x(), pos.y(), Region.none)
            gpt = GsPoint(pt)
            self.points.append(pt)
            self.addItem(gpt)
            return gpt

    def add_segment(self, gpt1, gpt2):
        seg = Segment(gpt1.pt, gpt2.pt, Region.none)
        gseg = GsSegment(seg)
        self.segments.append(seg)
        self.addItem(gseg)

    def new_point(self, pos, gitem, close = False):
        #print("below: ", gitem)
        new_g_point = self.add_point(pos, gitem)
        if self.last_point is not None:
            self.add_segment(self.last_point, new_g_point)
        if not close:
            self.last_point = new_g_point
            pt = new_g_point.pos()
            self.move_aux_segment(pt, origin=pt)
        else:
            self.last_point = None
            self.hide_aux_line()

    def mouse_create_event(self, event):
        transform = self.parent().transform()
        below_item = self.itemAt(event.scenePos(), transform)
        close = event.modifiers() & mouse.Event.Ctrl
        self.new_point(event.scenePos(), below_item, close)
        event.accept()

    def mousePressEvent(self, event):
        """
        :param event: QGraphicsSceneMouseEvent
        :return:
        """
        #print("P last: ", event.lastScenePos())
        if event.button() == mouse.Event.Right and self.last_point is None:
            self.mouse_create_event(event)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        :param event: QGraphicsSceneMouseEvent
        :return:
        """
        #print("R last: ", event.lastScenePos())
        if event.button() == mouse.Event.Right:
            self.mouse_create_event(event)
        super().mouseReleaseEvent(event)


    def mouseMoveEvent(self, event):
        if self.last_point is not None:
            self.move_aux_segment(event.scenePos())
        super().mouseMoveEvent(event)


class DiagramView(QtWidgets.QGraphicsView):
    def __init__(self):

        super(DiagramView, self).__init__()
        print(self)


        self._zoom = 0
        self._empty = True
        self._scene = Diagram(self)
        self.setScene(self._scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        #self.setFrameShape(QtWidgets.QFrame.Box)
        #self.ensureVisible(self._scene.sceneRect())
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)



    #def mouseDoubleClickEvent(self, QMouseEvent):
    #    pass


    def mousePressEvent(self, event):
        return super().mousePressEvent(mouse.event_swap_buttons(event, QtCore.QEvent.MouseButtonPress))

    def mouseReleaseEvent(self, event):
        return super().mouseReleaseEvent(mouse.event_swap_buttons(event, QtCore.QEvent.MouseButtonRelease))

    def mouseMoveEvent(self, event):
        return super().mouseMoveEvent(mouse.event_swap_buttons(event, QtCore.QEvent.MouseMove))






    # def hasPhoto(self):
    #     return not self._empty
    #
    # def fitInView(self, scale=True):
    #     rect = QtCore.QRectF(self._photo.pixmap().rect())
    #     if not rect.isNull():
    #         self.setSceneRect(rect)
    #         if self.hasPhoto():
    #             unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
    #             self.scale(1 / unity.width(), 1 / unity.height())
    #             viewrect = self.viewport().rect()
    #             scenerect = self.transform().mapRect(rect)
    #             factor = min(viewrect.width() / scenerect.width(),
    #                          viewrect.height() / scenerect.height())
    #             self.scale(factor, factor)
    #         self._zoom = 0
    #
    # def setPhoto(self, pixmap=None):
    #     self._zoom = 0
    #     if pixmap and not pixmap.isNull():
    #         self._empty = False
    #         self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
    #         self._photo.setPixmap(pixmap)
    #     else:
    #         self._empty = True
    #         self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
    #         self._photo.setPixmap(QtGui.QPixmap())
    #     self.fitInView()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        self.scale(factor, factor)

    # def toggleDragMode(self):
    #     if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
    #         self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
    #     elif not self._photo.pixmap().isNull():
    #         self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    # def mousePressEvent(self, event):
    #     if self._photo.isUnderMouse():
    #         self.photoClicked.emit(QtCore.QPoint(event.pos()))
    #     super(PhotoViewer, self).mousePressEvent(event)

    def show_electrodes(self, electrode_groups):
        for eg in electrode_groups:
            region = Region()
            for el in eg.electrodes:
                x = el.x
                y = -el.y
                print("x: {}, y: {}".format(x, y))
                pt = Point(x, y, region)
                gpt = GsPoint(pt)
                self._scene.addItem(gpt)

        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def show_map(self):
        map = QtSvg.QGraphicsSvgItem("bukov_situace.svg")

        # map transform
        # 622380 - 247.266276267186
        # 1128900 - 972.212997362655
        # 1128980 - 1309.97292588439
        map.setTransformOriginPoint(247.266276267186, 972.212997362655)
        map.setScale((1128980 - 1128900) / (1309.97292588439 - 972.212997362655))
        map.setPos(-622380 - 247.266276267186, 1128900 - 972.212997362655)

        self._scene.addItem(map)
        map.setCursor(QtCore.Qt.CrossCursor)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Cursor.setup_cursors()
    mainWindow = DiagramView()
    mainWindow.setGeometry(500, 300, 800, 600)
    mainWindow.show()
    sys.exit(app.exec_())
