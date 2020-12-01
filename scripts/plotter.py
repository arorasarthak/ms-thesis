from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import zmq
import time
from collections import deque

app = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget()
win.show()
win.resize(1000, 600)
win.setWindowTitle('Error Plot')

pg.setConfigOptions(antialias=False)

p6 = win.addPlot(title="RMSE")
curve = p6.plot()
data_ = np.random.normal(size=(10, 1000))
ptr = 0

err_context = zmq.Context()
socket_err = err_context.socket(zmq.PULL)
socket_err.connect("tcp://localhost:%s" % "9818")

buffer = deque()
p6.enableAutoRange('y', False)
# p6.setAutoPan(x=True)
# p6.setXRange(min=0, max=1)
def update():
    global curve, data_, ptr, p6
    msg = socket_err.recv_pyobj()

    y = msg[1]
    y_hat = msg[0]
    error = np.sqrt(np.square(y - y_hat).mean())

    # buffer.append(error)
    point = np.array([int(ptr), error]).reshape(1, 2)
    # print(point)
    if len(buffer) > 1000:
        buffer.popleft()
    else:
        buffer.append(point)

    if len(buffer) > 100:
        curve.setData(np.vstack(buffer))

    # if ptr == 0:
    #     p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    ptr += 1


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

# win.nextRow()
#
# view = win.addViewBox()
# view.setAspectLocked(True)
# img = pg.ImageItem(border='w')
# view.addItem(img)
# view.setRange(QtCore.QRectF(0, 0, 8, 8))
# #data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
# #print(data.shape)
# #data = np.random.randn(8, 8)
#
# data = np.random.random_integers(5, size=(8, 8))
# i = 0
#
# updateTime = ptime.time()
# fps = 0
#
#
# def updateData():
#     global img, data, i, updateTime, fps
#
#     ## Display the data
#     img.setImage(np.random.randn(8, 8))
#     #i = (i + 1) % data.shape[0]
#
#     #QtCore.QTimer.singleShot(1, updateData)


# timer1 = QtCore.QTimer()
# timer1.timeout.connect(updateData)
# timer1.start(1)


if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
