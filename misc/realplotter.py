from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
import zmq

zmq_port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:%s" % zmq_port)


pp = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('Sensor View')

g = gl.GLGridItem()
g.setSize(x=6, y=6)
w.addItem(g)

origin = np.zeros((1, 3))

pc_plot = gl.GLScatterPlotItem(pos=origin, color=(1, 1, 1, .3), size=0.1, pxMode=False)
sensor_readings = gl.GLScatterPlotItem(pos=origin, color=(1, 1, 1, .3), size=0.1, pxMode=False)
w.addItem(pc_plot)
w.addItem(sensor_readings)

ctr = 0

data = np.load("/home/sarthak/PycharmProjects/research_ws/data/real-data/dataset14.npy")
data1 = np.load("/home/sarthak/PycharmProjects/research_ws/data/real-data/dataset14_1.npy")
print(data.shape)
#base_sensors = np.reshape()
#elbow_sensors =
dim = data.shape[0]
tool_sensors = np.reshape(data[:, -27: -3], (dim, -1, 3))
elbow_sensors = np.reshape(data[:, -51: -27], (dim, -1, 3))
base_sensors = np.reshape(data[:, -75:-51], (dim, -1, 3))

q = data[:,51:57]
q[:,1], q[:,3] = q[:,1] + 1.57079, q[:,3] + 1.57079
print(q)

hum_pose = data[:, -3:]
print(elbow_sensors.shape)

base_obs = data[:, 1: 9]
base_gt = data[:, 9:17] ##ground truth labels


print(base_gt[400])

elbow_obs = data[:, 18: 26]
elbow_gt = data[:, 26:34] #ground truth labels

tool_obs = data[:, 35: 43]
tool_gt = data[:, 43:51] #ground truth labels

base_quat = np.reshape(data1[:, 24:56], (dim, 8, 4))
elbow_quat = np.reshape(data1[:, 80:112], (dim, 8, 4))
tool_quat = np.reshape(data1[:, 136:], (dim, 8, 4))


old_sum = 0
new_sum = 0

def update():
    global old_sum, new_sum
    global ctr
    global dim
    #data = np.vstack([base_sensors[ctr], elbow_sensors[ctr], tool_sensors[ctr], hum_pose[ctr]])

    dists = [tool_obs[ctr], elbow_obs[ctr], base_obs[ctr]]
    object_labels = [tool_gt[ctr], elbow_gt[ctr], base_gt[ctr]]

    sensors = [tool_sensors[ctr], elbow_sensors[ctr], base_sensors[ctr]]
    quats = [tool_quat[ctr], elbow_quat[ctr], base_quat[ctr]]
    points = []

    tool_mask = np.array(np.logical_and(tool_gt[ctr] + 1, tool_gt[ctr] + 1), dtype='int')
    elbow_mask = np.array(np.logical_and(elbow_gt[ctr] + 1, elbow_gt[ctr] + 1), dtype='int')
    base_mask = np.array(np.logical_and(base_gt[ctr] + 1, base_gt[ctr] + 1), dtype='int')

    mask_column = np.hstack([tool_mask, elbow_mask, base_mask])


    for i in range(3):
        for idx, each in enumerate(dists[i]):
            if each > 0:
                vector = np.zeros((4, 1))
                vector[-1] = 1
                vector[0] = each/1000.0
                matrix = np.identity(4)
                matrix[:-1, -1] = sensors[i][idx]
                matrix[0:3, 0:3] = R.from_quat(quats[i][idx]).as_dcm()
                point = np.matmul(matrix, vector)
                point = np.reshape(point[:-1], (1, 3))
                points.append(point)
            if each == 0:
                point = np.zeros((1, 3))
                points.append(point)

    #print(base_obs[ctr])
    if len(points) > 0:
        pt = np.vstack(points)
        #print(pt)
        sensor_readings.setData(pos=pt, size=0.1, color=(1.0, 1.0, 1.0, 1.0))
    #print(pt)
    data = np.vstack([base_sensors[ctr], elbow_sensors[ctr], tool_sensors[ctr], hum_pose[ctr]])
    pc_plot.setData(pos=data, size=0.025, color=(1.0, 0.0, 0.0, 1.0))

    #num_detections = points
    points = np.vstack(points)
    socket.send_pyobj([points,  hum_pose[ctr], mask_column, q[ctr]])

    if ctr < dim - 1:
        ctr += 1
    else:
        ctr = 0


t = QtCore.QTimer()
t.timeout.connect(update)
t.start(62.5)

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()