from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
import zmq


zmq_port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:%s" % zmq_port)

points = []
q = []
points_filtered = []
human_position = []

try:
    while True:
        msg = socket.recv_pyobj()
        non_self_pc = msg[0] * np.reshape(msg[2], (24, 1))
        points_filtered.append(non_self_pc)
        points.append(msg[0])
        human_position.append(msg[1])
        q.append(msg[-1])
except KeyboardInterrupt:
    print("Saving")
    np.save("../data/1filter_points.npy", np.vstack(points_filtered))
    np.save("../data/1raw_points.npy", np.vstack(points))
    np.save("../data/1op_position.npy", np.vstack(human_position))
    np.save("../data/1q.npy", np.vstack(q))
