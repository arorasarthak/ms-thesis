from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import zmq
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from scipy.stats import norm, gamma, uniform

R_std = 0.35
Q_std = 0.04
tracker = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.0  # time step

tracker.F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])
tracker.u = 0.
tracker.H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

tracker.R = np.eye(2) * R_std ** 2
q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std ** 2)
tracker.Q = block_diag(q, q)
tracker.x = np.array([[0, 0, 0, 0]]).T
tracker.P = np.eye(4) * 500.


# pf = ParticleFilter(
#         prior_fn=prior_fn,
#         observe_fn=blob,
#         n_particles=100,
#         dynamics_fn=tracker.F,
#         noise_fn=lambda x: cauchy_noise(x, sigmas=[0.05, 0.05, 0.01, 0.005, 0.005]),
#         weight_fn=lambda x, y: squared_error(x, y, sigma=2),
#         resample_proportion=0.2,
#         column_names=columns,
#     )


app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('Sensor View')

g = gl.GLGridItem()
g.setSize(x=6, y=6)
w.addItem(g)

origin = np.zeros((1, 3))

pc_plot = gl.GLScatterPlotItem(pos=origin, color=(1, 1, 1, .3), size=0.1, pxMode=False)
human_plot = gl.GLScatterPlotItem(pos=origin, color=(1, 1, 1, .3), size=0.1, pxMode=False)
ground_plot = gl.GLScatterPlotItem(pos=origin, color=(1, 1, 1, .3), size=0.1, pxMode=False)

self_plot = gl.GLScatterPlotItem(pos=origin, color=(1, 1, 1, .3), size=0.1, pxMode=False)
nothing_plot = gl.GLScatterPlotItem(pos=origin, color=(1, 1, 1, .3), size=0.1, pxMode=False)
filtered_plot = gl.GLScatterPlotItem(pos=origin, color=(1.0, 0, 0, 1.0), size=0.1, pxMode=False)

kalman_plot = gl.GLScatterPlotItem(pos=origin, color=(1.0, 0, 0, 1.0), size=0.1, pxMode=False)
pf_plot = gl.GLScatterPlotItem(pos=origin, color=(1.0, 0, 0, 1.0), size=0.1, pxMode=False)


w.addItem(pc_plot)
w.addItem(human_plot)
w.addItem(ground_plot)
w.addItem(self_plot)
w.addItem(nothing_plot)
w.addItem(filtered_plot)
w.addItem(kalman_plot)
w.addItem(pf_plot)

# Point Cloud Receiver
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:%s" % port)

# Error Plot Sender
# err_context = zmq.Context()
# socket_err = err_context.socket(zmq.PUSH)
# socket_err.bind("tcp://*:%s" % "9818")

# Data Sender for matlab engine
matlab_snd_context = zmq.Context()
matlab_send = matlab_snd_context.socket(zmq.PUSH)
matlab_send.bind("tcp://*:%s" % "6667")

# Data Receiver for matlab engine
matlab_recv_context = zmq.Context()
matlab_recv = matlab_recv_context.socket(zmq.PULL)
matlab_recv.connect("tcp://localhost:%s" % "6668")


# socket.setsockopt_string(zmq.SUBSCRIBE, '')



def update():

    # update volume colors
    buffer = []
    nonself = []
    nothing = []
    filtered = None
    last_ground_pose = np.zeros((1, 3))
    # w.setCameraPosition(np.random.randn(1, 3))
    msg = socket.recv_pyobj()
    # print(msg)
    # filtered_plot.setData(pos=msg[0], size=0.2, color=(1.0, 0, 0, 1.0))
    human_plot.setData(pos=msg[1], size=0.5, color=(1.0, 1.0, 0.0, 0.3))
    pc_plot.setData(pos=msg[0][:, :-1], size=0.1, color=(1.0, 0.0, 0.0, 0.5))

    for each in msg[0]:
        if each[-1] == 0:
            buffer.append(each[:-1])
        if each[-1] == 1:
            nonself.append(each[:-1])

    if len(nonself) > 0:
        nonself_buff = np.vstack(nonself)
        ground_pose = np.mean(nonself_buff, axis=0)
        ground_pose[-1] = 0
        ground_pose = ground_pose.reshape(1, 3)

        # KALMAN FILTER
        tracker.predict()
        tracker.update(z=np.transpose(ground_pose)[:-1])

        # PARTICLE FILTER
        filter_msg = np.transpose(ground_pose)[:-1]
        matlab_send.send_pyobj(filter_msg)
        #print(filter_output == filter_msg)


        nothing_plot.setData(pos=ground_pose, size=0.3, color=(0.0, 0.0, 1.5, 1.0))
        #socket_err.send_pyobj([ground_pose, msg[1]])
        last_ground_pose = ground_pose
    else:
        tracker.predict()
        matlab_send.send_pyobj(np.zeros((1, 2)))

        #socket_err.send_pyobj([last_ground_pose, msg[1]])

    if len(buffer) > 0:
        self_buff = np.vstack(buffer)
        self_plot.setData(pos=self_buff, size=0.1, color=(0.0, 1.0, 1.0, 0.5))
    #tracker.predict()

    #print(tracker.x)
    xp = tracker.x
    tracked_pose = np.array([xp[0][0], xp[2][0], 0])
    tracked_pose = np.reshape(tracked_pose, (1, 3))
    kalman_plot.setData(pos=tracked_pose, size=0.3, color=(0.3, 0.6, 0.7, 1.0))

    pf_output = matlab_recv.recv_pyobj()
    pf_output = pf_output.flatten()

    temp = np.zeros((1, 3))
    temp[:, 0] = pf_output[0]
    temp[:, 1] = pf_output[2]
    #print(temp)
    pf_plot.setData(pos=temp, size=0.3, color=(0.2, 0.9, 0.1, 1.0))

    # msg = msg[0]
    # for each in msg[:-1]:
    #     if each[-1] == 1:
    #         buffer.append(each[:-1])
    #     elif each[-1] == 0:
    #         self.append(each[:-1])
    #     else:
    #         nothing.append(each[:-1])
    # pc_plot.setData(pos=np.vstack(buffer), size=0.1, color=(1.0, 0.0, 0.0, 0.5))
    # self_plot.setData(pos=np.vstack(self), size=0.1, color=(0.0, 1.0, 0.0, 0.5))
    # nothing_plot.setData(pos=np.vstack(nothing), size=0.1, color=(0.5, 0.5, 0.5, 1.0))

    # pc_plot.setData(pos=msg, size=0.1, color=(1.0, 0.0, 0.0, 0.5))

    # ground_plot.setData(pos=msg_new, size=0.1, color=(0.5, 0.5, 0.5, 0.5))


# socket.setsockopt(zmq.SUBSCRIBE)

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(1)

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
