import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from scripts.model import ConvNet
from scripts.utils import generate_self_vector, point_cloud_2_birdseye
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

#zmqstuff
import zmq
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:%s" % port)
keep_alive = True

# load model
conv_net = ConvNet()
conv_net.load_weights("../tf_models/model")

# pyqt graph stuff
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('Sensor View')

g = gl.GLGridItem()
g.setSize(x=6, y=6)
w.addItem(g)

origin = np.zeros((1, 3))

gt_plot = gl.GLScatterPlotItem(pos=origin, color=(0.5, 0, 0, .3), size=0.1, pxMode=False)
pred_plot = gl.GLScatterPlotItem(pos=origin, color=(0, 0.5, 1, .3), size=0.1, pxMode=False)
human_plot = gl.GLScatterPlotItem(pos=origin, color=(0.5, 0.5, 0.5, .3), size=0.1, pxMode=False)

#raw_plot = gl.GLScatterPlotItem(pos=origin, color=(0.5, 0.5, 0.5, .3), size=0.1, pxMode=False)
w.addItem(gt_plot)
w.addItem(pred_plot)
w.addItem(human_plot)

predictions = []
ground_truths = []
human_position = []
q = []

ctr = 0

def update():
    global ctr, predictions
    msg = socket.recv_pyobj()
    pc = msg[0] # msg[0] for simulation
    hum_pose = np.reshape(msg[1], (1, 3))

    #print(hum_pose)
    bev_img = point_cloud_2_birdseye(pc, res=0.2)
    bev_img = bev_img / 255.
    pred = conv_net(bev_img)
    # tf.print(pred)
    gt = generate_self_vector(pc)
    # loss = tf.losses.mse(gt, pred)
    gt = np.reshape(gt, (24, 1))
    pred_np = np.reshape(pred, (24, 1))
    gt_pc = pc[:, :-1] * np.logical_not(gt)
    r_prediction = np.round(pred_np)
    gt_mask = gt #for sim world data
    pred_pc = pc[:,:-1] * np.logical_not(np.round(pred_np))

    predictions.append(pred_pc)
    ground_truths.append(gt_pc)
    human_position.append(hum_pose)
    q.append(np.array(msg[-1]))

    if ctr == 1000:
        print("Saving")
        np.save("../data/pred_pc_sim.npy", np.vstack(predictions))
        np.save("../data/gt_pc_sim.npy", np.vstack(ground_truths))
        np.save("../data/human_position_sim.npy", np.vstack(human_position))
        np.save("../data/q_sim.npy", np.vstack(q))
    # print(confusion_matrix(gt_mask, r_prediction))
    # print(accuracy_score(gt_mask, r_prediction))
    # print(balanced_accuracy_score(gt_mask, r_prediction))


    #print(pred_pc)
    #print(pred_pc)

    gt_plot.setData(pos=gt_pc, size=0.2, color=(0.0, 0.1, 1.0, 0.5))
    pred_plot.setData(pos=pred_pc, size=0.5, color=(1.0, 0.1, 0.1, 0.5))
    human_plot.setData(pos=hum_pose, size=0.5, color=(0.5, 0.5, 0.5, 0.5))
    #print("raw error", np.linalg.norm((gt, pred_np)))
    #print("quant error", np.linalg.norm((np.round(gt), np.round(pred_np))))
    ctr += 1

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(1)

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




# while keep_alive:
#     try:
#         msg = socket.recv_pyobj()
#         pc = msg[0]
#         bev_img = point_cloud_2_birdseye(pc, res=0.2)
#         bev_img = bev_img / 255.
#         pred = conv_net(bev_img)
#         gt = generate_self_vector(pc)
#         #loss = tf.losses.mse(gt, pred)
#         gt = np.reshape(gt, (24, 1))
#         pred_np = np.reshape(pred, (24, 1))
#         gt_pc = pc[:, :-1] * np.logical_not(gt)
#         pred_pc = pc[:, :-1] * np.logical_not(pred_np)
#
#
#     except KeyboardInterrupt:
#         keep_alive = False
#         socket.close()
#         context.destroy()
