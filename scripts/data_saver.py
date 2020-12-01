import numpy as np
import zmq
from scripts.bev_plotter import point_cloud_2_birdseye, scale_to_255
EXP_NUMBER = 14
# ZMQ stuff
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:%s" % port)
keep_alive = True

# data stuff
bev = []
pc = []
human_pos = []
q = []

while keep_alive:
    try:
        msg = socket.recv_pyobj()
        img_buffer = msg[0]
        img = point_cloud_2_birdseye(img_buffer, res=0.2)
        bev.append(img)
        pc.append(img_buffer)
        human_pos.append(msg[1])
        q.append(msg[-1])
        print(len(bev), len(pc), len(human_pos), len(q))

    except KeyboardInterrupt:
        keep_alive = False
        print("Saving Data")
        bev_np = np.array(bev)
        pc_np = np.array(pc)
        human_pos_np = np.array(human_pos)
        q_np = np.array(q)
        print(bev_np.shape, pc_np.shape, human_pos_np.shape, q_np.shape)
        np.save("../data/bev" + str(EXP_NUMBER) + ".npy", bev_np)
        np.save("../data/pc_with_object_ids" + str(EXP_NUMBER) + ".npy", pc_np)
        np.save("../data/human_pos" + str(EXP_NUMBER) + ".npy", human_pos_np)
        np.save("../data/q" + str(EXP_NUMBER) + ".npy", q_np)
        socket.close()
        context.destroy()

