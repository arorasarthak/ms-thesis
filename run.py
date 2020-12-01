import numpy as np
import math
from pyrep import PyRep
from pyrep.backend import vrep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.vision_sensor import VisionSensor
from scripts.experiment import Experiment
import time
from scripts.rtde_helper import rtde_helper
import zmq
from collections import deque
from threading import Thread


def main():
    data = deque()
    ground_truth = deque()

    # RTDE Stuff
    rtd = rtde_helper('conf/record_configuration.xml', 'localhost', 30004, 125)

    # ZMQ stuff
    zmq_port = "5556"
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:%s" % zmq_port)

    # Launch the application with a scene file in headless mode
    pr = PyRep()
    pr.launch('/home/sarthak/PycharmProjects/research_ws/scenes/bill_new.ttt', headless=True)
    pr.set_simulation_timestep(0.1)
    expr = Experiment(5, 0)
    ur10 = UR10()
    pr.start()  # Start the simulation
    time.sleep(0.1)
    pr.step()

    velocities = [5, 5, 5, 5, 5, 5]

    ur10.set_joint_target_velocities(velocities)

    sim_step = 0
    STEPS = 150000
    buffer = deque()
    depth_cam = vrep.simGetObjectHandle('eyecam')  # Bill head cam
    # js_thread = Thread(target=sim_work, args=(rtd, pr, ur10))
    pr.step()
    time.sleep(5)
    data_store = []

    while True:#sim_step <= STEPS:

        q, _ = rtd.get_joint_states()
        # ur10.set_joint_target_positions(q)
        #q = np.random.randn(6,)
        #print(q)
        ur10.set_joint_target_positions(q)

        # raw distance readings
        base = expr.get_proximity_reading_from('Base')
        elbow = expr.get_proximity_reading_from('Elbow')
        tool = expr.get_proximity_reading_from('Tool')

        # 3d point readings with id
        base_pc, base_obj = expr.get_points_from('Base')
        elbow_pc, elbow_obj = expr.get_points_from('Elbow')
        tool_pc, tool_obj = expr.get_points_from('Tool')

        base_obs = np.hstack([base_pc, base_obj])
        elbow_obs = np.hstack([elbow_pc, elbow_obj])
        tool_obs = np.hstack([tool_pc, tool_obj])

        complete_obs = np.vstack([base_obs, elbow_obs, tool_obs])

        # human position
        hum_position = np.array(expr.get_human_position()).reshape(1, 3)

        if len(buffer) == 1:
            buffer.popleft()
        else:
            buffer.append(complete_obs)

        if len(buffer) > 0:
            buff_stack = np.vstack(buffer)
            socket.send_pyobj([buff_stack, hum_position, q])

        pr.step()
        sim_step += 1

    # np.save('data.npy', np.vstack(data_store))
    # np.save('gt.npy', np.vstack(ground_truth))
    rtd.stop()
    pr.stop()  # Stop the simulation
    pr.shutdown()  # Close the application
    import sys
    sys.exit()

if __name__ == '__main__':
    main()
