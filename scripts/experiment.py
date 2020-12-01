import numpy as np
import math
from numpy import array as pt
from pyrep import PyRep
from pyrep.backend import vrep
from pyrep.robots.arms.ur10 import UR10
from collections import namedtuple, deque

Cylinder = namedtuple('Cylinder', 'radius height')
Sphere = namedtuple('Sphere', 'radius')


class Experiment:
    def __init__(self, duration=None, mode=None):
        self.dur = duration
        self.mode = mode
        # these names are defined in the VREP scene
        self.prox_sensor_prefix = "sensor_model"
        self.vis_sensor_prefix = "Vision_sensor#"
        self.human = 'Bill'
        self.robot = 'UR10'
        self.world = 'world_frame'
        self.floor = 'sim_floor'
        self.pedestal = 'UR10_pedestal'
        self.links = ['Base', 'Elbow', 'Tool']
        self.robot_links = ['UR10_link2_visible', 'UR10_link3_visible', 'UR10_link4_visible', 'UR10_link7_visible']
        self.link_handles = [vrep.simGetObjectHandle(each) for each in self.robot_links]

        # self.robot_shield = [self.cylinder(0.15, np.array([0, 0, 0.3])), self.cylinder(0.15, np.array([0, 0, 0.8])),
        #                      self.cylinder(0.15, np.array([0, 0, 0.8])), self.cylinder(0.15, np.array([0, 0, 0.3]))]
        self.link_margins = [(pt([0, 0, -0.30]), pt([0, 0, 0.30]), 0.3),
                             (pt([0, 0, 0]), pt([0, 0, 1.5]), 1),
                             (pt([0, 0, -0.5]), pt([0, 0, 0.5]), 0.3),
                             (pt([0, 0, -0.30]), pt([0, 0, 0.30]), 0.3)]

        self.link_cylinders = [Cylinder(0.15, 0.5), Cylinder(0.15, 0.8), Cylinder(0.15, 0.8), Sphere(0.3)]

        # safety stuff
        self.safety_margin = 0.6

        # handles
        self.human_handle = vrep.simGetObjectHandle(self.human)
        self.world_handle = vrep.simGetObjectHandle(self.world)
        self.robot_handle = vrep.simGetObjectHandle(self.robot)
        self.floor_handle = vrep.simGetObjectHandle(self.floor)
        self.pedestal_handle = vrep.simGetObjectHandle(self.pedestal)
        self.table_handle = vrep.simGetObjectHandle('customizableTable')
        self.table_handle1 = vrep.simGetObjectHandle('customizableTable0')

        # handles for self (not referring to the object)
        self.robot_handles = vrep.simGetObjectsInTree(self.robot_handle, vrep.sim_handle_all, 0)
        self.pedestal_elements = vrep.simGetObjectsInTree(self.pedestal_handle, vrep.sim_handle_all, 0)
        self.floor_elements = vrep.simGetObjectsInTree(self.floor_handle, vrep.sim_handle_all, 0)
        self.table = vrep.simGetObjectsInTree(self.table_handle, vrep.sim_handle_all, 0)
        self.table1 = vrep.simGetObjectsInTree(self.table_handle1, vrep.sim_handle_all, 0)

        self.environment = self.robot_handles + self.floor_elements + self.pedestal_elements + self.table + self.table1
        # print(len(self.environment))
        # handles for non-self
        self.non_self = vrep.simGetObjectsInTree(self.human_handle, vrep.sim_handle_all, 0)
        print(self.environment, self.non_self)

    def get_object_handle(self, object_name):
        return vrep.simGetObjectHandle(object_name)

    def get_human_position(self):
        return vrep.simGetObjectPosition(self.human_handle, self.world_handle)

    def get_object_position(self, object_handle):
        return vrep.simGetObjectPosition(object_handle, self.world_handle)

    def get_euc_distance(self, sensor_handle):
        reading = vrep.simReadProximitySensor(sensor_handle)
        euc_dist = 0.0
        object_name = None
        if reading[0] == 0:
            euc_dist = 0.0
            object_name = -1  # -1 means nothing is being seen
        elif reading[0] == 1:
            point = reading[2]
            euc_dist = np.linalg.norm(point) + np.random.uniform(0.002, -0.002)
            # object_name = vrep.simGetObjectName(reading[1][0])
            if reading[1][0] in self.environment:
                object_name = 0  # 0 means seeing it self

            # Didnt use "if else" for safety
            # ended up using another if statement to do an explicit check

            else:
                object_name = 1  # 0 means seeing non-self
        return euc_dist, object_name

    def get_object_name(self, sensor_handle):
        object_id = vrep.simReadVisionSensor(sensor_handle)

        if object_id[0] == 1:
            object_name = vrep.simGetObjectName(object_id[1][0])
        else:
            object_name = "None"
        return object_name

    def get_dist_transform(self, sensor_handle):
        raw_dist, object_name = self.get_euc_distance(sensor_handle)
        sensor_tf = vrep.simGetObjectMatrix(sensor_handle, self.world_handle)
        sensor_mat = np.zeros((4, 4))
        k = 0
        # raw dist lower bound goes here
        if raw_dist < 0.0:
            raw_dist = 0.0
        # populate the matrix
        if raw_dist != 0:
            for i in range(0, 3):
                sensor_mat[i] = sensor_tf[k:k + 4]
                k += 4
        sensor_mat[-1] = [0, 0, 0, 1]  # last row
        distance_mat = np.identity(4)
        distance_mat[2, 3] = raw_dist
        # print(sensor_mat)
        distance_mat = np.matmul(sensor_mat, distance_mat)
        dist_point = distance_mat[:, -1][:-1]  # last column till the last element is a point in (x,y,z) form
        # print(distance_mat)
        if dist_point[-1] == 0:
            dist_point = np.zeros(3)
        return distance_mat, dist_point, object_name

    def _gen_sensor_names(self, prefix, link):
        sensor_names = [None] * 8
        if link == 'Base':
            for i in range(0, 8):
                sensor_names[i] = prefix + str(i)
        elif link == 'Elbow':
            for i in range(8, 16):
                sensor_names[i - 8] = prefix + str(i)
        else:
            for i in range(16, 24):
                sensor_names[i - 16] = prefix + str(i)
        return sensor_names

    def get_prox_sensors(self, link):
        """
        :return: Sensor names of all the proximity sensors
        """
        return self._gen_sensor_names(self.prox_sensor_prefix, link)

    def get_vision_sensors(self, link):
        """
        :return:
        """
        return self._gen_sensor_names(self.vis_sensor_prefix, link)

    def get_proximity_reading_from(self, link):
        sensor_names = self.get_prox_sensors(link)
        distances = [0] * 8
        distances = np.zeros((8,), dtype=float)
        dist_euc = 0.0
        for s in range(0, 8):
            handle = self.get_object_handle(sensor_names[s])
            dist, _ = self.get_euc_distance(handle)
            distances[s] = dist
        return distances

    def get_objects_seenby(self, link):
        sensor_names = self.get_prox_sensors(link)
        objects = []
        for each in sensor_names:
            sensor_handle = self.get_object_handle(each)
            objects.append(self.get_object_name(sensor_handle))
        return objects

    def get_depth_reading(self):
        sensor_names = self.get_vision_sensors('Base')  # the experiment only uses base
        depth_array = np.zeros((8, 8, 8), dtype='float')
        for s in range(0, 8):
            handle = self.get_object_handle(sensor_names[s])
            depth_img = vrep.simGetVisionSensorDepthBuffer(handle, [8, 8], in_meters=True)
            depth_array[s] = depth_img
        return depth_array

    def get_points_from(self, link):
        sensor_names = self.get_prox_sensors(link)
        point_buffer = np.zeros((8, 3))  # this holds all the 3d points seen by the sensor
        object_names = np.zeros((8, 1))
        for s in range(0, 8):
            handle = self.get_object_handle(sensor_names[s])
            _, point_buffer[s], object_names[s] = self.get_dist_transform(handle)
        return point_buffer, object_names

    def get_inverse(self, transform_matrix):
        inverse = np.identity(4)  # prealloc
        rot = transform_matrix[0:-1, 0:-1]
        trans = transform_matrix[:, -1][:-1]
        inverse[0:-1, 0:-1] = np.transpose(rot)
        inverse[:, -1][:-1] = np.matmul(-np.transpose(rot), trans)
        return inverse

    def point_in_cylinders(self, point):
        """
        Assumes the point is with respect to the world
        :param point:
        :return:
        """
        vector = np.ones((4, 1))
        vector[:-1] = point.reshape(3, 1)
        link_mat = np.identity(4)
        for idx, link in enumerate(self.robot_links):
            link_handle = self.get_object_handle(link)
            link_mat[:-1, :] = np.array(vrep.simGetObjectMatrix(link_handle, self.world_handle)).reshape(3, 4)
            # link_mat_inverse = self.get_inverse(link_mat)
            new_point = np.matmul(link_mat, vector)
            new_point = new_point[:-1].flatten()
            if idx != 3:
                r = self.link_cylinders[idx].radius
                h = self.link_cylinders[idx].height
                if h / 2 > new_point[2] > -h / 2 and math.sqrt(new_point[0] ** 2 + new_point[1] ** 2) < r:
                    return True
                else:
                    return False
            else:
                r = self.link_cylinders[idx].radius
                if np.linalg.norm(new_point) < r:
                    return True
                else:
                    return False

    def filter_self(self, point_cloud):  # point_cloud dimension should be (24,3,1)
        # mat, link_mat = np.identity(4), np.identity(4)
        non_self = []
        # _self_ = []
        for each in point_cloud:
            if np.linalg.norm(each) != 0:
                if not self.point_in_cylinders(each):
                    non_self.append(each)
                    # blob = vrep.simGetObjectHandle('nonself_blob')
                    # vrep.simSetObjectPosition(blob, self.world_handle, list(each))
        return np.vstack(non_self)

    @staticmethod
    def zero_filter(point_cloud):
        # check for non-zero points
        if len(point_cloud) > 0:
            nonzero_idx = np.where(np.any(point_cloud, axis=1))
            pc_nonzero = point_cloud[nonzero_idx]
            ones = np.ones((pc_nonzero.shape[0], 1))
            pc_nonzero_ex = np.hstack([pc_nonzero, ones])
            return pc_nonzero, pc_nonzero_ex
        else:
            return np.array([0, 0, 0]), np.array([0, 0, 0, 1])


    def point_in_cylinder(self, cylinder, p):
        pass

    def get_link_tf_masks(self, point_cloud):
        mat = np.identity(4)
        links = [list(), list(), list(), list()]
        _, point_cloud = self.zero_filter(point_cloud)

        for point in point_cloud:
            for idx in range(len(self.link_handles)):
                l = self.link_handles[idx]
                world2link = np.array(vrep.simGetObjectMatrix(l, self.world_handle)).reshape(3, 4)
                mat[:-1, :-1] = world2link
                link2world = np.linalg.inv(mat)
                point_tx = np.matmul(link2world, point)
                c = self.link_cylinders[idx]
                if self.point_in_cylinder(c, point_tx):
                    links[idx].append(True)
                else:
                    links[idx].append(False)


    def link_transformer(self, point_cloud):
        _, pc = self.zero_filter(point_cloud)
        pc1, pc2, pc3, pc4 = [], [], [], []
        pass
