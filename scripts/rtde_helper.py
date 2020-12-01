import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


class rtde_helper:

    def __init__(self, configfile, hostname, port, frequency):
        self.configfile = configfile
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.conf = rtde_config.ConfigFile(self.configfile)
        self.output_names, self.output_types = self.conf.get_recipe('out')
        self.con = rtde.RTDE(self.hostname, self.port)
        self.con.connect()
        self.con.get_controller_version()
        self.con.send_output_setup(self.output_names, self.output_types, self.frequency)
        self.con.send_start()

    def get_joint_states(self):
        state = self.con.receive()
        if state is not None:
            q = state.__dict__['actual_q']  # for joint angles
            qd = state.__dict__['actual_qd']  # for joint velocities
            q[1], q[3] = q[1] + 1.57079, q[3] + 1.57079
        else:
            q, qd = [None]*6, [None]*6
        return q, qd

    def stop(self):
        self.con.send_pause()
        self.con.disconnect()
