import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import math
import numpy as np

# Ros message 
import rclpy
from rclpy.node import Node
from nav_msgs import Odometry
from sensor_msgs import Imu

class kinematic():
	def __init__(self):
		super(kinematic, self).__init__()
		# Omni robot 
		self.r = 6.10
		self.d = 25.4
		# Mecanum robot in meter
		self.R = 0.05 
		self.l1 = 0.2
		self.l2 = 0.215
	
	def map(self,Input, min_input, max_input, min_output, max_output):
		value = ((Input - min_input)*(max_output-min_output)/(max_input - min_input) + min_output)
		return value
		
	def o_forward_kinematic(self, w1, w2, w3, w4, phi):
		vx = float((self.r/(4*0.7071))*(w1*(np.cos(phi) - np.sin(phi)) + w2*(np.cos(phi) + np.sin(phi))+ w3*(-np.cos(phi)+ np.sin(phi)) + w4*(-np.cos(phi) - np.sin(phi))))
		vy = float((self.r/(4*0.7071))*(w1*(np.cos(phi) + np.sin(phi)) + w2*(-np.cos(phi)-np.sin(phi)) + w3*(-np.cos(phi) - np.sin(phi)) + w4*(np.cos(phi) - np.sin(phi))))
		omega = float(self.r/(-1*8*0.7071*self.d)*(w1 + w2 + w3 +w4))
		return vx, vy, omega
		
	def m_forward_kinematic(self, w1, w2, w3, w4):
		vx = float((self.R/4)*(w1 - w2 - w3 + w4))
		vy = float((self.R/4)*(w1 + w2 - w3 - w4))
		omega = float((self.R/4)*(-w1/(self.l1 + self.l2) - w2/(self.l1 + self.l2) - w3/(self.l1 + self.l2) - w4/(self.l1 + self.l2)))
		return vx, vy, omega
		
	def euler_from_quaternion(self, x, y, z, w):
		t0 = 2.0 * (w * x + y * z)
		t1 = 1.0 - 2.0 * (x * x + y * y)
		roll = math.atan2(t0, t1)
		
		t2 = 2.0 * (w * y - z * x)
		t2 = 1.0 if t2 > 1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch = math.asin(t2)
		
		t3 = 2.0 * (w * z +x * y)
		t4 = 1.0 - 2.0*(y * y + z * z)
		yaw = math.atan2(t3, t4)
		
		return roll, pitch, yaw

class ros_node(Node, kinematic):
    def __init__(self):
        super(ros_node,self).__init__('sub_node')
        self.row = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.q = np.zeros(4)

        # ROS2 publisher and subscriber
        self.imu_subscriber = self.create_subscription(Imu, "/bno055/imu", self.imu_callback, 100)

    def imu_callback(self, msg):
        self.q[0] = msg.orientation.x
        self.q[1] = msg.orientation.y
        self.q[2] = msg.orientation.z
        self.q[3] = msg.orientation.w
        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(self.q[0], self.q[1], self.q[2], self.q[3])
        


    


def main(args=None):
    rclpy.init(args=args)
    node = ros_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main



