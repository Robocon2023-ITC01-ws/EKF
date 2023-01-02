import rclpy
from rclpy.node import Node
import numpy as np

# Ros2 message
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

import math

np.set_printoptions(precision=3,suppress=True)
class kinematic():
	def __init__(self):
		super(kinematic, self).__init__()
		# Omni robot 
		# Mecanum robot in meter
		self.R = 0.05 
		self.l1 = 0.2
		self.l2 = 0.215
	
	def map(self,Input, min_input, max_input, min_output, max_output):
		value = ((Input - min_input)*(max_output-min_output)/(max_input - min_input) + min_output)
		return value
		
	def o_forward_kinematic(self, w1, w2, w3, w4, phi):
		self.r = 0.061
		self.d = 0.254
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
		#if yaw < 0:
		#	yaw = self.map(yaw, -3.1399, -0.001, 3.1399, 6.2799)
		
		return roll, pitch, yaw
		
	def quaternion_from_euler(self, roll, pitch, yaw):
		qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
		qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
		qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		return qx, qy, qz, qw
		

class extended_kalman_filter(kinematic):
	def __init__(self):
		super(extended_kalman_filter, self).__init__()
		self.INPUT_NOISE = np.diag([1.0, 1.0, np.deg2rad(30.0)]) ** 2
		self.ENCODER_NOISE = np.diag([0.5, 0.5]) ** 2
		


	def cal_input(self, vx, vy, omega):
		U = np.array([[vx], [vy], [omega]])
		return U

	def motion_model(self, X, U, dt):
		yaw = X[2]

		F = np.array([[1.0, 0, 0, 0, 0],
                 	[0, 1.0, 0, 0, 0],
                 	[0, 0, 1.0, 0, 0],
                 	[0, 0, 0, 0, 0],
                 	[0, 0, 0, 0, 0]])
		B = np.array([[dt * math.cos(yaw), -dt * math.sin(yaw),0],
                 	[dt * math.sin(yaw), dt * math.cos(yaw),0],
                 	[0,0,0],
                 	[1,0,0],
                 	[0,1,0]])
		Xpred = F @ X + B @ U
		return Xpred

	def observation_model(self, X):
		H = np.array([
        [0, 0, 1, 0, 0]
    	])
		Z = H @ X
		return Z

	def jacob_f(self, X, U, dt):
		yaw = X[2, 0]
		vx = U[0, 0]
		vy = U[1, 0]
		jF = np.array([
        [1.0, 0, (-vx * dt * math.sin(yaw) - vy * dt * math.cos(yaw)), dt * math.cos(yaw), -dt * math.sin(yaw)],
        [0, 1.0, (vx * dt * math.cos(yaw) -vy * dt * math.sin(yaw)), dt * math.sin(yaw), dt * math.cos(yaw)],
        [0, 0, 1.0, 0, 0],
        [0, 0, 0, 1.0, 0],
        [0, 0, 0, 0, 1.0]
    	])
		return jF

	def jacob_h(self):
		jH = np.array([
        	[0, 0, 1, 0, 0]
    	])
		return jH

	def observation(self, XTrue, Xd, U, dt):
		XTrue = self.motion_model(XTrue, U, dt)

		# Add noise to sensor yaw
		Z = self.observation_model(XTrue) 

		# Add noise to input
		Ud = U
		Xd = self.motion_model(Xd, Ud, dt)

		return XTrue, Z, Xd, Ud

	def ekf_estimation(self, XEst, PEst, Z, U, dt):
		# Predict
		self.R = np.deg2rad(1.0)  # Observation x,y position covariance
		self.Q = np.diag([
			0.0144, # variance of location on x-axis
    		0.0169, # variance of location on y-axis
    		np.deg2rad(1.0), # variance of yaw angle
    		0, # variance of vx
    		0  # variance of vy
		]) ** 2
		XPred = self.motion_model(XEst, U, dt)
		jF = self.jacob_f(XEst, U, dt)
		PPred = jF @ PEst @ jF.T + self.Q

		# Update
		jH = self.jacob_h()
		ZPred = self.observation_model(XPred)
		y = Z - ZPred
		S = jH @ PPred @ jH.T + self.R
		K = PPred @ jH.T @ np.linalg.inv(S)
		XEst = XPred + K @ y
		PEst = (np.eye(len(XEst)) - K @ jH) @ PPred
		return XEst, PEst

class listen_node(Node, extended_kalman_filter, kinematic):
	def __init__(self):
		super(listen_node, self).__init__('sub_node')
		self.maxV = 100
		self.maxOmega = np.pi
		self.wheel_vel = np.zeros(4)
		self.command_vel = np.zeros(3)
		self.velocity_calback = np.zeros(4)
		self.phi = 0.0
		self.yaw = 0.0
		self.row = 0.0
		self.pitch = 0.0
		self.q = np.zeros(4)
		timer = 0.01
		self.dt = 0.01

		# Noise
		self.Xest = np.zeros((5,1))
		self.Xtrue = np.zeros((5,1))
		self.Pest = np.eye(5)
		self.Xdr = np.zeros((5,1))
		

		self.vel_subscriber = self.create_subscription(Float32MultiArray, 'feedback',self.velocity_callback, 100)
		self.imu_subscriber = self.create_subscription(Imu, "/bno055/imu", self.imu_callback, 100)
		self.compute_kalman_timer = self.create_timer(timer, self.compute_kalman)
		self.filted_odom = self.create_publisher(Odometry, 'odom', 10)
		self.filted_timer = self.create_timer(0.1, self.ekf_filted_data)
		

	def velocity_callback(self, msg):
		self.wheel_vel[0] = msg.data[0]
		self.wheel_vel[1] = msg.data[1]
		self.wheel_vel[2] = msg.data[2]
		self.wheel_vel[3] = msg.data[3]

		self.command_vel[0], self.command_vel[1], self.command_vel[2] = self.o_forward_kinematic(self.wheel_vel[0],self.wheel_vel[1], self.wheel_vel[2], self.wheel_vel[3],0.0)
		
	def imu_callback(self,msg):
		self.q[0] = msg.orientation.x
		self.q[1] = msg.orientation.y
		self.q[2] = msg.orientation.z
		self.q[3] = msg.orientation.w
		self.roll, self.pitch, self.yaw = self.euler_from_quaternion(self.q[0], self.q[1], self.q[2], self.q[3])
		self.Xtrue = np.array([
		[0],
		[0],
		[self.yaw],
		[0],
		[0]
		])
		

	def compute_kalman(self):
		U = self.cal_input(self.command_vel[0], self.command_vel[1], 0.0)
		self.Xtrue, z, self.Xdr, ud = self.observation(self.Xtrue, self.Xdr, U,self.dt)
		self.Xest, self.Pest = self.ekf_estimation(self.Xest, self.Pest,z, ud, self.dt)
	def ekf_filted_data(self):
		odom = Odometry()
		odom.header.frame_id = "odom"
		odom.child_frame_id = "base_link"
		odom.pose.pose.position.x = float(self.Xest[0,0])
		odom.pose.pose.position.y = -float(self.Xest[1,0])
		self.q[0], self.q[1], self.q[2], self.q[3] = self.quaternion_from_euler(0.0, 0.0, self.Xest[2,0])
		odom.pose.pose.orientation.x = self.q[0]
		odom.pose.pose.orientation.y = self.q[1]
		odom.pose.pose.orientation.z = self.q[2]
		odom.pose.pose.orientation.w = self.q[3]
		self.filted_odom.publish(odom)
		data = np.array([-self.Xest[1,0],self.Xest[0,0], self.Xest[2,0]])
		print(data)



def main(args=None):
	rclpy.init(args=args)
	sub_node = listen_node()
	rclpy.spin(sub_node)
	sub_node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
	


	

	

		
		
		
	
	
