#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage: python drl_vo_inference.py
#
# This script is the inference code of the DRL-VO policy.
#------------------------------------------------------------------------------

# import modules
#
import sys
import os

# ros:
import rospy
import tf
import numpy as np
import message_filters

# custom define messages:
from sensor_msgs.msg import LaserScan
from st_msgs.msg import ST_data
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from stable_baselines3 import PPO
from st_network import *


#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------


# for reproducibility, we seed the rng
#
policy_kwargs = dict(
    features_extractor_class=selfAttn_merge_SRNN,
    features_extractor_kwargs=dict(features_dim=256),
)

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------
class DrlInference:
    # Constructor
    def __init__(self):
        # initialize data:
        self.ped_pos = [] #np.ones((3, 20))*20.
        self.scan = [] #np.zeros((3, 720))
        self.goal = [] #np.zeros((3, 2))
        self.vx = 0
        self.wz = 0
        self.model = None

        # load model:
        model_file = rospy.get_param('~model_file', "./model/drl_vo.zip")
        self.model = PPO.load(model_file)
        print("Finish loading model.")

        # initialize ROS objects
        self.st_data_sub = rospy.Subscriber("/st_data", ST_data, self.st_data_callback)
        self.cmd_vel_pub = rospy.Publisher('/drl_cmd_vel', Twist, queue_size=10, latch=False)

        # d = {}
        # d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5,), dtype=np.float32)
        # d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # self.spatial_edge_dim = int(2*(self.predict_steps+1))
        # d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num * self.spatial_edge_dim, ), dtype=np.float32)
        # d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # self.observation_space = gym.spaces.Dict(d)


    # Callback function for the cnn_data subscriber
    def st_data_callback(self, st_data_msg):
        #self.ped_pos = st_data_msg.spatial_edges
        self.scan = st_data_msg.scan
        self.goal = st_data_msg.robot_node[ : 2]
        cmd_vel = Twist()

        # if the goal is close to the robot:
        if(np.linalg.norm(self.goal) <= 0.9):  # goal margin
            cmd_vel.linear.x = 0
            cmd_vel.angular.z = 0
        else:
            # MaxAbsScaler:
            v_min = -2
            v_max = 2
            # self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
            # self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1)

            # MaxAbsScaler:
            # temp = np.array(self.scan, dtype=np.float32)
            # scan_avg = np.zeros((20,80))
            # for n in range(10):
            #     scan_tmp = temp[n*720:(n+1)*720]
            #     for i in range(80):
            #         scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
            #         scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])

            # scan_avg = scan_avg.reshape(1600)
            # scan_avg_map = np.matlib.repmat(scan_avg,1,4)
            # self.scan = scan_avg_map.reshape(6400)
            # s_min = 0
            # s_max = 30
            # self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)

            # # goal:
            # # MaxAbsScaler:
            # g_min = -2
            # g_max = 2
            # goal_orignal = np.array(self.goal, dtype=np.float32)
            # self.goal = 2 * (goal_orignal - g_min) / (g_max - g_min) + (-1)
            #self.goal = self.goal.tolist()

            obs = {}

            obs['robot_node'] = torch.tensor(st_data_msg.robot_node).unsqueeze(0)
            obs['temporal_edges'] = torch.tensor(st_data_msg.temporal_edges).unsqueeze(0)
            obs['spatial_edges'] = torch.tensor(st_data_msg.spatial_edges)
            obs['visible_masks'] = torch.tensor(st_data_msg.visible_masks)
            obs['detected_human_num'] = torch.tensor(st_data_msg.detected_human_num)

            self.observation = obs


            #self.inference()
            action, _states = self.model.predict(self.observation)
            # calculate the goal velocity of the robot and send the command
            # MaxAbsScaler:
            vx_min = 0
            vx_max = 0.5
            vz_min = -2 # -0.7
            vz_max = 2 # 0.7
            cmd_vel.linear.x = (action[0] + 1) * (vx_max - vx_min) / 2 + vx_min
            cmd_vel.angular.z = (action[1] + 1) * (vz_max - vz_min) / 2 + vz_min
            print(f"cmd_vel x: {cmd_vel.linear.x}, cmd_vel z: {cmd_vel.angular.z}")

        if not np.isnan(cmd_vel.linear.x) and not np.isnan(cmd_vel.angular.z): # ensure data is valid
            self.cmd_vel_pub.publish(cmd_vel)


    #
    # end of function


# begin gracefully
#

if __name__ == '__main__':
    rospy.init_node('drl_inference')
    drl_infe = DrlInference()
    rospy.spin()

# end of file
