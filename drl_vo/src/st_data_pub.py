#!/usr/bin/env python
#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to publish Spatial and Temporal data of env.
#------------------------------------------------------------------------------
import numpy as np
import rospy
from st_msgs.msg import ST_data
# custom define messages:
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from pedsim_msgs.msg import TrackedPerson, TrackedPersons
from sensor_msgs.msg import LaserScan
import threading

# parameters:
NUM_TP = 10     # the number of timestamps
NUM_PEDS = 10 # the number of total pedestrians
PREDICTION_NUM = 5 # the number of dredict pose
SENSOR_RANGE = 5.0 #sensor range

data_lock = threading.Lock()

class STData:
    # Constructor
    def __init__(self):
        # initialize data:  

        self.scan = [] #np.zeros(720)
        self.scan_all = np.zeros(1080)
        self.scan_tmp = np.zeros(720)
        self.scan_all_tmp = np.zeros(1080)
        
        self.robot_node = np.zeros(5) # gx, gy, r, v_pref, theta
        self.temporal_edges = np.zeros(2)
        self.spatial_edges = []  #(PREDICTION_NUM + 1) * NUM_PEDS
        self.visible_masks = []
        self.detected_human_num = np.zeros(1)

        # initialize ROS objects
        self.ped_sub = rospy.Subscriber("/track_ped", TrackedPersons, self.ped_callback)
        self.goal_sub = rospy.Subscriber("/cnn_goal", Point, self.goal_callback)
        self.vel_sub = rospy.Subscriber("/mobile_base/commands/velocity", Twist, self.vel_callback)
        self.st_data_pub = rospy.Publisher('/st_data', ST_data, queue_size=1, latch=False)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # timer:
        self.rate = 20  # 20 Hz velocity controller
        self.ts_cnt = 0  # maximum 10 timesteps
        # initialize timer for controller update
        self.timer = rospy.Timer(rospy.Duration(1./self.rate), self.timer_callback)

        self.robot_radius = 0.3
        self.v_pref = 0.5
        self.theta = 2
        
    # Callback function for the pedestrian subscriber
    def ped_callback(self, trackPed_msg):
        data_lock.acquire()
        # get the pedstrain's position:
        # self.spatial_edges = np.zeros((PREDICTION_NUM + 1) * NUM_PEDS)
        time_step = 1.0 / NUM_TP
        self.spatial_edges.clear()
        self.visible_masks.clear()
        # (x, y) w.r.t robot_frame
        if(len(trackPed_msg.tracks) != 0):  # tracker results
            for ped in trackPed_msg.tracks:
                #ped_id = ped.track_id 
                x = ped.pose.pose.position.x
                y = ped.pose.pose.position.y
                vx = ped.twist.twist.linear.x
                vy = ped.twist.twist.linear.y
                self.spatial_edges.extend([x, y])
                self.visible_masks.append(self.dist_to_robot(x, y) < SENSOR_RANGE)
                for i in range(PREDICTION_NUM):
                    x = x + vx * time_step * (i + 1)
                    y = y + vy * time_step * (i + 1)
                    self.spatial_edges.extend([x, y])
        if self.visible_masks.count(True) == 0:
            self.detected_human_num[0] = 1
        else:
            self.detected_human_num[0] = self.visible_masks.count(True)
        data_lock.release()

    # Callback function for the current goal subscriber
    def goal_callback(self, goal_msg):
        # Cartesian coordinate:
        data_lock.acquire()
        self.robot_node[0] = goal_msg.x
        self.robot_node[1] = goal_msg.y
        self.robot_node[2] = self.robot_radius
        self.robot_node[3] = self.v_pref
        self.robot_node[4] = self.theta
        data_lock.release()

    # Callback function for the velocity subscriber
    # vx theta
    def vel_callback(self, vel_msg):
        data_lock.acquire()
        self.temporal_edges[0] = vel_msg.linear.x
        self.temporal_edges[1] = vel_msg.angular.z
        data_lock.release()
        
    def dist_to_robot(self, x, y):
        dist = np.linalg.norm([x, y])
        return dist
    
    def scan_callback(self, laserScan_msg):
        # get the laser scan data:
        data_lock.acquire()
        self.scan_tmp = np.zeros(720)
        self.scan_all_tmp = np.zeros(1080)
        scan_data = np.array(laserScan_msg.ranges, dtype=np.float32)
        scan_data[np.isnan(scan_data)] = 0.
        scan_data[np.isinf(scan_data)] = 0.
        self.scan_tmp = scan_data[180:900]
        self.scan_all_tmp = scan_data
        data_lock.release()

    def timer_callback(self, event):
        data_lock.acquire()
        self.scan.append(self.scan_tmp.tolist())
        self.scan_all = self.scan_all_tmp
        
        st_data = ST_data()
        st_data.robot_node = self.robot_node
        st_data.temporal_edges = self.temporal_edges
        st_data.spatial_edges = self.spatial_edges
        st_data.visible_masks = self.visible_masks
        st_data.detected_human_num = self.detected_human_num
        st_data.scan = [float(val) for sublist in self.scan for val in sublist]
        data_lock.release()

        #print(f"spatial data pub len {len(st_data.spatial_edges)}")

        self.st_data_pub.publish(st_data)

if __name__ == '__main__':
    try:
        rospy.init_node('st_data')
        STData()
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
