#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

from scipy.spatial import KDTree
import numpy as np

import math


'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        rospy.loginfo("init_node complete.")

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        # rospy.spin()
        self.loop()
   
    def loop(self): 
        rospy.loginfo("Loop called. self.pose: %s", self.pose)
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                rospy.loginfo("Entered if in the loop. len(self.base_waypoints.waypoints): %s", len(self.base_waypoints.waypoints))
                # Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        if not self.waypoint_tree:
            return 0

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self, closest_idx):
        final_lane = self.generate_lane()
        # rospy.loginfo("Lane: %s", final_lane)
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()

        colosest_idx = self.get_closest_waypoint_idx()
        farthest_idx = colosest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[colosest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else: 
            lane.waypoints = self.decelerate_waypoints(base_waypoints, colosest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        if self.stopline_wp_idx == -1:
            return waypoints
        
        deccel_wps = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)  # 2 waypoints behind the line, so front of the car stops at the line
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0
            
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            deccel_wps.append(p)

        return deccel_wps

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, wps):
        rospy.loginfo("waypoint_cb callback called.")
        self.base_waypoints = wps
        
        if not self.waypoints_2d: 
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in wps.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            rospy.loginfo("self.waypoints_2 set.")

    def traffic_cb(self, msg):
        rospy.loginfo("wp_updated.traffic_cb() called with msg: %s", msg)
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint_idx, velocity):
        waypoints[waypoint_idx].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
