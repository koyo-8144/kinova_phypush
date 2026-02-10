#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def move_arm():
    rospy.init_node('send_joint_command')
    pub = rospy.Publisher('/my_gen3_lite/gen3_lite_joint_trajectory_controller/command', JointTrajectory, queue_size=10)
    
    # Wait for the publisher to connect
    rospy.sleep(1)

    traj = JointTrajectory()
    traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    point = JointTrajectoryPoint()
    point.positions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # Target angles in radians
    point.time_from_start = rospy.Duration(2.0)    # Reach goal in 2 seconds

    traj.points.append(point)
    
    pub.publish(traj)
    rospy.loginfo("Trajectory command sent!")

if __name__ == '__main__':
    try:
        move_arm()
    except rospy.ROSInterruptException:
        pass