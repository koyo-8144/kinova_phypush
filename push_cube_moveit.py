#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose

import time


class GraspObjectNode():
    def __init__(self):
        self.listener = tf.TransformListener()
        self._rate = rospy.Rate(10)  # Broadcast at 10 Hz

        self.init_moveit()


    

    def init_moveit(self):
        # Initialize planning group
        self.robot = moveit_commander.RobotCommander(robot_description="my_gen3_lite/robot_description")
        self.arm_group = moveit_commander.MoveGroupCommander(
            "arm", 
            robot_description="my_gen3_lite/robot_description", 
            ns="/my_gen3_lite"
        )
        self.gripper_group = moveit_commander.MoveGroupCommander(
            "gripper", 
            robot_description="my_gen3_lite/robot_description",
            ns="/my_gen3_lite"
        )

        # Set robot arm's speed and acceleration
        self.arm_group.set_max_acceleration_scaling_factor(1)
        self.arm_group.set_max_velocity_scaling_factor(1)
        self.arm_group.clear_path_constraints()

        # We can get the name of the reference frame for this robot:
        planning_frame = self.arm_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        ee_link = self.arm_group.get_end_effector_link()
        print("============ End effector link: %s" % ee_link)

        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", group_names)

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())

        print("============ Printing active joints in arm group")
        print(self.arm_group.get_active_joints())

        print("============ Printing joints in arm group")
        print(self.arm_group.get_joints())

        print("============ Printing active joints in gripper group")
        print(self.gripper_group.get_active_joints())

        print("============ Printing joints in gripper group")
        print(self.gripper_group.get_joints())
        # breakpoint()

        # Targe grasp position
        self.target_pose = PoseStamped()
        self.target_pose_update = True

        # We can get the name of the reference frame for this robot:
        planning_frame = self.arm_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = self.arm_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", self.robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")


    ####### push cube #######

    def start(self):
        rospy.loginfo("Moving to start position")
        self.go_sp()
        rospy.loginfo("Moving to push set position")
        self.go_pushset()


    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)

        self.gripper_move(0.7)
    
    def go_pushset(self):
        self.arm_group.set_joint_value_target([-0.8352743769246702, -2.0099201652481007, 
                                               -0.9632429957285034, 0.4314691145319887, 
                                               -2.221564240237787, -2.498026460366695,])
        self.arm_group.go(wait=True)

        self.gripper_move(0.0)
        


    ####### utils #######
    
    def gripper_move(self, width):
        joint_state_msg = rospy.wait_for_message("/my_gen3_lite/joint_states", JointState, timeout=1.0)
        # print("joint_state_msg: ", joint_state_msg)

        # Find indices of the gripper joints
        right_finger_bottom_index = joint_state_msg.name.index('right_finger_bottom_joint')
        # print("right finger bottom index: ", right_finger_bottom_index)

        # self.gripper_group.set_joint_value_target([width])
        self.gripper_group.set_joint_value_target(
            {"right_finger_bottom_joint": width})
        self.gripper_group.go()

    def construct_rot_matrix_homogeneous_transform(self, translation, quaternion):
        # Convert quaternion to rotation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
        # print("Rotation Matrix:")
        # print(rotation_matrix)

        # Construct the 4x4 homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation

        return T
    
    def construct_homogeneous_transform(self, translation, rotation_matrix):
        # Construct the 4x4 homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation

        return T
    
    def get_frame1_to_frame2(self, frame1, frame2):
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.listener.lookupTransform(frame1, frame2, rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T = self.construct_rot_matrix_homogeneous_transform(trans, rot)

        return T
    
    def transformation_to_pose(self, T):
        # Extract translation (position)
        translation = T[:3, 3]  # [x, y, z]

        # Extract rotation matrix
        rotation_matrix = T[:3, :3]

        # Convert rotation matrix to quaternion
        quaternion = tf_trans.quaternion_from_matrix(T)

        # Create a Pose message
        pose = Pose()
        pose.position.x = translation[0]-0.03 # translation[0]
        pose.position.y = translation[1]+0.065 # translation[1]
        pose.position.z = -0.06 # -0.04731002  # translation[2]

        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose

    def publish_tf(self, T, parent_frame, child_frame):
        """ Continuously broadcast the transformation """
        # Extract translation (last column of the first 3 rows)
        translation = T[:3, 3]
        # Extract rotation matrix (upper-left 3×3)
        rotation_matrix = T[:3, :3]
        # Convert rotation matrix to quaternion
        quaternion = tf.transformations.quaternion_from_matrix(T)

        rospy.loginfo(f"Publishing transformation: {parent_frame} → {child_frame}")
        rospy.loginfo(f"Translation: {translation}")
        rospy.loginfo(f"Quaternion: {quaternion}")

        while not rospy.is_shutdown():
            # Broadcast transform
            self.broadcaster.sendTransform(
                translation,    # Position (x, y, z)
                quaternion,     # Orientation (x, y, z, w)
                rospy.Time.now(),
                child_frame,  # Child frame
                parent_frame      # Parent frame
            )

            # Publish completion signal
            self.tf_done_pub.publish(True)
            rospy.loginfo("Object found and completion signal sent.")

            self.rate.sleep()



        ####### MQTT #######

    


def main():
    rospy.init_node('grasp_object', anonymous=True)
    grasp_planner_node = GraspObjectNode()
    grasp_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()