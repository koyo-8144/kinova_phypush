#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from kortex_driver.msg import TwistCommand, Twist

try:
    import PyKDL as kdl
    from kdl_parser_py.urdf import treeFromParam
except ImportError:
    rospy.logerr("Missing KDL libraries. Run: sudo apt-get install ros-noetic-kdl-parser-py")

class JacobianSolver:
    def __init__(self, robot_description):
        success, self.tree = treeFromParam(robot_description)
        if not success:
            raise RuntimeError("Failed to extract KDL tree from URDF")
        self.chain = self.tree.getChain("base_link", "end_effector_link")
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.num_joints = self.chain.getNrOfJoints()

    def get_jacobian(self, joint_positions):
        kdl_joints = kdl.JntArray(self.num_joints)
        for i in range(self.num_joints):
            kdl_joints[i] = joint_positions[i]
        jacobian = kdl.Jacobian(self.num_joints)
        self.jac_solver.JntToJac(kdl_joints, jacobian)
        res = np.zeros((6, self.num_joints))
        for i in range(6):
            for j in range(self.num_joints):
                res[i, j] = jacobian[i, j]
        return res

class PushCube():
    def __init__(self):
        self.listener = tf.TransformListener()
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_joint_positions = []
        
        # Initialize MoveIt
        self.init_moveit()
        
        # Initialize Jacobian Solver for Phase 2
        self.j_solver = JacobianSolver("my_gen3_lite/robot_description")
        
        # FK Service setup
        rospy.loginfo("Waiting for FK service...")
        rospy.wait_for_service('/my_gen3_lite/compute_fk')
        self.fk_srv = rospy.ServiceProxy('/my_gen3_lite/compute_fk', GetPositionFK)
        
        # Direct Command Publisher for Velocity Loop (low latency)
        self.cmd_pub = rospy.Publisher(
            '/my_gen3_lite/gen3_lite_joint_trajectory_controller/command', 
            JointTrajectory, queue_size=1
        )
        
        # Subscribe to JointStates for Jacobian feedback
        rospy.Subscriber('/my_gen3_lite/joint_states', JointState, self.state_callback)
        
        self.twist_pub = rospy.Publisher('/my_gen3_lite/in/cartesian_velocity', TwistCommand, queue_size=1)

    def state_callback(self, msg):
        temp = [0.0]*6
        found = 0
        for name in self.joint_names:
            if name in msg.name:
                temp[self.joint_names.index(name)] = msg.position[msg.name.index(name)]
                found += 1
        if found == 6:
            self.current_joint_positions = temp
        
        # print("Current Joint Positions:", self.current_joint_positions)
        q_curr = np.array(self.current_joint_positions)
        J = self.j_solver.get_jacobian(q_curr)
        # print("Current Jacobian:\n", J)
        
        
    
    def get_current_ee_pose_via_fk(self):
        """ Replaces MoveIt get_current_pose with direct service call for speed """
        req = GetPositionFKRequest()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool_frame"]  # End-effector link
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = self.current_joint_positions
        try:
            resp = self.fk_srv(req)
            return resp.pose_stamped[0].pose
        except Exception as e:
            rospy.logerr(f"FK Service failed: {e}")
            return None

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
        rospy.loginfo("PHASE 1: Moving to start position")
        self.go_sp()
        rospy.loginfo("PHASE 2: Moving to pushset position")
        self.go_pushset_left()
        rospy.loginfo("PHASE 3: Pushing cube with velocity control")
        # direction_xy=[0.0, 1.0] moves the robot forward along the Y (Green) axis
        self.execute_velocity_push(direction_xy=[0.0, -1.0], push_dist=0.1, target_vel=0.03)
        rospy.sleep(0.1)
        rospy.loginfo("PHASE 4: Moving to start position")
        self.go_sp()
        
        
    def execute_velocity_push(self, direction_xy, push_dist, target_vel):
        """
        Updated Jacobian Push using Direct Cartesian Twist Commands.
        Ensures responsive movement on hardware by bypassing joint trajectory buffers.
        """
        # Ensure the arm group is not holding a previous MoveIt goal
        self.arm_group.stop()
        
        rate = rospy.Rate(50)  # 50 Hz control loop
        dt = 0.02
        traveled = 0.0
        
        # 1. Initial State Capture via FK Service
        start_pose = self.get_current_ee_pose_via_fk()
        if not start_pose: 
            rospy.logerr("[VEL PUSH] Could not get initial FK pose. Aborting.")
            return
        
        # Target orientation and height to maintain
        target_quat = [start_pose.orientation.x, start_pose.orientation.y, 
                       start_pose.orientation.z, start_pose.orientation.w]
        target_z = start_pose.position.z


        rospy.loginfo(f"[VEL PUSH] Starting. Direction: {direction_xy} | Target Z: {target_z:.4f}")

        step_count = 0
        while traveled < push_dist and not rospy.is_shutdown():
            curr_pose = self.get_current_ee_pose_via_fk()
            if not curr_pose: continue

            # --- 1. Linear Velocity Setup ---
            # XY: Use the provided direction (e.g., [0, 1] for Y-axis forward)
            v_xy = np.array(direction_xy) * target_vel
            
            # Z: Proportional station-keeping to prevent digging/drifting
            z_error = target_z - curr_pose.position.z
            v_z_corr = z_error * 4.0  # Gain tuned for hardware stability
            
            # --- 2. Orientation Control (Angular Velocity) ---
            curr_q = [curr_pose.orientation.x, curr_pose.orientation.y, 
                      curr_pose.orientation.z, curr_pose.orientation.w]
            
            # Quaternion error to maintain fixed orientation
            q_err = tf_trans.quaternion_multiply(target_quat, tf_trans.quaternion_conjugate(curr_q))
            if q_err[3] < 0: q_err = -q_err  # Shortest path check
            
            # Angular velocity command (Roll, Pitch, Yaw speeds)
            v_angular = q_err[:3] * 2.0  # Lower gain for smoother hardware rotation

            # --- 3. Construct and Publish Twist Command ---
            cmd = TwistCommand()
            cmd.reference_frame = 1  # Set to Task Frame (Base)
            
            cmd.twist.linear_x = v_xy[0]
            cmd.twist.linear_y = v_xy[1]
            cmd.twist.linear_z = np.clip(v_z_corr, -0.05, 0.05)
            
            cmd.twist.angular_x = v_angular[0]
            cmd.twist.angular_y = v_angular[1]
            cmd.twist.angular_z = v_angular[2]

            # --- DEBUG PRINT (Every 10 steps) ---
            if step_count % 10 == 0:
                print(f"\n--- TWIST STEP {step_count} ---")
                print(f"Traveled: {traveled:.4f} / {push_dist}")
                print(f"Current Pos: [{curr_pose.position.x:.4f}, {curr_pose.position.y:.4f}, {curr_pose.position.z:.4f}]")
                print(f"Commanded Twist: Linear[{cmd.twist.linear_x:.3f}, {cmd.twist.linear_y:.3f}, {cmd.twist.linear_z:.3f}]")

            self.twist_pub.publish(cmd)
            
            traveled += target_vel * dt
            step_count += 1
            rate.sleep()

        # 4. Mandatory Safety Stop: Publish zero velocity to halt the robot
        self.twist_pub.publish(TwistCommand())
        rospy.loginfo(f"[VEL PUSH] Finished. Total Traveled: {traveled:.4f}")

    ####### fixed position #######

    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)
        self.gripper_move(0.7)
    
    def go_pushset_right(self):
        self.arm_group.set_joint_value_target([-0.8352743769246702, -2.0099201652481007, 
                                               -0.9632429957285034, 0.4314691145319887, 
                                               -2.221564240237787, -2.498026460366695,])
        self.arm_group.go(wait=True)
        self.gripper_move(0.0)
    
    def go_pushset_left(self):
        self.arm_group.set_joint_value_target([-2.3817445913744564, 1.7951649854729823, 
                                               -0.9060777102977351, 0.7090221654433928, 
                                               -1.895770955656797, 1.301064980005055,])
        self.arm_group.go(wait=True)
        self.gripper_move(0.0)

    def gripper_move(self, width):
        self.gripper_group.set_joint_value_target({"right_finger_bottom_joint": width})
        self.gripper_group.go(wait=True)


    ####### utils #######
    
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
    grasp_planner_node = PushCube()
    grasp_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()