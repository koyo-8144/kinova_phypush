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
import matplotlib.pyplot as plt
import os

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
        
        # History parameters
        self.history_len = 100
        # Initialize velocity and acceleration buffers (100 samples, 6 components each)
        self._ee_vel_a_his = np.zeros((self.history_len, 6))
        self._ee_accel_a_his = np.zeros((self.history_len, 6))
        
        # Buffers for finite difference calculation
        self._prev_ee_vel_w = np.zeros(6) # Spatial velocity in world frame
        self._last_vel_time = None

        self.t_before = 15
        self.t_after = 35
        
        # Initialize a buffer for the moving average (e.g., window size of 5)
        self.smoothing_window = 10
        self._vel_buffer = []

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
        # self.go_pushset_left()
        # self.execute_velocity_push_lr(direction_xy=[0.0, -1.0], push_dist=0.5, target_vel=0.06)
        # self.go_pushset_right()
        # self.execute_velocity_push_lr(direction_xy=[0.0, 1.0], push_dist=0.5, target_vel=0.06)
        self.go_pushset_front()
        # start_pose = self.get_current_ee_pose_via_fk()
        # print("start_pose:", start_pose)
        self.execute_velocity_push_front(direction_xy=[1.0, 0.0], push_dist=0.3875, target_vel=2.25)
        rospy.loginfo("PHASE 3: Pushing cube with velocity control")
        # direction_xy=[0.0, 1.0] moves the robot forward along the Y (Green) axis
        
        rospy.sleep(0.1)

        # Identifies the window around peak impact
        start_t, _ = self.get_start_end_t(t_before=self.t_before, t_after=self.t_after)
        # Trigger the visualization
        rospy.loginfo("Visualizing Captured Histories...")
        self.visualize_push_history(start_t, window_len=self.t_after-self.t_before)

        rospy.loginfo("PHASE 4: Moving to start position")
        self.go_sp()
        
        
    def execute_velocity_push_lr(self, direction_xy, push_dist, target_vel):
        """
        Updated Jacobian Push using Direct Cartesian Twist Commands.
        Ensures responsive movement on hardware by bypassing joint trajectory buffers.
        """
        # Ensure the arm group is not holding a previous MoveIt goal
        self.arm_group.stop()
        
        # dt = 0.02 # 50 Hz control loop
        dt = 0.01 # 100 Hz control loop
        rate = rospy.Rate(1/dt)
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
        
        start_y = start_pose.position.y

        self._prev_joint_pos_for_accel = np.array(self.current_joint_positions)
        self._prev_ee_vel_w = np.zeros(6)

        rospy.loginfo(f"[VEL PUSH] Starting. Direction: {direction_xy} | Target Z: {target_z:.4f}")
        
        # Updated to match the provided gen3_lite_urdf exactly
        joint_limits = [
            (-2.69, 2.69), # Joint 1
            (-2.69, 2.69), # Joint 2
            (-2.69, 2.69), # Joint 3
            (-2.59, 2.59), # Joint 4
            (-2.57, 2.57), # Joint 5
            (-2.59, 2.59)  # Joint 6
        ]
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
            
            # Update history buffers exactly like Isaac Lab logic
            q_curr = np.array(self.current_joint_positions)
            self.update_history(curr_pose, q_curr, target_quat, dt)

            current_twist_linear = self._ee_vel_a_his[-1, :3]

            # --- DEBUG PRINT (Every 10 steps) ---
            if step_count % 10 == 0:
                print(f"\n--- TWIST STEP {step_count} ---")
                print(f"Traveled: {traveled:.4f} / {push_dist}")
                print(f"Current Pos: [{curr_pose.position.x:.4f}, {curr_pose.position.y:.4f}, {curr_pose.position.z:.4f}]")
                print(f"Commanded Twist: Linear[{cmd.twist.linear_x:.3f}, {cmd.twist.linear_y:.3f}, {cmd.twist.linear_z:.3f}]")
                print(f"Current Twist:   Linear[{current_twist_linear[0]:.3f}, {current_twist_linear[1]:.3f}, {current_twist_linear[2]:.3f}]")

                # Check for joint limits
                for idx, pos in enumerate(self.current_joint_positions):
                    low, high = joint_limits[idx]
                    if pos < low + 0.1 or pos > high - 0.1:
                        rospy.logwarn(f"!!! ALERT: Joint_{idx+1} near limit: {np.degrees(pos):.2f} deg")
                        
                # --- DISTANCE TO LIMIT CALCULATION ---
                for i, pos in enumerate(self.current_joint_positions):
                    low, high = joint_limits[i]
                    dist_to_low = abs(pos - low)
                    dist_to_high = abs(pos - high)
                    min_dist = min(dist_to_low, dist_to_high)
                    
                    if min_dist < 0.05: # Warn if within 3 degrees
                        side = "LOWER" if dist_to_low < dist_to_high else "UPPER"
                        rospy.logwarn(f"!!! Joint_{i+1} NEAR {side} LIMIT: {min_dist:.3f} rad left")

            self.twist_pub.publish(cmd)
            
            traveled += target_vel * dt
            # Inside execute_velocity_push_lr initialization
            # traveled = abs(curr_pose.position.y - start_y)
        
            step_count += 1
            rate.sleep()

        # 4. Mandatory Safety Stop: Publish zero velocity to halt the robot
        self.twist_pub.publish(TwistCommand())
        rospy.loginfo(f"[VEL PUSH] Finished. Total Traveled: {traveled:.4f}")


    # def execute_velocity_push_front(self, direction_xy, push_dist, target_vel):
    #     """
    #     Updated Jacobian Push using Direct Cartesian Twist Commands.
    #     Ensures responsive movement on hardware by bypassing joint trajectory buffers.
    #     """
    #     # Ensure the arm group is not holding a previous MoveIt goal
    #     self.arm_group.stop()
        
    #     # dt = 0.02 # 50 Hz control loop
    #     dt = 0.01 # 100 Hz control loop
    #     rate = rospy.Rate(1/dt)
    #     traveled = 0.0
        
    #     # 1. Initial State Capture via FK Service
    #     start_pose = self.get_current_ee_pose_via_fk()
    #     if not start_pose: 
    #         rospy.logerr("[VEL PUSH] Could not get initial FK pose. Aborting.")
    #         return
        
    #     # Target orientation and height to maintain
    #     target_quat = [start_pose.orientation.x, start_pose.orientation.y, 
    #                    start_pose.orientation.z, start_pose.orientation.w]
    #     target_z = start_pose.position.z
        
    #     start_x = start_pose.position.x

    #     self._prev_joint_pos_for_accel = np.array(self.current_joint_positions)
    #     self._prev_ee_vel_w = np.zeros(6)

    #     rospy.loginfo(f"[VEL PUSH] Starting. Direction: {direction_xy} | Target Z: {target_z:.4f}")
        
    #     # Updated to match the provided gen3_lite_urdf exactly
    #     joint_limits = [
    #         (-2.69, 2.69), # Joint 1
    #         (-2.69, 2.69), # Joint 2
    #         (-2.69, 2.69), # Joint 3
    #         (-2.59, 2.59), # Joint 4
    #         (-2.57, 2.57), # Joint 5
    #         (-2.59, 2.59)  # Joint 6
    #     ]
    #     step_count = 0
    #     while traveled < push_dist and not rospy.is_shutdown():
    #         curr_pose = self.get_current_ee_pose_via_fk()
    #         if not curr_pose: continue
            

    #         # --- 1. Linear Velocity Setup ---
    #         # XY: Use the provided direction (e.g., [0, 1] for Y-axis forward)
    #         v_xy = np.array(direction_xy) * target_vel
            
    #         # Z: Proportional station-keeping to prevent digging/drifting
    #         z_error = target_z - curr_pose.position.z
    #         v_z_corr = z_error * 4.0  # Gain tuned for hardware stability
            
    #         # --- 2. Orientation Control (Angular Velocity) ---
    #         curr_q = [curr_pose.orientation.x, curr_pose.orientation.y, 
    #                   curr_pose.orientation.z, curr_pose.orientation.w]
            
    #         # Quaternion error to maintain fixed orientation
    #         q_err = tf_trans.quaternion_multiply(target_quat, tf_trans.quaternion_conjugate(curr_q))
    #         if q_err[3] < 0: q_err = -q_err  # Shortest path check
            
    #         # Angular velocity command (Roll, Pitch, Yaw speeds)
    #         v_angular = q_err[:3] * 2.0  # Lower gain for smoother hardware rotation

    #         # --- 3. Construct and Publish Twist Command ---
    #         cmd = TwistCommand()
    #         cmd.reference_frame = 1  # Set to Task Frame (Base)
            
    #         cmd.twist.linear_x = v_xy[0]
    #         cmd.twist.linear_y = v_xy[1]
    #         cmd.twist.linear_z = np.clip(v_z_corr, -0.05, 0.05)
            
    #         cmd.twist.angular_x = v_angular[0]
    #         cmd.twist.angular_y = v_angular[1]
    #         cmd.twist.angular_z = v_angular[2]
            
    #         # Update history buffers exactly like Isaac Lab logic
    #         q_curr = np.array(self.current_joint_positions)
    #         self.update_history(curr_pose, q_curr, target_quat, dt)

    #         current_twist_linear = self._ee_vel_a_his[-1, :3]

    #         # --- DEBUG PRINT (Every 10 steps) ---
    #         if step_count % 10 == 0:
    #             print(f"\n--- TWIST STEP {step_count} ---")
    #             print(f"Traveled: {traveled:.4f} / {push_dist}")
    #             print(f"Current Pos: [{curr_pose.position.x:.4f}, {curr_pose.position.y:.4f}, {curr_pose.position.z:.4f}]")
    #             print(f"Commanded Twist: Linear[{cmd.twist.linear_x:.3f}, {cmd.twist.linear_y:.3f}, {cmd.twist.linear_z:.3f}]")
    #             print(f"Current Twist:   Linear[{current_twist_linear[0]:.3f}, {current_twist_linear[1]:.3f}, {current_twist_linear[2]:.3f}]")

    #             # Check for joint limits
    #             for idx, pos in enumerate(self.current_joint_positions):
    #                 low, high = joint_limits[idx]
    #                 if pos < low + 0.1 or pos > high - 0.1:
    #                     rospy.logwarn(f"!!! ALERT: Joint_{idx+1} near limit: {np.degrees(pos):.2f} deg")
                        
    #             # --- DISTANCE TO LIMIT CALCULATION ---
    #             for i, pos in enumerate(self.current_joint_positions):
    #                 low, high = joint_limits[i]
    #                 dist_to_low = abs(pos - low)
    #                 dist_to_high = abs(pos - high)
    #                 min_dist = min(dist_to_low, dist_to_high)
                    
    #                 if min_dist < 0.05: # Warn if within 3 degrees
    #                     side = "LOWER" if dist_to_low < dist_to_high else "UPPER"
    #                     rospy.logwarn(f"!!! Joint_{i+1} NEAR {side} LIMIT: {min_dist:.3f} rad left")

    #         self.twist_pub.publish(cmd)
            
    #         traveled += target_vel * dt
    #         # Inside execute_velocity_push_front initialization
    #         # traveled = abs(curr_pose.position.x - start_x)

    #         step_count += 1
    #         rate.sleep()

    #     # 4. Mandatory Safety Stop: Publish zero velocity to halt the robot
    #     self.twist_pub.publish(TwistCommand())
    #     rospy.loginfo(f"[VEL PUSH] Finished. Total Traveled: {traveled:.4f}")

    def execute_velocity_push_front(self, direction_xy, push_dist, target_vel):
        self.arm_group.stop()
        dt = 0.01 
        rate = rospy.Rate(1/dt)
        traveled = 0.0
        
        start_pose = self.get_current_ee_pose_via_fk()
        if not start_pose: return
        
        # 1. APPLY VELOCITY SCALING
        # Since actual (0.073) / commanded (0.03) ≈ 2.4, we scale the target down
        # v_scale = 0.4  # Scaling factor to hit the 0.03 m/s actual target
        v_scale = 1.0
        scaled_vel = target_vel * v_scale

        target_quat = [start_pose.orientation.x, start_pose.orientation.y, 
                       start_pose.orientation.z, start_pose.orientation.w]
        target_z = start_pose.position.z
        start_x = start_pose.position.x

        self._prev_joint_pos_for_accel = np.array(self.current_joint_positions)
        self._prev_ee_vel_w = np.zeros(6)

        step_count = 0
        # while traveled < push_dist and not rospy.is_shutdown():
        while step_count < self.history_len and not rospy.is_shutdown():
            curr_pose = self.get_current_ee_pose_via_fk()
            if not curr_pose: continue

            traveled = abs(curr_pose.position.x - start_x)

            # --- 2. SMOOTHED VELOCITY COMMANDS ---
            v_xy = np.array(direction_xy) * scaled_vel
            
            # Reduce Z-gain from 4.0 to 2.0 to stop the 0.10 m/s Z-spikes
            z_error = target_z - curr_pose.position.z
            v_z_corr = z_error * 2.0 
            
            cmd = TwistCommand()
            cmd.reference_frame = 1 
            cmd.twist.linear_x = v_xy[0]
            cmd.twist.linear_y = v_xy[1]
            cmd.twist.linear_z = np.clip(v_z_corr, -0.03, 0.03) # Tighter clip for stability
            
            # --- 3. ORIENTATION LOGIC ---
            curr_q = [curr_pose.orientation.x, curr_pose.orientation.y, 
                      curr_pose.orientation.z, curr_pose.orientation.w]
            q_err = tf_trans.quaternion_multiply(target_quat, tf_trans.quaternion_conjugate(curr_q))
            if q_err[3] < 0: q_err = -q_err 
            v_angular = q_err[:3] * 1.5 # Lowered angular gain for smoother travel

            self.update_history(curr_pose, np.array(self.current_joint_positions), target_quat, dt)
            self.twist_pub.publish(cmd)
            step_count += 1
            rate.sleep()

        self.twist_pub.publish(TwistCommand())

    def update_history(self, curr_pose, q_curr, target_quat_w, dt):
        # 1. Get current spatial velocity in WORLD frame
        # For Gen3 Lite, we get this from the Jacobian: V = J * q_dot
        # Note: Since you are in a velocity loop, you can use the q_dot you just calculated
        J = self.j_solver.get_jacobian(q_curr)
        # print("Jacobian:\n", J)
        q_dot_actual = (np.array(self.current_joint_positions) - self._prev_joint_pos_for_accel) / dt
        self._prev_joint_pos_for_accel = np.array(self.current_joint_positions)
        
        raw_V_w = J @ q_dot_actual # 6D Twist in world frame [v_x, v_y, v_z, w_x, w_y, w_z]
        
        # --- MOVING AVERAGE FILTER ---
        self._vel_buffer.append(raw_V_w)
        if len(self._vel_buffer) > self.smoothing_window:
            self._vel_buffer.pop(0)
        V_w = np.mean(self._vel_buffer, axis=0)
        
        # 2. Calculate Spatial Acceleration in WORLD frame (Finite Difference)
        V_dot_w = (V_w - self._prev_ee_vel_w) / dt
        self._prev_ee_vel_w = V_w.copy()
        
        # 3. Transform to Push Set Frame {a} (Rotation only)
        # Mirroring Isaac Lab: ee_lin_vel_pushset = quat_apply(push_set_inv_quat, ee_lin_vel_w)
        # ROS quat_conjugate/multiply uses [x,y,z,w]
        inv_q_a_w = tf_trans.quaternion_conjugate(target_quat_w)
        
        # Split into Linear/Angular for transformation
        v_w_lin = V_w[:3]
        v_w_ang = V_w[3:]
        a_w_lin = V_dot_w[:3]
        a_w_ang = V_dot_w[3:]
        
        # Transform linear and angular components
        # We need to rotate these vectors by the inverse of the push_set orientation
        def rotate_vec(q, v):
            # Helper to rotate vector by quaternion
            v_quat = list(v) + [0.0]
            return tf_trans.quaternion_multiply(tf_trans.quaternion_multiply(q, v_quat), tf_trans.quaternion_conjugate(q))[:3]

        # V_a = np.concatenate([rotate_vec(inv_q_a_w, v_w_lin), rotate_vec(inv_q_a_w, v_w_ang)])
        # V_dot_a = np.concatenate([rotate_vec(inv_q_a_w, a_w_lin), rotate_vec(inv_q_a_w, a_w_ang)])
        V_a = np.concatenate([v_w_lin, v_w_ang]) # Angular velocity is already in the correct frame since it's a pure rotation
        V_dot_a = np.concatenate([a_w_lin, a_w_ang]) # Angular acceleration is also in the correct frame

        
        # 4. Update Rolling History Buffers
        # Mirroring Isaac Lab: self._ee_vel_a_his = roll(..., shifts=-6, dims=1)
        self._ee_vel_a_his = np.roll(self._ee_vel_a_his, -1, axis=0)
        self._ee_vel_a_his[-1] = V_a
        
        self._ee_accel_a_his = np.roll(self._ee_accel_a_his, -1, axis=0)
        self._ee_accel_a_his[-1] = V_dot_a
        
    def get_start_end_t(self, t_before=15, t_after=35):
        """
        Identifies the window of interest around the peak end-effector acceleration.
        Mirroring the logic of finding t_peak and defining a window.
        """
        # 1. Get the linear acceleration history along the push axis
        # In your case, self._ee_accel_a_his is (100, 6) -> [ax, ay, az, wx, wy, wz]
        # acc_x_history = self._ee_accel_a_his[:, 1] 
        acc_x_history = self._ee_accel_a_his[:, 0]
        
        # 2. Find index of Max Acceleration Magnitude (t_peak)
        # Using absolute value to find the point of highest impact
        t_peak_index = np.argmax(np.abs(acc_x_history))
        
        # 3. Define Start and End indices
        # We use np.clip to ensure indices stay within the buffer bounds [0, 99]
        seq_len = self.history_len # which is 100
        
        start_t = np.clip(t_peak_index + t_before, 0, seq_len - 1)
        end_t   = np.clip(t_peak_index + t_after, 0, seq_len - 1)
        
        # Debugging print for the identified window
        rospy.loginfo(f"[ANALYSIS] Peak Acc at index: {t_peak_index}")
        rospy.loginfo(f"[ANALYSIS] Window defined: {start_t} to {end_t}")
        
        return int(start_t), int(end_t)


    def visualize_push_history(self, start_t, window_len=50):
        """
        Visualizes all 6 axes of the End Effector Velocity and Acceleration in the Local Tool Frame.
        Saves the output as PNG files in the /vis folder.
        """
        # 1. Create the /vis directory if it doesn't exist
        vis_dir = os.path.join(os.getcwd(), "vis")
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            rospy.loginfo(f"Created visualization directory at: {vis_dir}")

        # --- CONFIGURATION ---
        end_t = min(start_t + window_len, self.history_len)
        time_indices = np.arange(start_t, end_t)
        # Using 'Local' labels because of the inv_q_a_w transformation
        labels = ['Local Tool X', 'Local Tool Y', 'Local Tool Z', 'Local Tool Wx', 'Local Tool Wy', 'Local Tool Wz']
        units = ['m/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s']
        acc_units = ['m/s^2', 'm/s^2', 'm/s^2', 'rad/s^2', 'rad/s^2', 'rad/s^2']

        # Create two figures: one for Velocity, one for Acceleration
        fig_vel, axes_vel = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
        fig_acc, axes_acc = plt.subplots(6, 1, figsize=(10, 15), sharex=True)

        for i in range(6):
            # Velocity Data
            full_vel = self._ee_vel_a_his[:, i]
            win_vel = self._ee_vel_a_his[start_t:end_t, i]
            
            axes_vel[i].plot(full_vel, label='Full History', color='gray', alpha=0.3)
            axes_vel[i].plot(time_indices, win_vel, 'b-o', label='Push Window', markersize=3)
            axes_vel[i].set_ylabel(f"{units[i]}")
            axes_vel[i].set_title(f"Velocity: {labels[i]}")
            axes_vel[i].grid(True, alpha=0.3)

            # Acceleration Data
            full_acc = self._ee_accel_a_his[:, i]
            win_acc = self._ee_accel_a_his[start_t:end_t, i]
            
            axes_acc[i].plot(full_acc, label='Full History', color='gray', alpha=0.3)
            axes_acc[i].plot(time_indices, win_acc, 'r-o', label='Push Window', markersize=3)
            axes_acc[i].set_ylabel(f"{acc_units[i]}")
            axes_acc[i].set_title(f"Acceleration: {labels[i]}")
            axes_acc[i].grid(True, alpha=0.3)

        axes_vel[5].set_xlabel("Time Step")
        axes_acc[5].set_xlabel("Time Step")
        
        fig_vel.tight_layout()
        fig_acc.tight_layout()

        # 2. Save the figures
        timestamp = rospy.get_time()
        vel_path = os.path.join(vis_dir, f"velocity_history_{timestamp}.png")
        acc_path = os.path.join(vis_dir, f"acceleration_history_{timestamp}.png")
        
        fig_vel.savefig(vel_path)
        fig_acc.savefig(acc_path)
        
        rospy.loginfo(f"Saved visualization to {vis_dir}")
        
        # Optional: Still show them if you want
        # plt.show()
        
        
    ####### fixed position control #######

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
        # self.arm_group.set_joint_value_target([-2.3817445913744564, 1.7951649854729823, 
        #                                        -0.9060777102977351, 0.7090221654433928, 
        #                                        -1.895770955656797, 1.301064980005055,])
        self.arm_group.set_joint_value_target([-1.9565892991548273, 1.7936507120771634, 
                                               -0.9120517132550008, 1.0429641903278077, 
                                               -1.8944505603883348, 1.3008950703275082,])
        
        self.arm_group.go(wait=True)
        self.gripper_move(0.0)
        
    # def go_pushset_front(self):
    #     self.arm_group.set_joint_value_target([0.08763371251403841, -1.6131650892091853, 
    #                                            1.8311453570643887, 1.6748508242742544, 
    #                                            -1.944802680802355, 1.5674109164171879])
    #     self.arm_group.go(wait=True)
    #     self.gripper_move(0.0)
    
    def go_pushset_front(self):
        target_pose = Pose()
        target_pose.position.x = 0.4416
        target_pose.position.y = 0.0630
        target_pose.position.z = 0.0119
        
        # 2. Set orientation: Euler (0, 90deg, 0) or (0, 1.57, 0)
        q = tf_trans.quaternion_from_euler(0, 1.5708, 0)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]
        

        # 3. Use MoveIt to find the IK solution and move
        self.arm_group.set_pose_target(target_pose)
        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        
        # 4. Print the resulting joints so you can hardcode them later
        current_joints = self.arm_group.get_current_joint_values()
        # rospy.loginfo(f"Optimized Parallel Joints: {current_joints}")
        
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