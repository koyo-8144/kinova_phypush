#!/usr/bin/env python3
import rospy
import numpy as np
import tf.transformations as tf_trans
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionIK, GetPositionIKRequest

try:
    import PyKDL as kdl
    from kdl_parser_py.urdf import treeFromParam
except ImportError:
    rospy.logerr("Missing KDL libraries. Run: sudo apt-get install ros-noetic-kdl-parser-py")

class JacobianSolver:
    def __init__(self):
        success, self.tree = treeFromParam("/my_gen3_lite/robot_description")
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

class Gen3LiteController:
    def __init__(self):
        rospy.init_node('gen3_lite_custom_controller')
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_joint_positions = []
        
        # Perfectly Symmetric Home (Candle Pose)
        self.home_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.j_solver = JacobianSolver()
        
        rospy.loginfo("Connecting to services...")
        rospy.wait_for_service('/my_gen3_lite/compute_fk')
        rospy.wait_for_service('/my_gen3_lite/compute_ik')
        self.fk_srv = rospy.ServiceProxy('/my_gen3_lite/compute_fk', GetPositionFK)
        self.ik_srv = rospy.ServiceProxy('/my_gen3_lite/compute_ik', GetPositionIK)
        self.state_sub = rospy.Subscriber('/my_gen3_lite/joint_states', JointState, self.state_callback)
        self.cmd_pub = rospy.Publisher('/my_gen3_lite/gen3_lite_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        
        while not self.current_joint_positions and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Controller Ready.")

    def state_callback(self, msg):
        temp = [0.0]*6
        found = 0
        for name in self.joint_names:
            if name in msg.name:
                temp[self.joint_names.index(name)] = msg.position[msg.name.index(name)]
                found += 1
        if found == 6: self.current_joint_positions = temp

    def check_pose_error(self, phase_name, desired_pose):
        """ Computes and prints the Cartesian error between current and desired EE pose """
        current_pose = self.get_fk()
        
        # Position Error
        pos_curr = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
        pos_des = np.array([desired_pose.position.x, desired_pose.position.y, desired_pose.position.z])
        pos_error = np.linalg.norm(pos_des - pos_curr)

        # Orientation Error (Angle difference in radians)
        q_curr = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        q_des = [desired_pose.orientation.x, desired_pose.orientation.y, desired_pose.orientation.z, desired_pose.orientation.w]
        
        # q_rel = q_des * inv(q_curr)
        q_rel = tf_trans.quaternion_multiply(q_des, tf_trans.quaternion_conjugate(q_curr))
        angle_error = 2 * np.arccos(np.clip(abs(q_rel[3]), -1.0, 1.0))

        print("\n" + "-"*50)
        print(f"DEBUG DIAGNOSTICS: Phase [{phase_name}] Finished")
        print(f"Position Error:    {pos_error*1000:.2f} mm")
        print(f"Orientation Error: {np.degrees(angle_error):.2f} degrees")
        if pos_error < 0.005 and angle_error < 0.05:
            print("STATUS: SUCCESS (Within Tolerance)")
        else:
            print("STATUS: WARNING (High Drift Detected)")
        print("-"*50 + "\n")

    def wait_for_arrival(self, target_joints, timeout=12.0, threshold=0.01):
        start_time = rospy.get_time()
        while (rospy.get_time() - start_time) < timeout:
            diff = np.linalg.norm(np.array(self.current_joint_positions) - np.array(target_joints))
            if diff < threshold:
                rospy.sleep(1.0) # Settle time
                return True
            rospy.sleep(0.1)
        rospy.logwarn("Target arrival timed out.")
        return False

    def send_joint_command(self, joints, duration):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        p = JointTrajectoryPoint()
        p.positions = joints
        p.time_from_start = rospy.Duration(duration)
        traj.points.append(p)
        self.cmd_pub.publish(traj)

    def go_to_home(self):
        rospy.loginfo("Action: Moving to Symmetric Home Position...")
        desired_pose = self.get_fk(self.home_positions) # Get Cartesian Goal for Home
        self.send_joint_command(self.home_positions, 8.0)
        self.wait_for_arrival(self.home_positions)
        self.check_pose_error("HOME", desired_pose)

    def get_fk(self, joints=None):
        req = GetPositionFKRequest()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["end_effector_link"]
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = joints if joints else self.current_joint_positions
        resp = self.fk_srv(req)
        return resp.pose_stamped[0].pose

    def get_ik(self, target_pose):
        req = GetPositionIKRequest()
        req.ik_request.group_name = "arm"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.pose = target_pose
        req.ik_request.avoid_collisions = True
        resp = self.ik_srv(req)
        if resp.error_code.val == 1:
            return [resp.solution.joint_state.position[resp.solution.joint_state.name.index(n)] for n in self.joint_names]
        return None

    def execute_full_maneuver(self, cube_pos, push_angle, push_dist, target_vel):
        # 0. Initial Homing
        self.go_to_home()

        # Geometry Calc
        cube_half_size = 0.025
        r = cube_half_size + 0.08 
        dx = r * np.cos(push_angle)
        dy = r * np.sin(push_angle)
        target_push_z = cube_pos[2] 
        start_pos = [cube_pos[0] + dx, cube_pos[1] + dy, target_push_z]
        
        # Look-at Orientation Logic
        look_dir = np.array(cube_pos) - np.array(start_pos)
        look_dir[2] = 0 
        look_dir /= (np.linalg.norm(look_dir) + 1e-6)
        z_axis = look_dir
        up_vec = np.array([0, 0, 1]) 
        y_axis = np.cross(z_axis, up_vec)
        y_axis /= (np.linalg.norm(y_axis) + 1e-6)
        x_axis = np.cross(y_axis, z_axis)
        R = np.eye(4)
        R[:3, 0] = x_axis; R[:3, 1] = y_axis; R[:3, 2] = z_axis
        target_quat = tf_trans.quaternion_from_matrix(R)

        
            
        # --- Phase 1: Intermediate ---
        inter_pose = Pose(Point(start_pos[0], start_pos[1], start_pos[2] + 0.12), Quaternion(*target_quat))
        rospy.loginfo("Action: Moving to Intermediate approach...")
        j_inter = self.get_ik(inter_pose)
        
        if j_inter: 
            self.send_joint_command(j_inter, 6.0)
            self.wait_for_arrival(j_inter)
            self.check_pose_error("INTERMEDIATE", inter_pose)
        else:
            rospy.logerr("IK failed for Phase 1 (Intermediate). Stopping.")
            return # EXIT MANEUVER

        # --- Phase 2: Contact Height ---
        start_pose = Pose(Point(start_pos[0], start_pos[1], start_pos[2]), Quaternion(*target_quat))
        rospy.loginfo("Action: Lowering to contact point...")
        j_start = self.get_ik(start_pose)
        
        if j_start: 
            self.send_joint_command(j_start, 5.0)
            self.wait_for_arrival(j_start)
            self.check_pose_error("PUSH_START", start_pose)
        else:
            rospy.logerr("IK failed for Phase 2 (Push Start). Stopping.")
            return # EXIT MANEUVER
        

        # --- Phase 3: Jacobian Push ---
        rospy.loginfo("Action: Starting Jacobian Constant Velocity Push...")
        rate = rospy.Rate(50); dt = 0.02; traveled = 0.0
        approach_dir_xy = look_dir[:2]

        while traveled < push_dist and not rospy.is_shutdown():
            curr_pose = self.get_fk()
            q_curr = np.array(self.current_joint_positions)
            v_xy = approach_dir_xy * target_vel
            v_z_corr = (target_push_z - curr_pose.position.z) * 5.0 
            v_linear = np.array([v_xy[0], v_xy[1], np.clip(v_z_corr, -0.05, 0.05)])
            
            curr_q = [curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w]
            q_err = tf_trans.quaternion_multiply(target_quat, tf_trans.quaternion_conjugate(curr_q))
            if q_err[3] < 0: q_err = -q_err
            v_angular = q_err[:3] * 6.0
            
            v_spatial = np.concatenate([v_linear, v_angular])
            J = self.j_solver.get_jacobian(q_curr)
            q_dot = J.T @ np.linalg.solve(J @ J.T + (0.02**2) * np.eye(6), v_spatial)
            
            q_next = q_curr + q_dot * dt
            self.send_joint_command(q_next.tolist(), dt * 2.0)
            traveled += target_vel * dt
            rate.sleep()

        # Check Final Push Position
        final_push_pose = Pose(Point(start_pos[0] + approach_dir_xy[0]*push_dist, 
                                     start_pos[1] + approach_dir_xy[1]*push_dist, 
                                     target_push_z), 
                               Quaternion(*target_quat))
        self.check_pose_error("PUSH_FINISHED", final_push_pose)

        rospy.loginfo("Maneuver Completed. Returning Home...")
        rospy.sleep(1.0)
        self.go_to_home()

if __name__ == '__main__':
    try:
        arm = Gen3LiteController()
        rospy.sleep(1.0)
        arm.execute_full_maneuver(
            cube_pos=[0.2, 0.0, 0.4], 
            push_angle=-np.pi/2, 
            push_dist=0.3, 
            target_vel=0.1
        )
    except rospy.ROSInterruptException:
        pass