#!/usr/bin/env python 3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionIK, GetPositionIKRequest
from kortex_driver.srv import SendTwistCommand, SendTwistCommandRequest
from kortex_driver.msg import TwistCommand, Twist
import numpy as np

try:
    import PyKDL as kdl
    from kdl_parser_py.urdf import treeFromParam
except ImportError:
    rospy.logerr("Missing KDL libraries. Run: sudo apt-get install ros-noetic-kdl-parser-py")

class JacobianSolver:
    def __init__(self):
        # 1. Get URDF from parameter server
        success, self.tree = treeFromParam("/my_gen3_lite/robot_description")
        if not success:
            raise RuntimeError("Failed to extract KDL tree from URDF")

        # 2. Define chain
        self.chain = self.tree.getChain("base_link", "end_effector_link")
        
        # 3. Correct Solver Name: ChainJntToJacSolver
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.num_joints = self.chain.getNrOfJoints()

    def get_jacobian(self, joint_positions):
        kdl_joints = kdl.JntArray(self.num_joints)
        for i in range(self.num_joints):
            kdl_joints[i] = joint_positions[i]

        jacobian = kdl.Jacobian(self.num_joints)
        self.jac_solver.JntToJac(kdl_joints, jacobian)
        
        # Convert to numpy matrix
        res = np.zeros((6, self.num_joints))
        for i in range(6):
            for j in range(self.num_joints):
                res[i, j] = jacobian[i, j]
        return res

class Gen3LiteController:
    def __init__(self):
        rospy.init_node('gen3_lite_custom_controller')

        # List of joints we care about (ignoring the finger for now)
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_joint_positions = []
        self.last_print_time = rospy.Time.now()
        
        self.j_solver = JacobianSolver()
        
        rospy.loginfo("Waiting for MoveIt services...")
        rospy.wait_for_service('/my_gen3_lite/compute_fk')
        rospy.wait_for_service('/my_gen3_lite/compute_ik')
        self.fk_srv = rospy.ServiceProxy('/my_gen3_lite/compute_fk', GetPositionFK)
        self.ik_srv = rospy.ServiceProxy('/my_gen3_lite/compute_ik', GetPositionIK)

        # Subscriber: Listen to the robot's current state
        self.state_sub = rospy.Subscriber('/my_gen3_lite/joint_states', JointState, self.state_callback)

        # Publisher: Send commands to the robot
        self.cmd_pub = rospy.Publisher('/my_gen3_lite/gen3_lite_joint_trajectory_controller/command', JointTrajectory, queue_size=10)
        
        # Wait for the publisher to connect
        rospy.sleep(1)

        rospy.loginfo("Waiting for joint states...")
        while not self.current_joint_positions and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Joint states received. Controller ready.")
        


    
    def state_callback(self, msg):
        """
        Processes and prints the full state of the arm
        """
        # Store positions for the controller logic
        temp_positions = []
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                temp_positions.append(msg.position[idx])
        
        if len(temp_positions) == 6:
            self.current_joint_positions = temp_positions

        # --- THROTTLED PRINTING (Every 1.0 second) ---
        now = rospy.Time.now()
        if (now - self.last_print_time).to_sec() > 1.0:
            print("\n--- Current Arm State ---")
            print("{:<10} | {:<10} | {:<10} | {:<10}".format("Joint", "Pos (rad)", "Vel", "Effort"))
            print("-" * 50)
            
            for i, name in enumerate(msg.name):
                # Print all joints including the finger
                pos = msg.position[i]
                vel = msg.velocity[i]
                eff = msg.effort[i]
                print("{:<10} | {:>10.4f} | {:>10.4f} | {:>10.4f}".format(name, pos, vel, eff))
            
            self.last_print_time = now
        
        jac = self.j_solver.get_jacobian(self.current_joint_positions)
        print("Jacobian:\n", jac)

    def get_fk(self):
        """ Returns current Cartesian Pose and prints it """
        req = GetPositionFKRequest()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["end_effector_link"]
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = self.current_joint_positions
        
        try:
            resp = self.fk_srv(req)
            pose = resp.pose_stamped[0].pose
            
            print("\n" + "="*40)
            print("FORWARD KINEMATICS RESULT")
            print("-"*40)
            print("Position:    X: {:.4f}, Y: {:.4f}, Z: {:.4f}".format(pose.position.x, pose.position.y, pose.position.z))
            print("Orientation: X: {:.4f}, Y: {:.4f}, Z: {:.4f}, W: {:.4f}".format(
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
            print("="*40)
            return pose
        except Exception as e:
            rospy.logerr("FK Failed: %s" % e)

    def get_ik(self, target_pose):
        """ Returns Joint angles for a target pose and prints them """
        req = GetPositionIKRequest()
        req.ik_request.group_name = "arm"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.pose = target_pose
        req.ik_request.avoid_collisions = True

        try:
            resp = self.ik_srv(req)
            if resp.error_code.val == 1:
                # Filter solution for just our 6 joints
                full_sol = resp.solution.joint_state
                target_joints = []
                for name in self.joint_names:
                    idx = full_sol.name.index(name)
                    target_joints.append(full_sol.position[idx])
                
                print("\n" + "="*40)
                print("INVERSE KINEMATICS RESULT")
                print("-"*40)
                for name, pos in zip(self.joint_names, target_joints):
                    print("{}: {:.4f} rad".format(name, pos))
                print("="*40)
                return target_joints
            else:
                rospy.logwarn("IK Failed. Error Code: %s" % resp.error_code.val)
                return None
        except Exception as e:
            rospy.logerr("IK Failed: %s" % e)


    def move_to_position(self, target_positions, duration=3.0):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start = rospy.Duration(duration)
        traj.points.append(point)
        self.cmd_pub.publish(traj)

if __name__ == '__main__':
    try:
        arm = Gen3LiteController()
        rospy.sleep(1.0)

        # 1. Test FK
        current_pose = arm.get_fk()

        # 2. Test IK (Ask it to move 5cm higher in Z)
        if current_pose:
            target_pose = current_pose
            target_pose.position.z += 0.05 
            joint_goal = arm.get_ik(target_pose)
            
            # 3. Move if IK found a solution
            if joint_goal:
                arm.move_to_position(joint_goal)
    

        rospy.spin()
    except rospy.ROSInterruptException:
        pass