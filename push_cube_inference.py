#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
import tf.transformations as tf_trans
import torch
import torch.nn as nn
import math
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

# ==========================================
# 1. TRANSFORMER MODEL ARCHITECTURE (V4) 
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PhysicsTransformerEstimator(nn.Module):
    def __init__(self, input_dim=2, d_model=32, nhead=4, num_encoder_layers=2, seq_len=20, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.net_force_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) 
        self.fric_force_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) 

        # These are the "Unexpected Keys" that caused the error:
        self.phys_net_proj = nn.Linear(d_model, 1)
        self.phys_fric_proj = nn.Linear(d_model, 1)

        # Spotlight for Mass
        self.q_mass = nn.Parameter(torch.randn(1, 1, d_model) * 2.0)
        self.mass_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

        # Spotlight for Friction
        self.q_fric = nn.Parameter(torch.randn(1, 1, d_model) * 2.0)
        self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, x_acc, x_vel):
        x = torch.cat([x_vel, x_acc], dim=-1) 
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        h_enc = self.transformer_encoder(z)

        feat_net = self.net_force_mlp(h_enc)
        feat_fric = self.fric_force_mlp(h_enc)

        # Spotlight Mass
        q_m = self.q_mass.expand(x.size(0), -1, -1)
        mass_ctx, _ = self.mass_attn(query=q_m, key=feat_net, value=feat_net)
        mass_pred = self.mass_pred_mlp(mass_ctx.squeeze(1))

        # Spotlight Mu
        q_f = self.q_fric.expand(x.size(0), -1, -1)
        fric_ctx, _ = self.fric_attn(query=q_f, key=feat_fric, value=feat_fric)
        mu_pred = self.mu_pred_mlp(fric_ctx.squeeze(1))

        return torch.cat([mass_pred, mu_pred], dim=-1)

# ==========================================
# 2. HELPER CLASSES & ROBOT LOGIC
# ==========================================

class JacobianSolver:
    def __init__(self, robot_description):
        success, self.tree = treeFromParam(robot_description)
        if not success:
            raise RuntimeError("Failed to extract KDL tree from URDF")
        self.chain = self.tree.getChain("base_link", "tool_frame")
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
        
        # --- Model Setup ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PhysicsTransformerEstimator(input_dim=2, d_model=32, seq_len=20)
        self.model_path = "/home/catkin_ws/src/kinova_phypush/trained_models/transformer_epoch500.pth"
        
        if os.path.exists(self.model_path):
            # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))
            self.model.to(self.device).eval()
            rospy.loginfo(f"✅ Model loaded: {self.model_path}")
        else:
            rospy.logwarn(f"❌ Model not found at {self.model_path}")

        # Initialize MoveIt
        self.init_moveit()
        self.j_solver = JacobianSolver("my_gen3_lite/robot_description")
        
        rospy.loginfo("Waiting for FK service...")
        rospy.wait_for_service('/my_gen3_lite/compute_fk')
        self.fk_srv = rospy.ServiceProxy('/my_gen3_lite/compute_fk', GetPositionFK)
        
        self.cmd_pub = rospy.Publisher('/my_gen3_lite/gen3_lite_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        rospy.Subscriber('/my_gen3_lite/joint_states', JointState, self.state_callback)
        self.twist_pub = rospy.Publisher('/my_gen3_lite/in/cartesian_velocity', TwistCommand, queue_size=1)
        
        self.history_len = 100
        self._ee_vel_a_his = np.zeros((self.history_len, 6))
        self._ee_accel_a_his = np.zeros((self.history_len, 6))
        self._prev_ee_vel_w = np.zeros(6)
        self.smoothing_window = 5
        self._vel_buffer = []

        self.t_before = 15
        self.t_after = 35

    def state_callback(self, msg):
        temp = [0.0]*6
        found = 0
        for name in self.joint_names:
            if name in msg.name:
                temp[self.joint_names.index(name)] = msg.position[msg.name.index(name)]
                found += 1
        if found == 6:
            self.current_joint_positions = temp
        
    def get_current_ee_pose_via_fk(self):
        req = GetPositionFKRequest()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool_frame"]
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = self.current_joint_positions
        try:
            resp = self.fk_srv(req)
            return resp.pose_stamped[0].pose
        except Exception as e:
            rospy.logerr(f"FK Service failed: {e}")
            return None

    def init_moveit(self):
        self.robot = moveit_commander.RobotCommander(robot_description="my_gen3_lite/robot_description")
        self.arm_group = moveit_commander.MoveGroupCommander("arm", robot_description="my_gen3_lite/robot_description", ns="/my_gen3_lite")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper", robot_description="my_gen3_lite/robot_description", ns="/my_gen3_lite")
        self.arm_group.set_max_acceleration_scaling_factor(1)
        self.arm_group.set_max_velocity_scaling_factor(1)
        self.arm_group.clear_path_constraints()

    def start(self):
        rospy.loginfo("PHASE 1: Moving to start position")
        self.go_sp()
        rospy.loginfo("PHASE 2: Moving to pushset position")
        self.go_pushset_front()
        
        rospy.loginfo("PHASE 3: Pushing cube with velocity control")
        self.execute_velocity_push_front(direction_xy=[1.0, 0.0], push_dist=0.1, target_vel=0.03)
        # self.execute_velocity_push_front(direction_xy=[1.0, 0.0], push_dist=0.3875, target_vel=2.25)
        
        rospy.sleep(0.5)

        # Find window based on impact
        start_t, _ = self.get_start_end_t(t_before=self.t_before, t_after=self.t_after)
        
        # Logic: We need exactly 20 frames for the transformer
        inference_start = start_t
        inference_end = start_t + 20

        rospy.loginfo("Visualizing Captured Histories...")
        self.visualize_push_history(start_t, window_len=self.t_after-self.t_before)

        # --- MODEL INFERENCE ---
        self.perform_inference(inference_start, inference_end)

        rospy.loginfo("PHASE 4: Returning to start")
        self.go_sp()

    def perform_inference(self, start_idx, end_idx):
        if end_idx > self.history_len:
            rospy.logwarn("Inference window exceeds buffer. Clipping.")
            end_idx = self.history_len
            start_idx = end_idx - 20

        # Extract X-Axis data (Linear Vel and Acc) for the window
        # Reshape to [Batch=1, Seq=20, Dim=1]
        vel_window = self._ee_vel_a_his[start_idx:end_idx, 0].reshape(1, 20, 1)
        acc_window = self._ee_accel_a_his[start_idx:end_idx, 0].reshape(1, 20, 1)

        # Convert to Torch Tensors
        vel_tensor = torch.from_numpy(vel_window).float().to(self.device)
        acc_tensor = torch.from_numpy(acc_window).float().to(self.device)

        rospy.loginfo("--- Executing Inference ---")
        with torch.no_grad():
            output = self.model(acc_tensor, vel_tensor)
        
        mass_est = output[0, 0].item()
        mu_est = output[0, 1].item()

        print("\n" + "="*40)
        print("    DEPLOYMENT PREDICTION RESULTS")
        print("="*40)
        print(f"  PREDICTED MASS: {mass_est:.4f} kg")
        print(f"  PREDICTED MU:   {mu_est:.4f}")
        print("="*40 + "\n")

    def execute_velocity_push_front(self, direction_xy, push_dist, target_vel):
        self.arm_group.stop()
        dt = 0.01 
        rate = rospy.Rate(1/dt)
        traveled = 0.0
        
        start_pose = self.get_current_ee_pose_via_fk()
        if not start_pose: return
        
        v_scale = 1.0
        scaled_vel = target_vel * v_scale

        target_quat = [start_pose.orientation.x, start_pose.orientation.y, 
                       start_pose.orientation.z, start_pose.orientation.w]
        target_z = start_pose.position.z
        start_x = start_pose.position.x

        self._prev_joint_pos_for_accel = np.array(self.current_joint_positions)
        self._prev_ee_vel_w = np.zeros(6)

        step_count = 0
        while step_count < self.history_len and not rospy.is_shutdown():
            curr_pose = self.get_current_ee_pose_via_fk()
            if not curr_pose: continue

            traveled = abs(curr_pose.position.x - start_x)

            v_xy = np.array(direction_xy) * scaled_vel
            z_error = target_z - curr_pose.position.z
            v_z_corr = z_error * 2.0 
            
            cmd = TwistCommand()
            cmd.reference_frame = 1 
            cmd.twist.linear_x = v_xy[0]
            cmd.twist.linear_y = v_xy[1]
            cmd.twist.linear_z = np.clip(v_z_corr, -0.03, 0.03)
            
            curr_q = [curr_pose.orientation.x, curr_pose.orientation.y, 
                      curr_pose.orientation.z, curr_pose.orientation.w]
            q_err = tf_trans.quaternion_multiply(target_quat, tf_trans.quaternion_conjugate(curr_q))
            if q_err[3] < 0: q_err = -q_err 
            v_angular = q_err[:3] * 1.5

            self.update_history(curr_pose, np.array(self.current_joint_positions), target_quat, dt)
            self.twist_pub.publish(cmd)
            step_count += 1
            rate.sleep()

        self.twist_pub.publish(TwistCommand())

    def update_history(self, curr_pose, q_curr, target_quat_w, dt):
        J = self.j_solver.get_jacobian(q_curr)
        q_dot_actual = (np.array(self.current_joint_positions) - self._prev_joint_pos_for_accel) / dt
        self._prev_joint_pos_for_accel = np.array(self.current_joint_positions)
        
        raw_V_w = J @ q_dot_actual
        
        self._vel_buffer.append(raw_V_w)
        if len(self._vel_buffer) > self.smoothing_window:
            self._vel_buffer.pop(0)
        V_w = np.mean(self._vel_buffer, axis=0)
        
        V_dot_w = (V_w - self._prev_ee_vel_w) / dt
        self._prev_ee_vel_w = V_w.copy()
        
        # Local transformation can be added here if needed, 
        # using world frame X for now as per training logic
        V_a = V_w
        V_dot_a = V_dot_w
        
        self._ee_vel_a_his = np.roll(self._ee_vel_a_his, -1, axis=0)
        self._ee_vel_a_his[-1] = V_a
        
        self._ee_accel_a_his = np.roll(self._ee_accel_a_his, -1, axis=0)
        self._ee_accel_a_his[-1] = V_dot_a
        
    def get_start_end_t(self, t_before=15, t_after=35):
        acc_x_history = self._ee_accel_a_his[:, 0]
        t_peak_index = np.argmax(np.abs(acc_x_history))
        
        start_t = np.clip(t_peak_index + t_before, 0, self.history_len - 1)
        end_t   = np.clip(t_peak_index + t_after, 0, self.history_len - 1)
        
        rospy.loginfo(f"[ANALYSIS] Peak Acc at index: {t_peak_index}")
        return int(start_t), int(end_t)

    def visualize_push_history(self, start_t, window_len=50):
        vis_dir = os.path.join(os.getcwd(), "vis")
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        end_t = min(start_t + window_len, self.history_len)
        time_indices = np.arange(start_t, end_t)
        labels = ['X', 'Y', 'Z', 'Wx', 'Wy', 'Wz']

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(self._ee_vel_a_his[start_t:end_t, 0], 'b-', label='Vel X')
        axes[0].set_title("Inference Window: Velocity X")
        axes[1].plot(self._ee_accel_a_his[start_t:end_t, 0], 'r-', label='Accel X')
        axes[1].set_title("Inference Window: Accel X")
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"inference_window_{rospy.get_time()}.png"))
        plt.close()

    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)
        self.gripper_move(0.7)

    def go_pushset_front(self):
        target_pose = Pose()
        target_pose.position.x = 0.4416
        target_pose.position.y = 0.1 # 0.0630
        target_pose.position.z= 0.0
        q = tf_trans.quaternion_from_euler(0, 1.5708, 0)
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = q
        self.arm_group.set_pose_target(target_pose)
        self.arm_group.go(wait=True)
        self.gripper_move(0.0)

    def gripper_move(self, width):
        self.gripper_group.set_joint_value_target({"right_finger_bottom_joint": width})
        self.gripper_group.go(wait=True)

def main():
    rospy.init_node('grasp_object', anonymous=True)
    grasp_planner_node = PushCube()
    grasp_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()