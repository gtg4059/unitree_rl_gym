from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch
import os

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config_v1 import Config
from TrtPolicyRunner import TrtPolicyRunner
# from multiprocessing import Process, shared_memory, Array
# from multiprocessing import shared_memory, Array, Lock
# from robot_control.robot_hand_inspire import Inspire_Controller

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.robot_data = []
        # Initialize the policy network
        self.policy_run = TrtPolicyRunner(config.policy_run, num_layers=1, hidden_size=64)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.counter = 0
        self.start_time = None
        
        # Pre-compute indices for faster data extraction
        self.leg_motor_indices = np.array(config.leg_joint2motor_idx, dtype=np.int32)
        self.arm_motor_indices = np.array(config.arm_waist_joint2motor_idx, dtype=np.int32)
        self.all_motor_indices = np.concatenate([self.leg_motor_indices, self.arm_motor_indices])
        self.leg_offset = len(config.leg_joint2motor_idx)
        
        # Pre-allocate arrays for observation computation
        self.qj_obs = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(config.num_actions, dtype=np.float32)
        self.gravity_orientation = np.zeros(3, dtype=np.float32)
        self.ang_vel_scaled = np.zeros(3, dtype=np.float32)
        self.cmd_abs = np.zeros(3, dtype=np.float32)
        
        # Pre-compute scaled arrays
        self.dof_pos_scale_arr = np.array(config.dof_pos_scale, dtype=np.float32)
        self.dof_vel_scale_arr = np.array(config.dof_vel_scale, dtype=np.float32)
        self.ang_vel_scale_val = config.ang_vel_scale
        self.cmd_scale_max_cmd = config.cmd_scale * config.max_cmd

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
        
        # RecurrentThread for stable control loop timing
        self.lowCmdWriteThreadPtr = None
        self.running = False

    
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_default_angles), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        # left_hand_array = Array('d', 6, lock = True)          # [input]
        # right_hand_array = Array('d', 6, lock = True)         # [input]
        # left_hand_array[:] = np.array([0,0,0,0,0,0], dtype=np.float32)
        # right_hand_array[:] = np.array([0,0,0,0,0,0], dtype=np.float32)
        # dual_hand_data_lock = Lock()
        # dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        # dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        # hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def LowCmdWrite(self):
        """
        RecurrentThread에서 실행되는 제어 루프
        타이밍 제어는 RecurrentThread가 담당하므로 여기서는 제어 로직만 수행
        """
        if not self.running:
            return
            
        self.counter += 1

        # Get the current joint position and velocity using NumPy indexing (much faster than loops)
        motor_states = self.low_state.motor_state
        # Extract leg joint data
        for idx, motor_idx in enumerate(self.leg_motor_indices):
            self.qj[idx] = motor_states[motor_idx].q
            self.dqj[idx] = motor_states[motor_idx].dq
        # Extract arm/waist joint data
        for idx, motor_idx in enumerate(self.arm_motor_indices):
            pos_idx = idx + self.leg_offset
            self.qj[pos_idx] = motor_states[motor_idx].q
            self.dqj[pos_idx] = motor_states[motor_idx].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        imu_gyro = self.low_state.imu_state.gyroscope
        self.ang_vel_scaled[0] = imu_gyro[0]
        self.ang_vel_scaled[1] = imu_gyro[1]
        self.ang_vel_scaled[2] = imu_gyro[2]

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = motor_states[self.arm_motor_indices[0]].q
            waist_yaw_omega = motor_states[self.arm_motor_indices[0]].dq
            quat, ang_vel_temp = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=self.ang_vel_scaled)
            self.ang_vel_scaled[0] = ang_vel_temp[0]
            self.ang_vel_scaled[1] = ang_vel_temp[1]
            self.ang_vel_scaled[2] = ang_vel_temp[2]

        # create observation - optimized gravity orientation computation
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        self.gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        self.gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        self.gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        
        # In-place scaling to avoid copies
        np.multiply(self.qj, self.dof_pos_scale_arr, out=self.qj_obs)
        np.multiply(self.dqj, self.dof_vel_scale_arr, out=self.dqj_obs)
        np.multiply(self.ang_vel_scaled, self.ang_vel_scale_val, out=self.ang_vel_scaled)
        
        # Command processing
        ly = self.remote_controller.ly
        self.cmd[0] = ly * 2 if ly > 0 else ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        # Command thresholding - zero out small values
        np.abs(self.cmd, out=self.cmd_abs)
        self.cmd[self.cmd_abs < 0.08] = 0.0

        num_actions = self.config.num_actions
        # Build observation array efficiently
        self.obs[:3] = self.ang_vel_scaled
        self.obs[3:6] = self.gravity_orientation
        self.obs[6:6 + num_actions] = self.qj_obs
        self.obs[6 + num_actions:6 + num_actions * 2] = self.dqj_obs
        self.obs[6 + num_actions * 2:6 + num_actions * 3] = self.action
        self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.cmd_scale_max_cmd
        
        # Get the action from the policy network
        self.action = self.policy_run.infer(self.obs)
        
        target_dof_pos = self.action * self.config.action_scale #29
        
        # # # Build low cmd
        # for i in range(len(self.config.leg_joint2motor_idx)):
        #     motor_idx = self.config.leg_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = np.clip(target_dof_pos[i],self.config.limits_low[i],self.config.limits_high[i])
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0
        # # print("arm_waist_joint2motor_idx")
        # for i in range(len(self.config.arm_waist_joint2motor_idx)):
        #     # print(target_dof_pos[i+len(self.config.leg_joint2motor_idx)],sep=',',end='')
        #     motor_idx = self.config.arm_waist_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = np.clip(target_dof_pos[i+len(self.config.leg_joint2motor_idx)],self.config.arm_waist_limits_low[i],
        #                                                   self.config.arm_waist_limits_high[i])
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        # send the command
        self.send_cmd(self.low_cmd)
    
    def Start(self):
        """RecurrentThread 기반 제어 루프 시작"""
        if self.lowCmdWriteThreadPtr is not None:
            print("Control thread already running")
            return
        
        self.running = True
        self.lowCmdWriteThreadPtr = RecurrentThread(
            name="lowcmd_write",
            interval=self.config.control_dt,
            target=self.LowCmdWrite
        )
        self.lowCmdWriteThreadPtr.Start()
        print(f"Control thread started with {1.0/self.config.control_dt:.1f}Hz period ({self.config.control_dt*1000:.1f}ms)")
    
    def Stop(self):
        """제어 루프 중지"""
        if self.lowCmdWriteThreadPtr is not None:
            self.running = False
            self.lowCmdWriteThreadPtr.Stop()
            self.lowCmdWriteThreadPtr = None
            print("Control thread stopped")
    
    def __del__(self):
        """리소스 정리"""
        if self.lowCmdWriteThreadPtr is not None:
            self.Stop()
        print("controller terminated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    # Start RecurrentThread-based control loop
    controller.Start()

    # Main loop: wait for exit signal
    try:
        while True:
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
            time.sleep(0.1)  # Check exit condition periodically
    except KeyboardInterrupt:
        pass
    finally:
        # Stop control thread
        controller.Stop()
        # Enter the damping state
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)

    print("Exit")
