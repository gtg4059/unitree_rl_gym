from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch
import os
import pandas as pd
import datetime

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config_v1 import Config
# from multiprocessing import Process, shared_memory, Array
# from multiprocessing import shared_memory, Array, Lock
# from robot_control.robot_hand_inspire import Inspire_Controller


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.robot_data = []
        # Initialize the policy network
        self.policy_run = torch.jit.load(config.policy_run)
        self.policy_stop = torch.jit.load(config.policy_stop)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.start_time = None

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

    def save_data_to_csv(self, filename=None):
        """
        수집된 로봇 데이터를 CSV 파일로 저장
        """
        if not self.robot_data:
            print("No data to save.")
            return
        
        if filename is None:
            # 현재 시간을 포함한 파일명 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_data_{timestamp}.csv"
        
        # DataFrame 생성
        df = pd.DataFrame(self.robot_data)
        # CSV 파일로 저장
        df.to_csv(filename, index=False)
        
        # action 통계 (처음 5개 관절)
        print("Action (first 5 joints):")
        for j in range(min(5, 23)):
            col = f'action_{j}'
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"  joint_{j}: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
        
        return filename
    
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

    def run(self):
        self.counter += 1
        start_time = time.time()
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            self.qj[i+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].q
            self.dqj[i+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = qj_obs * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        for i in range(len(self.cmd)):
            if abs(self.cmd[i]) < 0.1:
                self.cmd[i] = 0

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6 : 6 + num_actions] = qj_obs
        self.obs[6 + num_actions : 6 + num_actions * 2] = dqj_obs
        self.obs[6 + num_actions * 2 : 6 + num_actions * 3] = self.action
        
        # print("self.obs:",*self.obs)
        # Get the action from the policy network
        if controller.remote_controller.button[KeyMap.X] == 1:
            self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
            obs_tensor = torch.from_numpy(self.obs[:96]).unsqueeze(0)
            self.action = self.policy_run(obs_tensor).detach().numpy().squeeze()
        else:
            self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0
            obs_tensor = torch.from_numpy(self.obs[:96]).unsqueeze(0)
            self.action = self.policy_stop(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.action * self.config.action_scale #29
        # target_dof_pos = self.action * self.config.action_scale #29
        # print("target_dof_pos:",*target_dof_pos)

        # Build low cmd
        # print("leg_joint2motor_idx")
        for i in range(len(self.config.leg_joint2motor_idx)):
            # print(target_dof_pos[i],sep=',',end='')
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = np.clip(target_dof_pos[i],self.config.limits_low[i],self.config.limits_high[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        # print("arm_waist_joint2motor_idx")
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            # print(target_dof_pos[i+len(self.config.leg_joint2motor_idx)],sep=',',end='')
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = np.clip(target_dof_pos[i+len(self.config.leg_joint2motor_idx)],self.config.arm_waist_limits_low[i],
                                                          self.config.arm_waist_limits_high[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # # 데이터 수집 (매 스텝마다)
        # data_row = {}
        
        # # 액션과 목표 위치 추가
        # for i in range(len(self.action)):
        #     data_row[f'action_{i}'] = float(self.action[i])
        #     # data_row[f'target_dof_pos_{i}'] = float(target_dof_pos[i])
            
        # # # obs 위치 추가
        # # for i in range(len(self.obs)):
        # #     data_row[f'obs_{i}'] = float(self.obs[i])
        
        # self.robot_data.append(data_row)

        if np.any(np.abs(self.dqj) > 16):
            print(f"\n[ERROR] Motor velocity limit exceeded! Max velocity: {np.max(np.abs(self.dqj)):.2f} rad/s")
            print(f"Terminating robot control for safety.")
            # 비상 종료를 위해 댐핑 모드 또는 토크 0 명령 전송
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(0.1) # 명령 전송 후 잠시 대기
            raise SystemExit("Robot control terminated due to excessive motor velocity.") # 프로그램 강제 종료

        # send the command
        self.send_cmd(self.low_cmd)

        elapsed = time.time() - start_time
        # print(elapsed)
        if elapsed < self.config.control_dt:
            time.sleep(self.config.control_dt-elapsed)



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

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)

    print("Saving robot data...")
    csv_filename = controller.save_data_to_csv()
    print(f"Data saved to: {csv_filename}")

    print("Exit")
