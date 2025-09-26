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
        self.pickuptrigger = False
        self.pickdowntrigger = False
        self.pickup_walk_trigger = False
        # Initialize the policy network
        self.policy_run = torch.jit.load(config.policy_run)
        self.policy_stop = torch.jit.load(config.policy_stop)
        self.policy_pickup = torch.jit.load(config.policy_pickup)
        self.policy_pickup_walk = torch.jit.load(config.policy_pickup_walk)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.action_27 = np.zeros(config.num_actions-2, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.hand_ctrl = None
        self.fusion_client = None
        self.robot_data = []
        self.dual_ee_pos = np.array([0.32, 0.106, 0.15, 0.707, 0.0, 0.0, 0.707, 0.32, -0.106, 0.15,0.707, 0.0, 0.0, -0.707])
        self.initial_time = time.time()
        self.total_softrun_time = 3
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

        # pos
        # self.left_hand_array[:] = np.array([0.9,0.9,0.9,0.9,0,1], dtype=np.float32)
        # self.right_hand_array[:] = np.array([0.9,0.9,0.9,0.9,0,1], dtype=np.float32)
        # self.hand_ctrl = Inspire_Controller(self.left_hand_array, self.right_hand_array, self.dual_hand_data_lock, self.dual_hand_state_array, self.dual_hand_action_array)
        # 글로벌 FusionClient 사용 - 이미 실행 중이면 재사용
        
        # self.fusion_client = get_global_fusion_client(server_ip="192.168.123.164")
        
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

        while self.remote_controller.button[KeyMap.Y] != 1:
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

        self.softstart_stop()
        # self.softstart_pickpupwalk()

        # while self.remote_controller.button[KeyMap.A] != 1:
        #     for i in range(len(self.config.leg_joint2motor_idx)):
        #         motor_idx = self.config.leg_joint2motor_idx[i]
        #         self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
        #         self.low_cmd.motor_cmd[motor_idx].qd = 0
        #         self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
        #         self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
        #         self.low_cmd.motor_cmd[motor_idx].tau = 0
        #     for i in range(len(self.config.arm_waist_joint2motor_idx)):
        #         motor_idx = self.config.arm_waist_joint2motor_idx[i]
        #         self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_default_angles[i]
        #         self.low_cmd.motor_cmd[motor_idx].qd = 0
        #         self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
        #         self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
        #         self.low_cmd.motor_cmd[motor_idx].tau = 0
        #     self.send_cmd(self.low_cmd)
        #     time.sleep(self.config.control_dt)

       

    # def default_pos_state(self):
    #     print("Enter default pos state.")
    #     print("Waiting for the Button A signal...")
    #     # pos
    #     self.left_hand_array[:] = np.array([600,600,600,600,0,1000], dtype=np.float32)
    #     self.right_hand_array[:] = np.array([600,600,600,600,0,1000], dtype=np.float32)
    #     # self.hand_ctrl = Inspire_Controller(self.left_hand_array, self.right_hand_array, self.dual_hand_data_lock, self.dual_hand_state_array, self.dual_hand_action_array)
    #     # 글로벌 FusionClient 사용 - 이미 실행 중이면 재사용
    #     self.fusion_client = get_global_fusion_client(server_ip="192.168.123.164")
    #     print("FusionClient started")
    #     while self.remote_controller.button[KeyMap.A] != 1:
    #         for i in range(len(self.config.leg_joint2motor_idx)):
    #             motor_idx = self.config.leg_joint2motor_idx[i]
    #             self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
    #             self.low_cmd.motor_cmd[motor_idx].qd = 0
    #             self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
    #             self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
    #             self.low_cmd.motor_cmd[motor_idx].tau = 0
    #         for i in range(len(self.config.arm_waist_joint2motor_idx)):
    #             motor_idx = self.config.arm_waist_joint2motor_idx[i]
    #             self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_default_angles[i]
    #             self.low_cmd.motor_cmd[motor_idx].qd = 0
    #             self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
    #             self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
    #             self.low_cmd.motor_cmd[motor_idx].tau = 0
    #         self.send_cmd(self.low_cmd)
    #         time.sleep(self.config.control_dt)
    def softstart_walk(self):
        print("Moving to default pos using policy_stop.")
        # move time 2s
        total_time = 2
        stabletimestep = int(total_time / self.config.control_dt)
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        dof_size = len(dof_idx)
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        print('test1=============')

        # DualPoseCommandCfg 학습 시 설정에 맞춰 dual pose 명령 구성
        # 학습 시 ranges: pos_x=(0.30, 0.34), pos_y=(0.106, 0.106), pos_z=(0.13, 0.17), yaw=(π/2, π/2)
            
        # 공통 파라미터
        pos_x = 0.32  # 학습 시 범위의 중간값
        pos_z = 0.15  # 학습 시 범위의 중간값
        
        # 왼손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        left_pos_y = 0.11  # 양수
        left_quat = [0.707, 0.0, 0.0, 0.707]  # yaw=π/2에 해당하는 quaternion
        left_hand_pose = [pos_x, left_pos_y, pos_z] + left_quat
        
        # 오른손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        right_pos_y = -0.11  # 음수 (대칭)
        right_quat = [0.707, 0.0, 0.0, -0.707]  # yaw=-π/2에 해당하는 quaternion
        right_hand_pose = [pos_x, right_pos_y, pos_z] + right_quat
        
        # 14차원으로 결합
        self.dual_ee_pose = left_hand_pose + right_hand_pose
        print('test2============')
        # move to default pos using policy_stop
        for i in range(stabletimestep):
            start_time = time.time()
            alpha = i / stabletimestep
            
            # 현재 관절 위치와 속도 업데이트
            for j in range(len(self.config.leg_joint2motor_idx)):
                self.qj[j] = self.low_state.motor_state[self.config.leg_joint2motor_idx[j]].q
                self.dqj[j] = self.low_state.motor_state[self.config.leg_joint2motor_idx[j]].dq
            for j in range(len(self.config.arm_waist_joint2motor_idx)):
                self.qj[j+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].q
                self.dqj[j+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].dq

            # IMU 데이터 처리
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

            if self.config.imu_type == "torso":
                waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
                waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
                quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
            print('test3=============')
            # observation 생성
            gravity_orientation = get_gravity_orientation(quat)
            qj_obs = self.qj.copy()
            dqj_obs = self.dqj.copy()
            ang_vel = ang_vel * self.config.ang_vel_scale

            num_actions = self.config.num_actions-2 # 29dof -> 27dof 허리 제거
            self.obs[:3] = ang_vel
            self.obs[3:6] = gravity_orientation
            self.obs[6 : 6 + num_actions] = np.delete(qj_obs, [13,14]) * self.config.dof_pos_scale
            self.obs[6 + num_actions : 6 + num_actions * 2] = np.delete(dqj_obs, [13,14]) * self.config.dof_vel_scale
            self.obs[6 + num_actions * 2 : 6 + num_actions * 3] = self.action_27
            self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0  # pickup_walk policy를 위해 cmd를 0으로 설정
            self.obs[9 + num_actions * 3:23 + num_actions * 3] = self.dual_ee_pose
            
            obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)  # 104차원
            # self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
            self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()
            print('test4============') 
            # action을 29차원으로 변환
            indices_to_skip = [13, 14]
            idx = 0
            for k in range(29):
                if k in indices_to_skip:
                    continue
                self.action[k] = self.action_27[idx]
                idx += 1
            
            # action을 target_dof_pos로 변환
            target_dof_pos = self.action * self.config.action_scale
            
            # # leg 관절(앞의 12개)은 건드리지 않고, arm/waist 관절만 제어
            # for j in range(len(self.config.leg_joint2motor_idx), dof_size):  # leg 관절 제외하고 시작
            #     motor_idx = dof_idx[j]
            #     start_pos = init_dof_pos[j]
            #     policy_target = target_dof_pos[j]
            #     interpolated_pos = start_pos * (1 - alpha) + policy_target * alpha
            #     self.low_cmd.motor_cmd[motor_idx].q = np.clip(interpolated_pos, self.config.arm_waist_limits_low[j-len(self.config.leg_joint2motor_idx)], 
            #                                                  self.config.arm_waist_limits_high[j-len(self.config.leg_joint2motor_idx)])
            #     self.low_cmd.motor_cmd[motor_idx].qd = 0
            #     self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
            #     self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
            #     self.low_cmd.motor_cmd[motor_idx].tau = 0
            print('test5=============')
            for i in range(len(self.config.leg_joint2motor_idx)):
                # print(target_dof_pos[i],sep=',',end='')
                motor_idx = self.config.leg_joint2motor_idx[i]
                start_pos = init_dof_pos[i]
                policy_target = target_dof_pos[i]
                interpolated_pos = start_pos * (1 - alpha) + policy_target * alpha
                self.low_cmd.motor_cmd[motor_idx].q = np.clip(interpolated_pos, self.config.limits_low[i], self.config.limits_high[i])
                # self.low_cmd.motor_cmd[motor_idx].q = np.clip(target_dof_pos[i],self.config.limits_low[i],self.config.limits_high[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            print('test6=============')
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                # print(target_dof_pos[i+len(self.config.leg_joint2motor_idx)],sep=',',end='')
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                start_pos = init_dof_pos[i+len(self.config.leg_joint2motor_idx)]
                policy_target = target_dof_pos[i+len(self.config.leg_joint2motor_idx)]
                interpolated_pos = start_pos * (1 - alpha) + policy_target * alpha
                self.low_cmd.motor_cmd[motor_idx].q = np.clip(interpolated_pos, self.config.arm_waist_limits_low[i], self.config.arm_waist_limits_high[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            print('test7=============')    

            self.send_cmd(self.low_cmd)
            elapsed = time.time() - start_time
            print("elapsed:",elapsed)
            if elapsed < self.config.control_dt:
                time.sleep(self.config.control_dt-elapsed)


        if np.any(np.abs(self.dqj) > 12):
            print(f"\n[ERROR] Motor velocity limit exceeded! Max velocity: {np.max(np.abs(self.dqj)):.2f} rad/s")
            print(f"Terminating robot control for safety.")
            # 비상 종료를 위해 댐핑 모드 또는 토크 0 명령 전송
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(0.1) # 명령 전송 후 잠시 대기
            raise SystemExit("Robot control terminated due to excessive motor velocity.") # 프로그램 강제 종료
    
    def softstart_pickupwalk(self):
        print("Moving to default pos using policy_stop.")
        # move time 2s
        total_time = 2
        stabletimestep = int(total_time / self.config.control_dt)
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        dof_size = len(dof_idx)
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # DualPoseCommandCfg 학습 시 설정에 맞춰 dual pose 명령 구성
        # 학습 시 ranges: pos_x=(0.30, 0.34), pos_y=(0.106, 0.106), pos_z=(0.13, 0.17), yaw=(π/2, π/2)
            
        # 공통 파라미터
        pos_x = 0.32  # 학습 시 범위의 중간값
        pos_z = 0.15  # 학습 시 범위의 중간값
        
        # 왼손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        left_pos_y = 0.11  # 양수
        left_quat = [0.707, 0.0, 0.0, 0.707]  # yaw=π/2에 해당하는 quaternion
        left_hand_pose = [pos_x, left_pos_y, pos_z] + left_quat
        
        # 오른손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        right_pos_y = -0.11  # 음수 (대칭)
        right_quat = [0.707, 0.0, 0.0, -0.707]  # yaw=-π/2에 해당하는 quaternion
        right_hand_pose = [pos_x, right_pos_y, pos_z] + right_quat
        
        # 14차원으로 결합
        self.dual_ee_pose = left_hand_pose + right_hand_pose
        # move to default pos using policy_stop
        for i in range(stabletimestep):
            start_time = time.time()
            alpha = i / stabletimestep
            
            # 현재 관절 위치와 속도 업데이트
            for j in range(len(self.config.leg_joint2motor_idx)):
                self.qj[j] = self.low_state.motor_state[self.config.leg_joint2motor_idx[j]].q
                self.dqj[j] = self.low_state.motor_state[self.config.leg_joint2motor_idx[j]].dq
            for j in range(len(self.config.arm_waist_joint2motor_idx)):
                self.qj[j+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].q
                self.dqj[j+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].dq

            # IMU 데이터 처리
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

            if self.config.imu_type == "torso":
                waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
                waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
                quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
            # observation 생성
            gravity_orientation = get_gravity_orientation(quat)
            qj_obs = self.qj.copy()
            dqj_obs = self.dqj.copy()
            ang_vel = ang_vel * self.config.ang_vel_scale

            num_actions = self.config.num_actions-2 # 29dof -> 27dof 허리 제거
            self.obs[:3] = ang_vel
            self.obs[3:6] = gravity_orientation
            self.obs[6 : 6 + num_actions] = np.delete(qj_obs, [13,14]) * self.config.dof_pos_scale
            self.obs[6 + num_actions : 6 + num_actions * 2] = np.delete(dqj_obs, [13,14]) * self.config.dof_vel_scale
            self.obs[6 + num_actions * 2 : 6 + num_actions * 3] = self.action_27
            self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0  # pickup_walk policy를 위해 cmd를 0으로 설정
            self.obs[9 + num_actions * 3:23 + num_actions * 3] = self.dual_ee_pose
            
            obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)  # 104차원
            # self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
            self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
            # action을 29차원으로 변환
            indices_to_skip = [13, 14]
            idx = 0
            for k in range(29):
                if k in indices_to_skip:
                    continue
                self.action[k] = self.action_27[idx]
                idx += 1
            
            # action을 target_dof_pos로 변환
            target_dof_pos = self.action * self.config.action_scale
            
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                # print(target_dof_pos[i+len(self.config.leg_joint2motor_idx)],sep=',',end='')
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                start_pos = init_dof_pos[i+len(self.config.leg_joint2motor_idx)]
                policy_target = target_dof_pos[i+len(self.config.leg_joint2motor_idx)]
                interpolated_pos = start_pos * (1 - alpha) + policy_target * alpha
                self.low_cmd.motor_cmd[motor_idx].q = np.clip(interpolated_pos, self.config.arm_waist_limits_low[i], self.config.arm_waist_limits_high[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            elapsed = time.time() - start_time
            if elapsed < self.config.control_dt:
                time.sleep(self.config.control_dt-elapsed)


        if np.any(np.abs(self.dqj) > 12):
            print(f"\n[ERROR] Motor velocity limit exceeded! Max velocity: {np.max(np.abs(self.dqj)):.2f} rad/s")
            print(f"Terminating robot control for safety.")
            # 비상 종료를 위해 댐핑 모드 또는 토크 0 명령 전송
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(0.1) # 명령 전송 후 잠시 대기
            raise SystemExit("Robot control terminated due to excessive motor velocity.") # 프로그램 강제 종료
        
    def softstart_stop(self):
        print("Moving to default pos using policy_stop.")
        # move time 2s
        total_time = 2
        stabletimestep = int(total_time / self.config.control_dt)
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        dof_size = len(dof_idx)
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # DualPoseCommandCfg 학습 시 설정에 맞춰 dual pose 명령 구성
        # 학습 시 ranges: pos_x=(0.30, 0.34), pos_y=(0.106, 0.106), pos_z=(0.13, 0.17), yaw=(π/2, π/2)
            
        # 공통 파라미터
        pos_x = 0.32  # 학습 시 범위의 중간값
        pos_z = 0.15  # 학습 시 범위의 중간값
        
        # 왼손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        left_pos_y = 0.11  # 양수
        left_quat = [0.707, 0.0, 0.0, 0.707]  # yaw=π/2에 해당하는 quaternion
        left_hand_pose = [pos_x, left_pos_y, pos_z] + left_quat
        
        # 오른손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        right_pos_y = -0.11  # 음수 (대칭)
        right_quat = [0.707, 0.0, 0.0, -0.707]  # yaw=-π/2에 해당하는 quaternion
        right_hand_pose = [pos_x, right_pos_y, pos_z] + right_quat
        
        # 14차원으로 결합
        self.dual_ee_pose = left_hand_pose + right_hand_pose
        # move to default pos using policy_stop
        for i in range(stabletimestep):
            start_time = time.time()
            alpha = i / stabletimestep
            
            # 현재 관절 위치와 속도 업데이트
            for j in range(len(self.config.leg_joint2motor_idx)):
                self.qj[j] = self.low_state.motor_state[self.config.leg_joint2motor_idx[j]].q
                self.dqj[j] = self.low_state.motor_state[self.config.leg_joint2motor_idx[j]].dq
            for j in range(len(self.config.arm_waist_joint2motor_idx)):
                self.qj[j+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].q
                self.dqj[j+len(self.config.leg_joint2motor_idx)] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[j]].dq

            # IMU 데이터 처리
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

            if self.config.imu_type == "torso":
                waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
                waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
                quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
            # observation 생성
            gravity_orientation = get_gravity_orientation(quat)
            qj_obs = self.qj.copy()
            dqj_obs = self.dqj.copy()
            ang_vel = ang_vel * self.config.ang_vel_scale

            num_actions = self.config.num_actions-2 # 29dof -> 27dof 허리 제거
            self.obs[:3] = ang_vel
            self.obs[3:6] = gravity_orientation
            self.obs[6 : 6 + num_actions] = np.delete(qj_obs, [13,14]) * self.config.dof_pos_scale
            self.obs[6 + num_actions : 6 + num_actions * 2] = np.delete(dqj_obs, [13,14]) * self.config.dof_vel_scale
            self.obs[6 + num_actions * 2 : 6 + num_actions * 3] = self.action_27
            self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0  # pickup_walk policy를 위해 cmd를 0으로 설정
            self.obs[9 + num_actions * 3:23 + num_actions * 3] = self.dual_ee_pose
            
            obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)  # 104차원
            # self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
            self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()
            # action을 29차원으로 변환
            indices_to_skip = [13, 14]
            idx = 0
            for k in range(29):
                if k in indices_to_skip:
                    continue
                self.action[k] = self.action_27[idx]
                idx += 1
            
            # action을 target_dof_pos로 변환
            target_dof_pos = self.action * self.config.action_scale
            
            for i in range(len(self.config.leg_joint2motor_idx)):
                # print(target_dof_pos[i],sep=',',end='')
                motor_idx = self.config.leg_joint2motor_idx[i]
                start_pos = init_dof_pos[i]
                policy_target = target_dof_pos[i]
                interpolated_pos = start_pos * (1 - alpha) + policy_target * alpha
                self.low_cmd.motor_cmd[motor_idx].q = np.clip(interpolated_pos, self.config.limits_low[i], self.config.limits_high[i])
                # self.low_cmd.motor_cmd[motor_idx].q = np.clip(target_dof_pos[i],self.config.limits_low[i],self.config.limits_high[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                # print(target_dof_pos[i+len(self.config.leg_joint2motor_idx)],sep=',',end='')
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                start_pos = init_dof_pos[i+len(self.config.leg_joint2motor_idx)]
                policy_target = target_dof_pos[i+len(self.config.leg_joint2motor_idx)]
                interpolated_pos = start_pos * (1 - alpha) + policy_target * alpha
                self.low_cmd.motor_cmd[motor_idx].q = np.clip(interpolated_pos, self.config.arm_waist_limits_low[i], self.config.arm_waist_limits_high[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            elapsed = time.time() - start_time
            if elapsed < self.config.control_dt:
                time.sleep(self.config.control_dt-elapsed)


        if np.any(np.abs(self.dqj) > 12):
            print(f"\n[ERROR] Motor velocity limit exceeded! Max velocity: {np.max(np.abs(self.dqj)):.2f} rad/s")
            print(f"Terminating robot control for safety.")
            # 비상 종료를 위해 댐핑 모드 또는 토크 0 명령 전송
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(0.1) # 명령 전송 후 잠시 대기
            raise SystemExit("Robot control terminated due to excessive motor velocity.") # 프로그램 강제 종료
 
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
        # qj_obs = qj_obs * self.config.dof_pos_scale
        # dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        
        # joystick command
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        # # # slam command
        # # send_walk_command_data=get_walk_command_data()
        # # self.cmd[0] = send_walk_command_data[0]
        # # self.cmd[1] = send_walk_command_data[1]
        # # self.cmd[2] = send_walk_command_data[2]

        for i in range(len(self.cmd)):
            if abs(self.cmd[i]) < 0.03:
                self.cmd[i] = 0

        num_actions = self.config.num_actions-2 # 29dof -> 27dof 허리 제거
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6 : 6 + num_actions] = np.delete(qj_obs, [13,14]) * self.config.dof_pos_scale
        self.obs[6 + num_actions : 6 + num_actions * 2] = np.delete(dqj_obs, [13,14]) * self.config.dof_vel_scale
        self.obs[6 + num_actions * 2 : 6 + num_actions * 3] = self.action_27


        pos_x = 0.31  # 학습 시 범위의 중간값
        pos_z = 0.1  # 학습 시 범위의 중간값
        roll = 0.0
        pitch = 0.0
        yaw = 1.57  # π/2
        left_pos_y = 0.11  # 양수
        left_quat = [0.707, 0.0, 0.0, 0.707]  # yaw=π/2에 해당하는 quaternion
        left_hand_pose = [pos_x, left_pos_y, pos_z] + left_quat
        # 오른손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        right_pos_y = -0.11  # 음수 (대칭)
        right_quat = [0.707, 0.0, 0.0, -0.707]  # yaw=-π/2에 해당하는 quaternion
        right_hand_pose = [pos_x, right_pos_y, pos_z] + right_quat
        
        # 14차원으로 결합
        self.dual_ee_pose = left_hand_pose + right_hand_pose
        self.obs[9 + num_actions * 3:23 + num_actions * 3] = self.dual_ee_pose


        # # self.softstart_walk()

        # # 일단 서서 box인식하는지 확인해야 한다
        # # box_center = get_box_center_data()
        # # print("boxdata:", box_center)
        
        # joystick
        # if self.remote_controller.button[KeyMap.A] == 1:
        #     print("Pickup Trigger On!")
        #     self.pickuptrigger = True
        #     self.pickup_walk_trigger = False
        #     self.pickdowntrigger = False
        #     # self.softstart_pickpup()
        # elif self.remote_controller.button[KeyMap.X] == 1:
        #     print("Pickup Walk Trigger On!")
        #     self.pickup_walk_trigger = True
        #     self.pickuptrigger = False
        #     self.pickdowntrigger = False
        #     self.softstart_pickupwalk()
        # elif self.remote_controller.button[KeyMap.B] == 1:
        #     print("Pickdown Trigger On!")
        #     self.pickdowntrigger = True
        #     self.pickup_walk_trigger = False
        #     self.pickuptrigger = False


        # if self.pickuptrigger:
        #     print("pickloop")
        #     # 공통 파라미터
        #     pos_x = 0.31  # 학습 시 범위의 중간값
        #     pos_z = 0.20  # 학습 시 범위의 중간값
        #     left_pos_y = 0.12  # 양수
        #     left_quat = [0.707, 0.0, 0.0, 0.707]  # yaw=π/2에 해당하는 quaternion
        #     left_hand_pose = [pos_x, left_pos_y, pos_z] + left_quat
        #     # 오른손 pose (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        #     right_pos_y = -0.12  # 음수 (대칭)
        #     right_quat = [0.707, 0.0, 0.0, -0.707]  # yaw=-π/2에 해당하는 quaternion
        #     right_hand_pose = [pos_x, right_pos_y, pos_z] + right_quat
            
        #     # 14차원으로 결합
        #     self.dual_ee_pose = left_hand_pose + right_hand_pose
        #     self.obs[6 + num_actions * 3:20 + num_actions * 3] = self.dual_ee_pose
        #     self.obs[20 + num_actions * 3:23 + num_actions * 3] = box_center #3
        #     obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #     self.action_27 = self.policy_pickup(obs_tensor).detach().numpy().squeeze()
        # elif self.pickup_walk_trigger:
        #     THR_VEL_XY=0.01
        #     THR_VEL_YAW=0.01
        #     if abs(send_walk_command_data[0])>THR_VEL_XY or abs(send_walk_command_data[1])>THR_VEL_XY or abs(send_walk_command_data[2])>THR_VEL_YAW:
        #         self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        #         obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #         # obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #         self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
        #     else:
        #         self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0
        #         obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #         # obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #         self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()

            # # 104 차원임
            # self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd  # pickup_walk policy를 위해 cmd를 0으로 설정
            # self.obs[9 + num_actions * 3:23 + num_actions * 3] = self.dual_ee_pose
            # obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)  # 104차원
            # self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
        # elif self.pickdowntrigger:
        #     self.obs[6 + num_actions * 3:13 + num_actions * 3] = np.array([0.3100, 0.1200, 0.1000, 0.7071, 0.0000, -0.0000, 0.7071],dtype=np.float32) #7
        #     self.obs[13 + num_actions * 3:20 + num_actions * 3] = np.array([0.3100, -0.1200,  0.1000,  0.7071, -0.0000,  0.0000, -0.7071],dtype=np.float32) #7
        #     self.obs[20 + num_actions * 3:23 + num_actions * 3] = box_center #3
        #     obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #     self.action_27 = self.policy_pickdown(obs_tensor).detach().numpy().squeeze()
        # else: # stand
        #     self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0
        #     # obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #     obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #     self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()

        # #slam
        # # THR_VEL_XY=0.01
        # # THR_VEL_YAW=0.01
        # if self.remote_controller.button[KeyMap.X] == 1:
        #     if abs(send_walk_command_data[0])>THR_VEL_XY or abs(send_walk_command_data[1])>THR_VEL_XY or abs(send_walk_command_data[2])>THR_VEL_YAW:
        #         self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        #         obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #         # obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #         self.action_27 = self.policy_run(obs_tensor).detach().numpy().squeeze()
        #     else:
        #         self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0
        #         obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #         # obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #         self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()
        # else:
        #     self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0
        #     obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #     # obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #     self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()

        if self.remote_controller.button[KeyMap.X] == 1: # run
            self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd #3
            obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
            self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
        else:
            self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0
            obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
            # obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
            self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()

        ## Test
        # self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd #3
        # obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        # if np.linalg.norm(self.cmd) > 0.07:
        #     self.action_27 = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
        # else:
        #     self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()

        # if self.remote_controller.button[KeyMap.X] == 1 and self.remote_controller.button[KeyMap.Y] != 1: # run
        #     self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd #3
        #     obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #     self.action_27 = self.policy_run(obs_tensor).detach().numpy().squeeze()
        # elif self.remote_controller.button[KeyMap.X] != 1 and self.remote_controller.button[KeyMap.Y] == 1: # pickup
        #     self.obs[6 + num_actions * 3:13 + num_actions * 3] = np.array([0.3200, 0.1400, 0.1600, 0.7071, 0.0000, -0.0000, 0.7071],dtype=np.float32) #7
        #     self.obs[13 + num_actions * 3:20 + num_actions * 3] = np.array([0.3200, -0.1400,  0.1600,  0.7071, -0.0000,  0.0000, -0.7071],dtype=np.float32) #7
        #     box_center = get_box_center_data()
        #     self.obs[20 + num_actions * 3:23 + num_actions * 3] = box_center #3
        #     obs_tensor = torch.from_numpy(self.obs[:104]).unsqueeze(0)
        #     self.action_27 = self.policy_pickup(obs_tensor).detach().numpy().squeeze()
        # elif self.remote_controller.button[KeyMap.B] == 1:
        #     self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        #     # 여기에 left_ee_pos_command + right ee pose command 들어가면 될 것 같음.
        #     # left_ee_pose_command (7) - pos(3) + quat(4)
        #     # left_ee_pos = [0.34, 0.14, 0.15]  # velocity_env_cfg.py에서 정의된 값
        #     left_ee_pos = [0.32, 0.16, 0.15]
        #     left_ee_quat = [0.707, 0.0, 0.0, 0.707]  # yaw=π/2에 해당하는 quaternion
        #     self.obs[9 + num_actions * 3:16 + num_actions * 3] = np.concatenate([left_ee_pos, left_ee_quat])
        #     # right_ee_pose_command (7) - pos(3) + quat(4)
        #     # right_ee_pos = [0.34, -0.14, 0.15]  # velocity_env_cfg.py에서 정의된 값
        #     right_ee_pos = [0.32, -0.16, 0.15]
        #     right_ee_quat = [-0.707, 0.0, 0.0, 0.707]  # yaw=-π/2에 해당하는 quaternion
        #     self.obs[16 + num_actions * 3:23 + num_actions * 3] = np.concatenate([right_ee_pos, right_ee_quat])
        #     obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        #     self.action = self.policy_pickup_walk(obs_tensor).detach().numpy().squeeze()
        # else:
        #     self.obs[6 + num_actions * 3:9 + num_actions * 3] = self.cmd * 0
        #     obs_tensor = torch.from_numpy(self.obs[:90]).unsqueeze(0)
        #     self.action_27 = self.policy_stop(obs_tensor).detach().numpy().squeeze()

        indices_to_skip = [13, 14]
        idx = 0
        for i in range(29):
            if i in indices_to_skip:
                continue
            self.action[i] = self.action_27[idx]
            idx += 1

        # transform action to target_dof_pos
        target_dof_pos = self.action * self.config.action_scale #29

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            # print(target_dof_pos[i],sep=',',end='')
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = np.clip(target_dof_pos[i],self.config.limits_low[i],self.config.limits_high[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

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
        
        # # # 액션과 목표 위치 추가
        # # for i in range(len(self.action)):
        # #     data_row[f'action_{i}'] = float(self.action[i])
        # #     # data_row[f'target_dof_pos_{i}'] = float(target_dof_pos[i])
            
        # # obs 위치 추가
        # for i in range(len(self.obs)):
        #     data_row[f'obs_{i}'] = float(self.obs[i])
        
        # self.robot_data.append(data_row)

        if np.any(np.abs(self.dqj) > 15):
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
        # print("elapsed:",elapsed)
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
    # controller.default_pos_state()

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