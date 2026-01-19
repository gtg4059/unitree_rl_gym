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
from multiprocessing import Process, Queue
import multiprocessing
# from multiprocessing import shared_memory, Array, Lock
# from robot_control.robot_hand_inspire import Inspire_Controller

def inference_process_worker(obs_queue, action_queue, policy_path, num_actions, num_obs, num_layers, hidden_size, stop_event, init_ready_event):
    """
    별도 프로세스에서 실행되는 추론 워커
    GPU 리소스를 독립적으로 사용하여 추론 성능 향상
    """
    try:
        # 자식 프로세스에서 CUDA 초기화 (멀티프로세싱 환경에서 필수)
        import pycuda.driver as cuda
        import pycuda.autoinit
        # CUDA 컨텍스트가 제대로 초기화되었는지 확인
        cuda.init()
        print(f"[Inference Process] CUDA initialized in PID {os.getpid()}")
        
        # 추론 프로세스에서 TrtPolicyRunner 초기화
        policy_runner = TrtPolicyRunner(policy_path, num_layers=num_layers, hidden_size=hidden_size)
        print(f"[Inference Process] Policy runner initialized in PID {os.getpid()}")
        
        # 초기화 완료 신호 전송
        init_ready_event.set()
        print(f"[Inference Process] Initialization complete, ready to process observations")
        
        while not stop_event.is_set():
            try:
                # obs_queue에서 관측값 받기 (timeout으로 블로킹 방지)
                if not obs_queue.empty():
                    obs = obs_queue.get(timeout=0.001)
                    
                    # 추론 수행
                    action = policy_runner.infer(obs)
                    
                    # action_queue에 결과 전송 (non-blocking)
                    if not action_queue.full():
                        action_queue.put(action, block=False)
                    else:
                        # 큐가 가득 찬 경우 오래된 항목 제거하고 새 항목 추가
                        try:
                            action_queue.get_nowait()
                        except:
                            pass
                        action_queue.put(action, block=False)
                        
            except:
                # 큐가 비어있거나 타임아웃된 경우 계속 진행
                continue
        
        print(f"[Inference Process] Worker stopped in PID {os.getpid()}")
    except Exception as e:
        print(f"[Inference Process] Error: {e}")
        import traceback
        traceback.print_exc()


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.robot_data = []
        # 추론 프로세스 관련 변수 (메인 프로세스에서는 TrtPolicyRunner 초기화 안 함)
        self.inference_process = None
        self.obs_queue = None
        self.action_queue = None
        self.inference_stop_event = None
        self.inference_init_ready = None  # 초기화 완료 이벤트
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.start_time = None
        # Pre-allocate observation slices for better performance
        self.num_actions = config.num_actions
        self.obs_ang_vel_slice = slice(0, 3)
        self.obs_gravity_slice = slice(3, 6)
        self.obs_qj_slice = slice(6, 6 + self.num_actions)
        self.obs_dqj_slice = slice(6 + self.num_actions, 6 + self.num_actions * 2)
        self.obs_action_slice = slice(6 + self.num_actions * 2, 6 + self.num_actions * 3)
        self.obs_cmd_slice = slice(6 + self.num_actions * 3, 9 + self.num_actions * 3)
        # Pre-compute scaled values
        self.dof_pos_scale = config.dof_pos_scale
        self.dof_vel_scale = config.dof_vel_scale
        self.ang_vel_scale = config.ang_vel_scale
        self.cmd_scale_mul = config.cmd_scale * config.max_cmd
        self.action_scale = config.action_scale
        # Thread handling for RecurrentThread pattern
        self.lowCmdWriteThreadPtr = None
        self.running = False
        # Thread-safe state access
        import threading
        self.low_state_lock = threading.Lock()
        # Action 버퍼 (추론 프로세스에서 받은 최신 action 저장)
        self.latest_action = np.zeros(config.num_actions, dtype=np.float32)
        # Loop time 측정 및 통계
        self.loop_times = []  # 최근 N개 루프 시간 저장
        self.max_loop_history = 100  # 통계 계산을 위한 최대 저장 개수
        self.stats_print_interval = 50  # 통계 출력 주기 (루프 횟수)
        self.last_stats_print = 0

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
        """DDS Subscriber 콜백 - 스레드 안전하게 상태 업데이트"""
        with self.low_state_lock:
            self.low_state = msg
            self.mode_machine_ = self.low_state.mode_machine
            self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        """DDS Subscriber 콜백 - 스레드 안전하게 상태 업데이트"""
        with self.low_state_lock:
            self.low_state = msg
            self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while True:
            with self.low_state_lock:
                if self.low_state.tick != 0:
                    break
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
        
        # record the current pos (thread-safe read)
        with self.low_state_lock:
            current_state = self.low_state
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = current_state.motor_state[dof_idx[i]].q
        
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
        
        # Loop time 측정 시작
        loop_start_time = time.perf_counter()
            
        self.counter += 1
        
        # Thread-safe read: get consistent snapshot of low_state
        with self.low_state_lock:
            current_state = self.low_state

        # Get the current joint position and velocity
        leg_idx = self.config.leg_joint2motor_idx
        arm_idx = self.config.arm_waist_joint2motor_idx
        leg_len = len(leg_idx)
        
        for i in range(leg_len):
            motor_idx = leg_idx[i]
            self.qj[i] = current_state.motor_state[motor_idx].q
            self.dqj[i] = current_state.motor_state[motor_idx].dq
        
        for i in range(len(arm_idx)):
            idx = i + leg_len
            motor_idx = arm_idx[i]
            self.qj[idx] = current_state.motor_state[motor_idx].q
            self.dqj[idx] = current_state.motor_state[motor_idx].dq

        # imu_state quaternion: w, x, y, z
        quat = current_state.imu_state.quaternion
        ang_vel = np.array([current_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = current_state.motor_state[arm_idx[0]].q
            waist_yaw_omega = current_state.motor_state[arm_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation - use in-place operations to avoid copies
        gravity_orientation = get_gravity_orientation(quat)
        
        # Scale ang_vel in-place
        ang_vel *= self.ang_vel_scale
        self.obs[self.obs_ang_vel_slice] = ang_vel
        self.obs[self.obs_gravity_slice] = gravity_orientation
        
        # Scale qj and dqj directly into obs (avoiding copy)
        np.multiply(self.qj, self.dof_pos_scale, out=self.obs[self.obs_qj_slice])
        np.multiply(self.dqj, self.dof_vel_scale, out=self.obs[self.obs_dqj_slice])
        self.obs[self.obs_action_slice] = self.action

        # Update command
        ly = self.remote_controller.ly
        self.cmd[0] = ly * 2 if ly > 0 else ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        # Dead zone check - vectorized
        abs_cmd = np.abs(self.cmd)
        self.cmd[abs_cmd < 0.08] = 0

        # Get the action from the policy network
        np.multiply(self.cmd, self.cmd_scale_mul, out=self.obs[self.obs_cmd_slice])
        
        # 추론 프로세스에 obs 전송 (non-blocking)
        if self.obs_queue is not None and not self.obs_queue.full():
            # obs 복사본을 큐에 전송 (원본은 유지)
            obs_copy = self.obs.copy()
            try:
                self.obs_queue.put(obs_copy, block=False)
            except:
                pass
        
        # 추론 프로세스에서 최신 action 받기 (non-blocking)
        if self.action_queue is not None:
            try:
                while not self.action_queue.empty():
                    self.latest_action = self.action_queue.get_nowait()
            except:
                pass
        
        # 최신 action 사용 (없으면 이전 action 유지)
        self.action = self.latest_action.copy()
        target_dof_pos = self.action * self.config.action_scale

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
        
        # Loop time 측정 및 통계 업데이트
        loop_elapsed = time.perf_counter() - loop_start_time
        loop_elapsed_ms = loop_elapsed * 1000.0
        
        # Loop time 저장 (최대 개수 제한)
        self.loop_times.append(loop_elapsed_ms)
        if len(self.loop_times) > self.max_loop_history:
            self.loop_times.pop(0)
        
        # 20ms 초과 시 즉시 경고
        if loop_elapsed_ms > 20.0:
            print(f"[WARNING] Loop time exceeded: {loop_elapsed_ms:.2f}ms (target: 20ms)")
        
        # 주기적으로 통계 출력
        if self.counter - self.last_stats_print >= self.stats_print_interval:
            self._print_loop_time_stats()
            self.last_stats_print = self.counter
    
    def _print_loop_time_stats(self):
        """Loop time 통계 출력"""
        if len(self.loop_times) == 0:
            return
        
        loop_times_array = np.array(self.loop_times)
        avg_time = np.mean(loop_times_array)
        min_time = np.min(loop_times_array)
        max_time = np.max(loop_times_array)
        std_time = np.std(loop_times_array)
        
        # 20ms 초과 비율 계산
        over_20ms_count = np.sum(loop_times_array > 20.0)
        over_20ms_ratio = (over_20ms_count / len(loop_times_array)) * 100.0
        
        print(f"[Loop Time Stats] Avg: {avg_time:.2f}ms | Min: {min_time:.2f}ms | Max: {max_time:.2f}ms | Std: {std_time:.2f}ms | >20ms: {over_20ms_ratio:.1f}% ({over_20ms_count}/{len(loop_times_array)})")
    
    def Start(self):
        """RecurrentThread 기반 제어 루프 및 추론 프로세스 시작"""
        if self.lowCmdWriteThreadPtr is not None:
            print("Control thread already running")
            return
        
        # 추론 프로세스 시작 및 초기화 완료 대기
        self._start_inference_process()
        # _start_inference_process() 내부에서 초기화 완료를 기다리므로
        # 여기서는 바로 제어 루프를 시작할 수 있음
            
        self.running = True
        self.lowCmdWriteThreadPtr = RecurrentThread(
            name="lowcmd_write",
            interval=self.config.control_dt,
            target=self.LowCmdWrite
        )
        self.lowCmdWriteThreadPtr.Start()
        print(f"Control thread started with {1.0/self.config.control_dt:.1f}Hz period ({self.config.control_dt*1000:.1f}ms)")
    
    def Stop(self):
        """제어 루프 및 추론 프로세스 중지"""
        if self.lowCmdWriteThreadPtr is not None:
            self.running = False
            self.lowCmdWriteThreadPtr.Stop()
            self.lowCmdWriteThreadPtr = None
            print("Control thread stopped")
        
        # 추론 프로세스 중지
        self._stop_inference_process()
    
    def _start_inference_process(self):
        """추론 프로세스 시작"""
        if self.inference_process is not None:
            print("Inference process already running")
            return
        
        # Linux에서 fork 대신 spawn 방식 사용 (CUDA 호환성 향상)
        # spawn 방식은 각 프로세스가 독립적으로 시작되어 CUDA 초기화 문제를 방지
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # 이미 설정된 경우 무시
            pass
        
        # Queue 생성 (크기 제한으로 메모리 사용량 제어)
        self.obs_queue = Queue(maxsize=2)
        self.action_queue = Queue(maxsize=2)
        
        # 프로세스 종료 이벤트 및 초기화 완료 이벤트
        self.inference_stop_event = multiprocessing.Event()
        self.inference_init_ready = multiprocessing.Event()
        
        # 추론 프로세스 시작
        self.inference_process = Process(
            target=inference_process_worker,
            args=(
                self.obs_queue,
                self.action_queue,
                self.config.policy_run,
                self.config.num_actions,
                self.config.num_obs,
                1,  # num_layers
                64,  # hidden_size
                self.inference_stop_event,
                self.inference_init_ready
            ),
            daemon=True
        )
        self.inference_process.start()
        print(f"[Main Process] Inference process started with PID {self.inference_process.pid}")
        
        # 초기화 완료 대기 (최대 10초)
        # 대기하는 동안 default position 유지
        print("[Main Process] Waiting for inference process initialization...")
        start_wait_time = time.time()
        timeout = 10.0
        
        while not self.inference_init_ready.is_set():
            # 초기화 대기 중에도 default position 유지
            self._send_default_pos_cmd()
            
            # 타임아웃 확인
            if time.time() - start_wait_time > timeout:
                raise RuntimeError("[Main Process] Inference process initialization timeout!")
            
            # 짧은 대기 (제어 주기 유지)
            time.sleep(self.config.control_dt)
        
        print("[Main Process] Inference process initialization complete!")
    
    def _send_default_pos_cmd(self):
        """Default position 명령 전송 (초기화 중 유지용)"""
        try:
            # Leg joints
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            
            # Arm and waist joints
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            
            self.send_cmd(self.low_cmd)
        except Exception as e:
            # 에러 발생 시 무시 (초기화 중일 수 있음)
            pass
    
    def _stop_inference_process(self):
        """추론 프로세스 중지"""
        if self.inference_process is not None:
            print("[Main Process] Stopping inference process...")
            self.inference_stop_event.set()
            self.inference_process.join(timeout=2.0)
            if self.inference_process.is_alive():
                print("[Main Process] Force terminating inference process...")
                self.inference_process.terminate()
                self.inference_process.join(timeout=1.0)
                if self.inference_process.is_alive():
                    self.inference_process.kill()
            self.inference_process = None
            self.obs_queue = None
            self.action_queue = None
            self.inference_stop_event = None
            self.inference_init_ready = None
            print("[Main Process] Inference process stopped")
    
    def __del__(self):
        """소멸자에서 리소스 정리"""
        try:
            self.Stop()
        except:
            pass
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
    
    try:
        # Main thread는 제어 루프 모니터링 및 종료 조건 확인
        while controller.running:
            time.sleep(0.1)  # 주기적으로 종료 조건 확인
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # 제어 스레드 중지
        controller.Stop()
        
        # Enter the damping state
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)

    print("Exit")
