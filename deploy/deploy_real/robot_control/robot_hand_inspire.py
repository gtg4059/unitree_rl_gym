# this file is legacy, need to fix.
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
# from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_ # OLD IDL, REMOVED
# from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_ # OLD IDL, REMOVED

# NEW IMPORTS for Inspire Hand SDK
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

import numpy as np
# from enum import IntEnum # Old Enums for joint indexing might not be needed for new DDS messages
import threading
import time
from multiprocessing import Process, Array, Lock # Removed shared_memory as Array is used

inspire_tip_indices = [4, 9, 14, 19, 24] # Assuming this remains relevant for hand_retargeting
Inspire_Num_Motors = 6 # Number of motors per hand

# NEW DDS TOPIC NAMES (assuming these are the correct topics based on SDK examples)
kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"

class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, dual_hand_state_array=None,
                 dual_hand_action_array=None, fps=100.0, Unit_Test=False, network_interface=""): # Added network_interface
        print("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.debug_counter = 0 # 디버깅 카운터 추가

        # Initialize DDS Channel Factory
        # This should ideally be called once per process.
        # If multiple controllers or DDS entities run in the same process, ensure this is handled.
        try:
            ChannelFactoryInitialize(0, network_interface)
            print(f"DDS ChannelFactory initialized with interface: '{network_interface if network_interface else 'default'}'")
        except Exception as e:
            print(f"Warning: DDS ChannelFactoryInitialize failed or already initialized: {e}")

        # Initialize hand command publishers
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init() # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state_loop)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait for initial DDS messages (optional, but good for ensuring connection)
        wait_count = 0
        while not (any(self.left_hand_state_array.get_obj()) or any(self.right_hand_state_array.get_obj())):
            if wait_count % 100 == 0: # Print every second
                print(f"[Inspire_Controller] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array.get_obj())}, R: {any(self.right_hand_state_array.get_obj())})...")
            time.sleep(0.01)
            wait_count +=1
            if wait_count > 500: # Timeout after 5 seconds
                print("[Inspire_Controller] Warning: Timeout waiting for initial hand states. Proceeding anyway.")
                break
        print("[Inspire_Controller] Initial hand states received or timeout.")

        hand_control_process = Process(target=self.control_process_loop, args=(
            left_hand_array, right_hand_array, self.left_hand_state_array, self.right_hand_state_array,
            dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state_loop(self):
        print("[Inspire_Controller] Subscribe thread started.")
        local_debug_counter = 0
        while True:
            # Left Hand
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i]*0.001
                else:
                    print(f"[Inspire_Controller] Warning: Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}")
            # Right Hand
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i]*0.001
                else:
                    print(f"[Inspire_Controller] Warning: Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}")
            
            local_debug_counter +=1
            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        # Left Hand Command
        left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        # Right Hand Command
        right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.RightHandCmd_publisher.Write(right_cmd_msg)


    def control_process_loop(self, left_hand_input_array, right_hand_input_array, 
                             shared_left_hand_state_array, shared_right_hand_state_array,
                             dual_hand_data_lock=None, dual_hand_state_array_shm=None, dual_hand_action_array_shm=None):
        print("[Inspire_Controller] Control process started.")
        running = True
        current_left_q_target_norm = np.ones(Inspire_Num_Motors, dtype=float) 
        current_right_q_target_norm = np.ones(Inspire_Num_Motors, dtype=float)

        try:
            while running:
                start_time = time.time()
                self.debug_counter +=1
                
                left_hand_mat = np.array(left_hand_input_array[:]).copy()
                right_hand_mat = np.array(right_hand_input_array[:]).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(shared_left_hand_state_array[:]), np.array(shared_right_hand_state_array[:])))

                # get dual hand action
                action_data = np.concatenate((left_hand_mat, right_hand_mat))    
                        
                if dual_hand_state_array_shm and dual_hand_action_array_shm:
                    with dual_hand_data_lock:
                        dual_hand_state_array_shm[:] = state_data
                        dual_hand_action_array_shm[:] = action_data

                left_q_target = left_hand_mat
                right_q_target = right_hand_mat
                
                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_hand_mat]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_hand_mat]
                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1.0 / self.fps) - time_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("[Inspire_Controller] Control process received KeyboardInterrupt. Exiting.")
        finally:
            running = False
            print("[Inspire_Controller] Control process has been closed.")

