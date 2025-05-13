from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
        # #    'left_hip_yaw_joint' : 0. ,   
        # #    'left_hip_roll_joint' : 0,               
        # #    'left_hip_pitch_joint' : -0.1,         
        # #    'left_knee_joint' : 0.3,       
        # #    'left_ankle_pitch_joint' : -0.2,     
        # #    'left_ankle_roll_joint' : 0,     
        # #    'right_hip_yaw_joint' : 0., 
        # #    'right_hip_roll_joint' : 0, 
        # #    'right_hip_pitch_joint' : -0.1,                                       
        # #    'right_knee_joint' : 0.3,                                             
        # #    'right_ankle_pitch_joint': -0.2,                              
        # #    'right_ankle_roll_joint' : 0,       
        # #    'torso_joint' : 0.
        #     "left_hip_roll_joint" : 0.00,
        #     "left_hip_pitch_joint": -0.20,
        #     "left_hip_yaw_joint": 0.00,

        #     "left_knee_joint": 0.42,
        #     "rightt_hip_pitch_joint": -0.20,
        #     # "left_knee_joint": 0.42,
        #     "right_knee_joint": 0.42,
        #     "left_ankle_pitch_joint": -0.23,
        #     "right_ankle_pitch_joint": -0.23,
        #     "left_shoulder_roll_joint": 0.3,
        #     "right_shoulder_roll_joint": -0.3,
        #     "left_wrist_roll_joint": -0.2,
        #     "right_wrist_roll_joint": 0.2,
        #     ".*_shoulder_pitch_joint": 0.8,
        #     ".*_elbow_joint": -0.4,
        #     ".*_wrist_pitch_joint": -0.4,
        #     ".*_thumb_proximal_pitch_joint": 0.52
        'left_hip_pitch_joint': -0.20, 
        'right_hip_pitch_joint': -0.20, 
        'waist_yaw_joint': -0.0, 
        'left_hip_roll_joint': 0.0,
        'right_hip_roll_joint': -0.0, 
        'waist_roll_joint': -0.0, 
        'left_hip_yaw_joint': -0.0, 
        'right_hip_yaw_joint': -0.0, 
        'waist_pitch_joint':  -0.0,
        'left_knee_joint':  0.42, 
        'right_knee_joint': 0.42, 
        'left_shoulder_pitch_joint': 0.8, 
        'right_shoulder_pitch_joint': 0.8, 
        'left_ankle_pitch_joint': -0.23, 
        'right_ankle_pitch_joint': -0.23, 
        'left_shoulder_roll_joint': 0.3, 
        'right_shoulder_roll_joint': -0.3,
        'left_ankle_roll_joint': -0.0, 
        'right_ankle_roll_joint': 0.0, 
        'left_shoulder_yaw_joint': -0.0, 
        'right_shoulder_yaw_joint': 0.0, 
        'left_elbow_joint': -0.4, 
        'right_elbow_joint': -0.4, 
        'left_wrist_roll_joint': -0.2, 
        'right_wrist_roll_joint': 0.2,
        'left_wrist_pitch_joint':  -0.4, 
        'right_wrist_pitch_joint': -0.4, 
        'left_wrist_yaw_joint': -0.0, 
        'right_wrist_yaw_joint': 0.0, 
        'L_index_proximal_joint': 0.0, 
        'L_middle_proximal_joint': 0.0, 
        'L_pinky_proximal_joint': 0.0, 
        'L_ring_proximal_joint': 0.0,
        'L_thumb_proximal_yaw_joint': -0.0, 
        'R_index_proximal_joint': 0.0, 
        'R_middle_proximal_joint': 0.0, 
        'R_pinky_proximal_joint': 0.0, 
        'R_ring_proximal_joint': 0.0, 
        'R_thumb_proximal_yaw_joint': -0.0, 
        'L_index_intermediate_joint': -0.0, 
        'L_middle_intermediate_joint': -0.0, 
        'L_pinky_intermediate_joint': -0.0, 
        'L_ring_intermediate_joint': -0.0, 
        'L_thumb_proximal_pitch_joint': 0.52, 
        'R_index_intermediate_joint': 0.0, 
        'R_middle_intermediate_joint': 0.0, 
        'R_pinky_intermediate_joint': -0.0, 
        'R_ring_intermediate_joint': 0.0, 
        'R_thumb_proximal_pitch_joint': 0.52, 
        'L_thumb_intermediate_joint': 0.0, 
        'R_thumb_intermediate_joint': 0.0, 
        'L_thumb_distal_joint':  0.0, 
        'R_thumb_distal_joint': 0.0
        }
    
    class env(LeggedRobotCfg.env):
        # 3 + 3 + 3 + 3 + 53 + 53 + 53 = 171
        num_observations = 171
        # num_privileged_obs = 174
        num_actions = 53#12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {
                    'left_hip_pitch_joint': 200.0, 
                    'right_hip_pitch_joint': 200.0, 
                    'waist_yaw_joint': 200.0, 
                    'left_hip_roll_joint': 150.0,
                    'right_hip_roll_joint': 150.0, 
                    'waist_roll_joint': 200.0, 
                    'left_hip_yaw_joint': 150.0, 
                    'right_hip_yaw_joint': 150.0, 
                    'waist_pitch_joint':  200.0,
                    'left_knee_joint':  200.0, 
                    'right_knee_joint': 200.0, 
                    'left_shoulder_pitch_joint': 40.0, 
                    'right_shoulder_pitch_joint': 40.0, 
                    'left_ankle_pitch_joint': 20.0, 
                    'right_ankle_pitch_joint': 20.0, 
                    'left_shoulder_roll_joint': 40.0, 
                    'right_shoulder_roll_joint': 40.0,
                    'left_ankle_roll_joint': 20.0, 
                    'right_ankle_roll_joint': 20.0, 
                    'left_shoulder_yaw_joint': 40.0, 
                    'right_shoulder_yaw_joint': 40.0, 
                    'left_elbow_joint': 40.0, 
                    'right_elbow_joint': 40.0, 
                    'left_wrist_roll_joint': 40.0, 
                    'right_wrist_roll_joint': 40.0,
                    'left_wrist_pitch_joint':  40.0, 
                    'right_wrist_pitch_joint': 40.0, 
                    'left_wrist_yaw_joint': 40.0, 
                    'right_wrist_yaw_joint': 40.0, 
                    'L_index_proximal_joint': 40.0, 
                    'L_middle_proximal_joint': 40.0, 
                    'L_pinky_proximal_joint': 40.0, 
                    'L_ring_proximal_joint': 40.0,
                    'L_thumb_proximal_yaw_joint': 40.0, 
                    'R_index_proximal_joint': 40.0, 
                    'R_middle_proximal_joint': 40.0, 
                    'R_pinky_proximal_joint': 40.0, 
                    'R_ring_proximal_joint': 40.0, 
                    'R_thumb_proximal_yaw_joint': 40.0, 
                    'L_index_intermediate_joint': 40.0, 
                    'L_middle_intermediate_joint': 40.0, 
                    'L_pinky_intermediate_joint': 40.0, 
                    'L_ring_intermediate_joint': 40.0, 
                    'L_thumb_proximal_pitch_joint': 40.0, 
                    'R_index_intermediate_joint': 40.0, 
                    'R_middle_intermediate_joint': 40.0, 
                    'R_pinky_intermediate_joint': 40.0, 
                    'R_ring_intermediate_joint': 40.0, 
                    'R_thumb_proximal_pitch_joint': 40.0, 
                    'L_thumb_intermediate_joint': 40.0, 
                    'R_thumb_intermediate_joint': 40.0, 
                    'L_thumb_distal_joint': 40.0, 
                    'R_thumb_distal_joint': 40.0


                    #  ".*_hip_yaw_joint": 150.0,
                    # ".*_hip_roll_joint": 150.0,
                    # ".*_hip_pitch_joint": 200.0,
                    # ".*_knee_joint": 200.0,
                    # "waist_pitch_joint": 200.0,
                    # "waist_roll_joint": 200.0,
                    # "waist_yaw_joint": 200.0,
                    # ".*_ankle_pitch_joint": 20.0,
                    # ".*_ankle_roll_joint": 20.0,
                    # ".*_shoulder_pitch_joint": 40.0,
                    # ".*_shoulder_roll_joint": 40.0,
                    # ".*_shoulder_yaw_joint": 40.0,
                    # ".*_elbow_joint": 40.0,
                    # ".*_wrist_roll_joint": 40.0,
                    # ".*_wrist_pitch_joint": 40.0,
                    # ".*_wrist_yaw_joint": 40.0,
                    # "R_.*": 40.0,
                    # "L_.*": 40.0,
                     }  # [N*m/rad]
        damping = {  
                'left_hip_pitch_joint': 5.0, 
                    'right_hip_pitch_joint': 5.0, 
                    'waist_yaw_joint': 5.0, 
                    'left_hip_roll_joint': 5.0,
                    'right_hip_roll_joint': 5.0, 
                    'waist_roll_joint': 5.0, 
                    'left_hip_yaw_joint': 5.0, 
                    'right_hip_yaw_joint': 5.0, 
                    'waist_pitch_joint':  5.0,
                    'left_knee_joint':  5.0, 
                    'right_knee_joint': 5.0, 
                    'left_shoulder_pitch_joint': 10.0, 
                    'right_shoulder_pitch_joint': 10.0, 
                    'left_ankle_pitch_joint': 2.0, 
                    'right_ankle_pitch_joint': 2.0, 
                    'left_shoulder_roll_joint': 10.0, 
                    'right_shoulder_roll_joint': 10.0,
                    'left_ankle_roll_joint': 2.0, 
                    'right_ankle_roll_joint': 2.0, 
                    'left_shoulder_yaw_joint': 10.0, 
                    'right_shoulder_yaw_joint': 10.0, 
                    'left_elbow_joint': 10.0, 
                    'right_elbow_joint': 10.0, 
                    'left_wrist_roll_joint': 10.0, 
                    'right_wrist_roll_joint': 10.0,
                    'left_wrist_pitch_joint':  10.0, 
                    'right_wrist_pitch_joint': 10.0, 
                    'left_wrist_yaw_joint': 10.0, 
                    'right_wrist_yaw_joint': 10.0, 
                    'L_index_proximal_joint': 10.0, 
                    'L_middle_proximal_joint': 10.0, 
                    'L_pinky_proximal_joint': 10.0, 
                    'L_ring_proximal_joint': 10.0,
                    'L_thumb_proximal_yaw_joint': 10.0, 
                    'R_index_proximal_joint': 10.0, 
                    'R_middle_proximal_joint': 10.0, 
                    'R_pinky_proximal_joint': 10.0, 
                    'R_ring_proximal_joint': 10.0, 
                    'R_thumb_proximal_yaw_joint': 10.0, 
                    'L_index_intermediate_joint': 10.0, 
                    'L_middle_intermediate_joint': 10.0, 
                    'L_pinky_intermediate_joint': 10.0, 
                    'L_ring_intermediate_joint': 10.0, 
                    'L_thumb_proximal_pitch_joint': 10.0, 
                    'R_index_intermediate_joint': 10.0, 
                    'R_middle_intermediate_joint': 10.0, 
                    'R_pinky_intermediate_joint': 10.0, 
                    'R_ring_intermediate_joint': 10.0, 
                    'R_thumb_proximal_pitch_joint': 10.0, 
                    'L_thumb_intermediate_joint': 10.0, 
                    'R_thumb_intermediate_joint': 10.0, 
                    'L_thumb_distal_joint':  10.0, 
                    'R_thumb_distal_joint': 10.0,

                #      ".*_hip_yaw_joint": 5.0,
                # ".*_hip_roll_joint": 5.0,
                # ".*_hip_pitch_joint": 5.0,
                # ".*_knee_joint": 5.0,
                # "waist_pitch_joint": 5.0,
                # "waist_roll_joint": 5.0,
                # "waist_yaw_joint": 5.0,
                # ".*_ankle_pitch_joint": 2.0,
                # ".*_ankle_roll_joint": 2.0,
                # ".*_shoulder_pitch_joint": 10.0,
                # ".*_shoulder_roll_joint": 10.0,
                # ".*_shoulder_yaw_joint": 10.0,
                # ".*_elbow_joint": 10.0,
                # ".*_wrist_roll_joint": 10.0,
                # ".*_wrist_pitch_joint": 10.0,
                # ".*_wrist_yaw_joint": 10.0,
                # "R_.*": 10.0,
                # "L_.*": 10.0,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 128]
        critic_hidden_dims = [256, 128, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.008
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        max_iterations = 5000
        run_name = ''
        experiment_name = 'g1'

  
