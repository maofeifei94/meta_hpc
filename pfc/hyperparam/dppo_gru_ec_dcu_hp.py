class DPPO_Ec_HyperParam:
    """
    'LunarLander-v2'
    'LunarLanderContinuous-v2'
    'BipedalWalker-v3'
    'BipedalWalkerHardcore-v3'
    """
    
    "env"
    GAME_NAME='BipedalWalkerHardcore-v3'
    GAME_LOAD_MODEL=None
    GAME_RENDER=0

    WORKER_NUM=1
    WORKER_USE_CPU=True
    WORKER_USE_CPU_INTERVAL=1

    "net structure"
    PPO_ACTOR_ENCODER_FC_DIM=64
    PPO_ACTOR_HEAD_FC_DIM=64
    PPO_CRITIC_ENCODER_FC_DIM=64
    PPO_CRITIC_HEAD_FC_DIM=64
    PPO_GRU_HID_DIM=64

    PPO_ACTOR_CONTINUOUS_SCALE_STD=2
    PPO_ACTOR_CONTINUOUS_MIN_STD=0.1
    PPO_ACTOR_CONTINUOUS_SCALE_LOGITS=0.1


    "RL common param"
    PPO_TRAIN_HISTORY_LEN=128

    PPO_ACTOR_LR = 1e-4
    PPO_CRITIC_LR = 1e-4
    
    PPO_GAMMA = 0.99
    PPO_EPS = 1e-12  # Adam optimizer epsilon (default: 1e-5)
    PPO_GAE_LAMBDA = 0.95 # Lambda parameter for calculating N-step advantage

    PPO_VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
    PPO_MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping



    "PPO param"
    PPO_NUM_STEPS = 2048  # data collecting time steps (ie. T in the paper
    
    PPO_ACTOR_DISCRETE_LOSS_RATIO=0.0 #discrete 损失系数
    PPO_ACTOR_DISCRETE_ENTROPY_COEF = 0.0 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_DISCRETE_CLIP_PARAM = 0.1

    PPO_ACTOR_CONTINUOUS_LOSS_RATIO=1.00 #continuous 部分损失系数
    PPO_ACTOR_CONTINUOUS_ENTROPY_COEF = 0.001 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_CONTINUOUS_CLIP_PARAM = 0.1

    PPO_CRITIC_USE_CLIPPED_VALUE_LOSS=True #是否对value更新启用截断
    PPO_CRITIC_CLIPPED_VALUE_RATIO=0.2#每一个epoch，value与target的距离降低多少比例(1代表以target为目标，0.5代表target与pred的中点为目标)

    PPO_CRITIC_TRAIN_INTERVAL=1
    PPO_ACTOR_TRAIN_INTERVAL=1
    PPO_EPOCH = 5  #3 number of epochs for updating using each T data (ie K in the paper)
    PPO_BATCH_SIZE = 64
    PPO_BATCH_NUM = 5 #8

class env_param():
    import gym
    env = gym.make(DPPO_Ec_HyperParam.GAME_NAME)
    env_obs_dim=env.observation_space.shape[0]
    env_action_dim=env.action_space.shape[0]
    add_reward_to_ec=1


class trainer_info():
    env_name='BreakoutDeterministic-v4'

    "ec"
    ec_train_freq=10
    ec_rpm_size=int(1e6)
    ec_train_min_rpm_size=int(2e4)

    ec_svaepred_load_model="newest"
    ec_svaepred_model_dir="all_models/ec_svaepred_model"
    ec_svaepred_train=False
    ec_svaepred_train_batch_size=256
    ec_svaepred_train_history_len=128


class svaepred_info():

    "input dim"
    gloinfo_gauss_dim=256
    state_vec_dim=env_param.env_obs_dim+env_param.add_reward_to_ec
    action_vec_dim=env_param.env_action_dim

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=1e-8

    # gloinfo
    "input fc"
    gloinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    gloinfo_ifc_mid_dim=128
    gloinfo_ifc_mid_layers=3
    gloinfo_ifc_output_dim=128
    "gru"
    gloinfo_gru_dim=128
    "gauss_fc"
    gloinfo_gfc_input_dim=gloinfo_gru_dim
    gloinfo_gfc_mid_dim=128
    gloinfo_gfc_mid_layers=3
    gloinfo_gfc_output_dim=gloinfo_gauss_dim*2

    # locinfo
    "input fc"
    locinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    locinfo_ifc_mid_dim=128
    locinfo_ifc_mid_layers=3
    locinfo_ifc_output_dim=128
    "gru"
    locinfo_gru_dim=128

    #predfc
    pred_fc_input_dim=locinfo_gru_dim+gloinfo_gauss_dim
    pred_fc_mid_dim=256
    pred_fc_mid_layers=4
    pred_fc_output_dim=state_vec_dim

input_output_info={
    'to_static':False,
    'env_vec_dim':env_param.env_obs_dim+svaepred_info.gloinfo_gauss_dim*2+svaepred_info.locinfo_gru_dim,
    'obs_dim':env_param.env_obs_dim+svaepred_info.gloinfo_gauss_dim*2+svaepred_info.locinfo_gru_dim,
    'action_env_discrete_dim':2,
    'action_env_continuous_dim':env_param.env_action_dim,
    'hyperparam':DPPO_Ec_HyperParam
    }