

env_action_discrete_num=15
ec_action_continuous_dim=4


#cae_hyperparam
class cae_hyperparam:
    model_dir='all_models/cae_model'
    model_load='newest'
    v_encoder_hid_c=4
    v_z_c=8
    v_decoder_hid_c=4

    v_encoder_hid_c_8x8=4
    v_z_c_8x8=2
    v_decoder_hid_c_8x8=4

    feature_vec_dim=8*8*(v_z_c+v_z_c_8x8)
    use_vae=True
    nl_func='Swish'
    img_shape=[3,64,64]
    kl_loss_ratio=0.000000001
    var=0.00001
    lr=0.0001
class ec_info:
    model_dir='all_models/ec_model'
    model_load='newest'
    recon_seq_len=64

    "input dim"
    gloinfo_gauss_dim=256
    state_vec_dim=cae_hyperparam.feature_vec_dim
    action_vec_dim=env_action_discrete_num

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=1e-3

    # gloinfo
    "input fc"
    gloinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    gloinfo_ifc_mid_dim=256
    gloinfo_ifc_mid_layers=4
    gloinfo_ifc_output_dim=256
    "gru"
    gloinfo_gru_dim=256
    "gauss_fc"
    gloinfo_gfc_input_dim=gloinfo_gru_dim
    gloinfo_gfc_mid_dim=256
    gloinfo_gfc_mid_layers=4
    gloinfo_gfc_output_dim=gloinfo_gauss_dim*2

    # locinfo
    "input fc"
    locinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    locinfo_ifc_mid_dim=256
    locinfo_ifc_mid_layers=4
    locinfo_ifc_output_dim=256
    "gru"
    locinfo_gru_dim=256

    #predfc
    "pred fc"
    pred_fc_input_dim=locinfo_gru_dim+gloinfo_gauss_dim+action_vec_dim
    pred_fc_mid_dim=256
    pred_fc_mid_layers=6
    pred_fc_output_dim=state_vec_dim
#PPO parameters
class DPPO_Hybrid_HyperParam:
    """
    'LunarLander-v2'
    'LunarLanderContinuous-v2'
    'BipedalWalker-v3'
    'BipedalWalkerHardcore-v3'
    'Breakout-ram-v4'
    'MsPacman-ram-v4'
    """
    
    "env"
    # GAME_NAME='Breakout-ram-v4'
    # GAME_LOAD_MODEL=5000
    # GAME_RENDER=1

    # GAME_OBS_SCALE=1.0/255
    # GAME_DATA_SAVE_DIR="all_data/breakout"
    # GAME_DATA_SAVE_SIZE=200*2000
    MODEL_SAVE_DIR="all_models/ppo_model"
    WORKER_NUM=4
    WORKER_USE_CPU=True
    WORKER_USE_CPU_INTERVAL=1

    "net structure"
    PPO_ACTOR_ENCODER_FC_DIM=128
    PPO_ACTOR_HEAD_FC_DIM=128
    PPO_CRITIC_ENCODER_FC_DIM=128
    PPO_CRITIC_HEAD_FC_DIM=128
    PPO_GRU_HID_DIM=128

    PPO_ACTOR_CONTINUOUS_SCALE_STD=2
    PPO_ACTOR_CONTINUOUS_MIN_STD=0.1
    PPO_ACTOR_CONTINUOUS_SCALE_LOGITS=0.05

    "RL common param"
    PPO_TRAIN_HISTORY_LEN=128

    PPO_ACTOR_LR = 1e-4
    PPO_CRITIC_LR = 1e-4
    
    PPO_GAMMA = 0.99
    PPO_EPS = 1e-12  # Adam optimizer epsilon (default: 1e-5)
    PPO_GAE_LAMBDA = 0.5 # Lambda parameter for calculating N-step advantage

    PPO_VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
    PPO_MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping

    "PPO param"
    PPO_NUM_STEPS = 2048  # data collecting time steps (ie. T in the paper
    
    PPO_ACTOR_DISCRETE_LOSS_RATIO=1.0 #discrete 损失系数
    PPO_ACTOR_DISCRETE_ENTROPY_COEF = 0.000001 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_DISCRETE_CLIP_PARAM = 0.02

    PPO_ACTOR_CONTINUOUS_LOSS_RATIO=0.00 #continuous 部分损失系数
    PPO_ACTOR_CONTINUOUS_ENTROPY_COEF = 0.00001 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_CONTINUOUS_CLIP_PARAM = 0.2

    PPO_CRITIC_USE_CLIPPED_VALUE_LOSS=True #是否对value更新启用截断
    PPO_CRITIC_CLIPPED_VALUE_RATIO=0.3#每一个epoch，value与target的距离降低多少比例(1代表以target为目标，0.5代表target与pred的中点为目标)

    PPO_CRITIC_TRAIN_INTERVAL=1
    PPO_ACTOR_TRAIN_INTERVAL=1
    PPO_EPOCH = 5  #3 number of epochs for updating using each T data (ie K in the paper)
    PPO_BATCH_SIZE = 64
    PPO_BATCH_NUM = 10 #8

class brain_hyperparam:
    queue_max_size=5
    env_action_discrete_num=env_action_discrete_num
    ec_action_continuous_dim=ec_action_continuous_dim

    train_batch_size=4
    train_seq_len=512
    #当iter%interval==0时执行一次
    # interval_train=1
    interval_train_cae=30
    inner_reward_ratio=0.001

    
    rpm_max_size=int(1e5)
    rpm_start_train_size=int(2048*32)
    rpm_buffer_size=int(2048*32)

    cae_hyperparam=cae_hyperparam
    ec_info=ec_info
    DPPO_Hybrid_HyperParam=DPPO_Hybrid_HyperParam
    
    PPO_input_output_info={
        'to_static':False,
        'env_vec_dim':cae_hyperparam.feature_vec_dim+ec_info.gloinfo_gauss_dim*2+ec_info.locinfo_gru_dim,
        'obs_dim':cae_hyperparam.feature_vec_dim+ec_info.gloinfo_gauss_dim*2+ec_info.locinfo_gru_dim,
        'action_env_discrete_dim':env_action_discrete_num,
        'action_env_continuous_dim':ec_action_continuous_dim,
        'actor_gru_hid_dim':128,
        'critic_gru_hid_dim':128,
        'hyperparam':DPPO_Hybrid_HyperParam
    }

    