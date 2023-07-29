debug=False

class Hybrid_PPO_HyperParam:
    """
    'LunarLander-v2'
    'LunarLanderContinuous-v2'
    'BipedalWalker-v3'
    'BipedalWalkerHardcore-v3'
    """

    GAME_NAME='BipedalWalker-v3'
    GAME_LOAD_MODEL=2000
    GAME_RENDER=1

    PPO_ACTOR_CONTINUOUS_SCALE_STD=2
    PPO_ACTOR_CONTINUOUS_MIN_STD=0.01
    PPO_ACTOR_CONTINUOUS_SCALE_LOGITS=0.05

    PPO_GRU_HID_DIM=64
    PPO_TRAIN_HISTORY_LEN=128

    PPO_GAMMA = 0.99
    PPO_EPS = 1e-12  # Adam optimizer epsilon (default: 1e-5)
    PPO_GAE_LAMBDA = 0.9 # Lambda parameter for calculating N-step advantage

    PPO_VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
    PPO_MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping
    PPO_NUM_STEPS = 2048  # data collecting time steps (ie. T in the paper

    PPO_ACTOR_LR = 1e-4
    PPO_CRITIC_LR = 1e-4

    PPO_ACTOR_DISCRETE_LOSS_RATIO=0.0 #discrete 损失系数
    PPO_ACTOR_DISCRETE_ENTROPY_COEF = 0.0 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_DISCRETE_CLIP_PARAM = 0.1

    PPO_ACTOR_CONTINUOUS_LOSS_RATIO=1.00 #continuous 部分损失系数
    PPO_ACTOR_CONTINUOUS_ENTROPY_COEF = 0.000 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_CONTINUOUS_CLIP_PARAM = 0.03

    PPO_CRITIC_USE_CLIPPED_VALUE_LOSS=True #是否对value更新启用截断
    PPO_CRITIC_CLIPPED_VALUE_RATIO=0.1#每一个epoch，value与target的距离降低多少比例(1代表以target为目标，0.5代表target与pred的中点为目标)

    PPO_CRITIC_TRAIN_RATE=1
    PPO_ACTOR_TRAIN_RATE=1
    PPO_EPOCH = 5  #3 number of epochs for updating using each T data (ie K in the paper)
    PPO_BATCH_SIZE = 64
    PPO_BATCH_NUM = 5 #8
    
"PPO超参数"
PPO_GRU_HID_DIM=64
PPO_TRAIN_HISTORY_LEN=128
PPO_LR_C = 1e-4
PPO_LR_A = 1e-4
PPO_GAMMA = 0.99
PPO_EPS = 1e-5  # Adam optimizer epsilon (default: 1e-5)
PPO_GAE_LAMBDA = 0.0  # Lambda parameter for calculating N-step advantage

PPO_ENTROPY_COEF = 0.001  # Entropy coefficient (ie. c_2 in the paper)

PPO_VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
PPO_MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping
PPO_NUM_STEPS = 2048  # data collecting time steps (ie. T in the paper)
PPO_EPOCH = 3  # number of epochs for updating using each T data (ie K in the paper)
PPO_CLIP_PARAM = 0.1  # epsilon in clipping loss (ie. clip(r_t, 1 - epsilon, 1 + epsilon))
# PPO_BATCH_SIZE = 64
# PPO_BATCH_NUM = 32
PPO_BATCH_SIZE = 64
PPO_BATCH_NUM = 8

"SAC超参数"

SAC_LOG_SIG_MAX = 20.0      #log(方差)最大值
SAC_LOG_SIG_MIN = -20.0    #log(方差)最小值

# SAC_WARMUP_STEPS = 1e4
# SAC_MEMORY_SIZE = int(1e6)

SAC_WARMUP_STEPS = int(1e3)
# SAC_MEMORY_SIZE = int(1e4)

SAC_EVAL_EPISODES = 5

SAC_TRAIN_BATCH_SIZE = 128
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_ACTOR_LR = 0.0003
SAC_CRITIC_LR = 0.001

SAC_alpha=0.0001        #强化学习温度系数

SAC_LSTM_HID_DIM=32
SAC_LSTM_ACTION_VEC_DIM=8
SAC_max_ep_num=3000
SAC_max_ep_size=500
SAC_MEMORY_SIZE=int(1e6)
SAC_BATCH_SIZE=256
"DDPG 超参数"
Cereb_DDPG_GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
Cereb_DDPG_TAU = 0.005  # target_model 跟 model 同步参数 的 软更新参数
Cereb_DDPG_ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
Cereb_DDPG_CRITIC_LR = 0.0005  # Critic网络更新的 learning rate

Cereb_DDPG_MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
Cereb_DDPG_MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
Cereb_DDPG_BATCH_SIZE = 256 

Cereb_DDPG_WARMUP_STEPS = int(1e5)
Cereb_DDPG_MEMORY_SIZE = int(1e6)
"DQN"
DQN_gamma=0.95
DQN_lr=0.0003
DQN_MEMORY_SIZE=int(1e5)
"hpc参数"
hpc_hid_neuron_num=1000
hpc_encoder_fc_num_list=[2,64,64,256,256,256,256,hpc_hid_neuron_num]
hpc_decoder_fc_num_list=[hpc_hid_neuron_num,2]

#WM 参数
wm_dim=10
wm_capacity=10



