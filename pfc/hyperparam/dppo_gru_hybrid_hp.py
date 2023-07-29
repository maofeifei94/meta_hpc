class DPPO_Hybrid_HyperParam:
    """
    'LunarLander-v2'
    'LunarLanderContinuous-v2'
    'BipedalWalker-v3'
    'BipedalWalkerHardcore-v3'
    """
    

    GAME_NAME='Breakout-ram-v4'
    GAME_LOAD_MODEL=None
    GAME_RENDER=0

    WORKER_NUM=4

    PPO_ACTOR_CONTINUOUS_SCALE_STD=2
    PPO_ACTOR_CONTINUOUS_MIN_STD=0.5
    PPO_ACTOR_CONTINUOUS_SCALE_LOGITS=0.05

    PPO_GRU_HID_DIM=128
    PPO_TRAIN_HISTORY_LEN=128

    PPO_GAMMA = 0.99
    PPO_EPS = 1e-12  # Adam optimizer epsilon (default: 1e-5)
    PPO_GAE_LAMBDA = 0.9 # Lambda parameter for calculating N-step advantage

    PPO_VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
    PPO_MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping
    PPO_NUM_STEPS = 2048  # data collecting time steps (ie. T in the paper

    PPO_ACTOR_LR = 1e-4
    PPO_CRITIC_LR = 1e-4

    PPO_ACTOR_DISCRETE_LOSS_RATIO=1.0 #discrete 损失系数
    PPO_ACTOR_DISCRETE_ENTROPY_COEF = 0.0 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_DISCRETE_CLIP_PARAM = 0.1

    PPO_ACTOR_CONTINUOUS_LOSS_RATIO=0.00 #continuous 部分损失系数
    PPO_ACTOR_CONTINUOUS_ENTROPY_COEF = 0.000 # Entropy coefficient (ie. c_2 in the paper)
    PPO_ACTOR_CONTINUOUS_CLIP_PARAM = 0.03

    PPO_CRITIC_USE_CLIPPED_VALUE_LOSS=True #是否对value更新启用截断
    PPO_CRITIC_CLIPPED_VALUE_RATIO=0.1#每一个epoch，value与target的距离降低多少比例(1代表以target为目标，0.5代表target与pred的中点为目标)

    PPO_CRITIC_TRAIN_RATE=1
    PPO_ACTOR_TRAIN_RATE=1
    PPO_EPOCH = 10  #3 number of epochs for updating using each T data (ie K in the paper)
    PPO_BATCH_SIZE = 64
    PPO_BATCH_NUM = 10 #8



