


# from pfc.gru_pred_conv_8x8 import GruPredHid
from pfc.gru_pfc_gym_hybrid import PPO_GRU_Module
from mlutils.ml import one_hot
from hyperparam import Hybrid_PPO_HyperParam as hp
import paddle
import numpy as np
import copy
import logging
import time
"minigrid"
def env_obs_to_paddle_vec(obs):
    obs=paddle.to_tensor(np.reshape(obs,[1,1,-1]).astype(np.float32))
    return obs
"find ball"
# def env_obs_to_paddle_vec(obs):
#     return paddle.to_tensor(np.reshape(np.array(obs,dtype=np.float32),[1,1,-1]))

def main():
    logging.basicConfig(filename="log/"+time.strftime("%Y_%m%d_%H%M%S",time.localtime(time.time()))+".txt", level=logging.INFO)
    
    "超参数写入log"
    for attr in dir(hp):
        if attr.startswith("__") and attr.endswith("__"):
            # print(None)
            pass
        else:
            print(f"{attr}={getattr(hp,attr)}")
            logging.info(f"{attr}={getattr(hp,attr)}")

    import gym
    # import gym_hybrid
    from mlutils.env import EnvNeverDone
    # game_name="LunarLander-v2"
    # game_name="LunarLanderContinuous-v2"
    # game_name="BipedalWalker-v3"
    env = gym.make(hp.GAME_NAME)
    """
    continuous: type:gym.spaces.box.Box action_shape:.shape
    discrete: type:gym.spaces.discrete.Discrete action_shape.n

    """
    game_type="discrete" if isinstance(env.action_space,gym.spaces.discrete.Discrete) else "continuous"
    # print(list(env.observation_space.shape),env.action_space.shape)
    logging.info(f"gamename={hp.GAME_NAME}")
    logging.info(f"gametype={game_type}")
    env = EnvNeverDone(env)

    # gym.spaces.box.Box
    input_output_info={
        # "game_name":"findball",
        'to_static':True,
        'env_vec_dim':env.observation_space.shape[0],
        'obs_dim':env.observation_space.shape[0],
        'action_env_discrete_dim':env.action_space.n if game_type=="discrete" else 2,
        'action_env_continuous_dim':env.action_space.shape[0] if game_type=="continuous" else 2,
        'actor_gru_hid_dim':256,
        'critic_gru_hid_dim':256
        }
    pfc_interact=PPO_GRU_Module(input_output_info)
    pfc_interact.model.eval() 


    input_output_info_train=copy.deepcopy(input_output_info)
    input_output_info_train['to_static']=False
    pfc_train=PPO_GRU_Module(input_output_info_train)
    if hp.GAME_LOAD_MODEL is not None:

        pfc_train.load_model('all_models/pfc_model',hp.GAME_LOAD_MODEL)
    pfc_interact.update(pfc_train.model.state_dict())
    

    "初始化"
    env_obs=env.reset()
    # kl_pred_net.pred_reset()
    # env_obs=np.reshape(np.transpose(env_obs,[2,0,1]).astype(np.float32),[-1])/255
    # input_dict=from_env_obs_to_input_dict(env_obs_dict,0)
    input_dict={
        "env_vec":env_obs_to_paddle_vec(env_obs)
    }
    pfc_interact._reset()
    pfc_train._reset()
    "初始化rpm第0步的obs"
    pfc_train.rpm.env_vec[0]=env_obs_to_paddle_vec(env_obs).numpy().reshape([-1])

    for e in range(10000000000000000000):
        ppo_reward=0
        ppo_inner_reward=0
        ppo_env_reward=0
        env_ep_time=0
        pfc_ep_time=0
        bg_ep_time=0

        pfc_interact.update(pfc_train.model.state_dict())
        # pfc_interact._reset()
        

        # print("init c_h",np.mean(np.abs(pfc_interact.agent.alg.model.actor_model.h.numpy())))
        
        with paddle.no_grad():
            for j in range(hp.PPO_NUM_STEPS):
                # tt.start()

                h_dict={
                    'actor_init_h':None,
                    'critic_init_h':None
                }
                

                pfc_output_dict=pfc_interact._input(input_dict,train=False,h_dict=h_dict)
                """
                output_dict={
                    "value":np.squeeze(value),
                    "action":np.squeeze(action),
                    "action_log_prob":np.squeeze(action_log_prob),
                    "actor_gru_h1":actor_gru_h1,
                    "actor_gru_h2":actor_gru_h2,
                    "critic_gru_h":critic_gru_h
                }
                """
                pfc_value,pfc_action_discrete,pfc_action_discrete_log_prob,pfc_action_continuous,pfc_action_continuous_log_prob,pfc_actor_gru_h,pfc_critic_gru_h=[pfc_output_dict[key] for key in ['value','action_discrete','action_discrete_log_probs','action_continuous','action_continuous_log_probs','actor_gru_h','critic_gru_h']]

                # print(pfc_action_discrete)
                "inner reward"
                if game_type=="discrete":
                    next_env_obs,reward,done,info=env.step(pfc_action_discrete)
                elif game_type=="continuous":
                    # (env.action_space.high-env.action_space.low)/2+env.action_space.low
                    # pfc_action_continuous_for_env=(np.tanh(pfc_action_continuous)+1)/2*(env.action_space.high-env.action_space.low)+env.action_space.low
                    pfc_action_continuous_for_env=(np.clip(pfc_action_continuous,-1,1)+1)/2*(env.action_space.high-env.action_space.low)+env.action_space.low
                    # print(pfc_action_continuous_for_env)
                    next_env_obs,reward,done,info=env.step(pfc_action_continuous_for_env)
                reward=np.maximum(reward,-10)


                next_input_dict={"env_vec":env_obs_to_paddle_vec(next_env_obs)}
                # tt.end_and_start("env")
                # print("reward and done",reward,done)
                # print("reward",reward)
                # print("info['inner_reward']=",info['inner_reward'])
                # print(reward)
                if hp.GAME_RENDER:
                    env.render()
                

                # if done:
                #     next_input_dict={"env_vec":env_obs_to_paddle_vec(env.reset())}
                    # pfc_interact._reset()

                # ppo_inner_reward+=info['inner_reward']
                # ppo_env_reward+=info['out_reward']
                ppo_reward+=reward

                masks = paddle.to_tensor(
                    [[1.0]] , dtype='float32')
                bad_masks = paddle.to_tensor(
                    [[1.0]],
                    dtype='float32')
                pfc_train._rpm_collect({
                    # 'env_img':np.transpose(np.concatenate([next_env_obs_dict['ball_obs'],next_env_obs_dict['label_img']],axis=-1),[2,0,1]),
                    'env_vec':env_obs_to_paddle_vec(next_env_obs).numpy(),
                    # 'hpc_img':np.transpose(next_env_obs_dict['hpc_img'],[2,0,1]),
                    # 'hpc_vec':next_env_obs_dict['ask_site_norm'],
                    # "model_probs":pfc_output_dict['model_probs'],
                    'action_discrete':pfc_action_discrete,
                    'action_discrete_log_prob':pfc_action_discrete_log_prob,
                    'action_continuous':pfc_action_continuous,
                    'action_continuous_log_prob':pfc_action_continuous_log_prob,
                    # 'action_log_prob':pfc_action_log_prob,
                    'value_pred':pfc_value,
                    'reward':reward,
                    'actor_gru_h':pfc_actor_gru_h,
                    # 'actor_gru2_h':pfc_actor_gru2_h,
                    'critic_gru_h':pfc_critic_gru_h,
                    'mask':masks,
                    'bad_mask':bad_masks,
                })
                # print(np.mean(np.abs(pfc_train.rpm.actor_gru_h[0])))
                input_dict=next_input_dict

            
        # print("action_abs_mean",np.mean(one_hot(np.array(pfc_train.rpm.actions,np.int64).reshape([-1]),env.action_dim_discrete),axis=0))
        print(f"iter {e} ppo reward={ppo_reward/hp.PPO_NUM_STEPS}")
        

        value_loss_epoch, action_continuous_loss_epoch,action_continuous_dist_entropy_epoch,action_discrete_loss_epoch, action_discrete_dist_entropy_epoch,value_clip_mean_ratio_epoch=pfc_train._learn()
        logging.info(
                f"|iter={e}"+
                f"|inner_reward={ppo_inner_reward/hp.PPO_NUM_STEPS}"+
                f"|env_reward={ppo_env_reward/hp.PPO_NUM_STEPS}"+
                f"|reward={ppo_reward/hp.PPO_NUM_STEPS}"+
                # f"|abs_action_mean={np.mean(np.abs(pfc_train.rpm.actions))}"+
                f"|v_loss={value_loss_epoch[0]}"+
                f"|action_discrete_loss={action_discrete_loss_epoch[0]}"+
                f"|action_discrete_dist_entropy={action_discrete_dist_entropy_epoch[0]}"+
                f"|action_continuous_loss={action_continuous_loss_epoch[0]}"+
                f"|action_continuous_dist_entropy={action_continuous_dist_entropy_epoch[0]}"+
                f"|value_clip_mean_ratio_epoch={value_clip_mean_ratio_epoch[0]}"
                )

        if e%100==0:
            pfc_train.save_model("all_models/pfc_model",e)



                
if __name__=="__main__":
    main()