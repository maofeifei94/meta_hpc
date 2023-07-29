from env.minigrid import MiniGrid_fourroom_obs8x8
from pfc.gru_pred_conv_8x8 import GruPredHid
from pfc.gru_pfc_minigrid import PPO_GRU_Module
from myf_ML_util import timer_tool
from mlutils.ml import continue_to_discrete,moving_avg
from mlutils.ml import one_hot
import hyperparam as hp
import paddle
import numpy as np
import copy
def env_obs_to_paddle_vec(obs):
    obs=np.transpose(obs,[2,0,1])
    obs=np.reshape(obs,[1,1,-1]).astype(np.float32)
    obs=obs/255
    obs=paddle.to_tensor(obs)
    return obs



def main():
    cd=continue_to_discrete(3,hard_mode=True)
    env=MiniGrid_fourroom_obs8x8()
    input_output_info={
        'to_static':True,
        'train_batchsize':4,
        'env_vec_dim':8*8*3,
        'obs_dim':8*8*3,
        'action_env_dim':2,
        'action_dim':2,
        'actor_gru_hid_dim':256,
        'critic_gru_hid_dim':256
        }
    pfc_interact=PPO_GRU_Module(input_output_info)
    pfc_interact.model.eval() 

    kl_pred_net=GruPredHid()
    kl_pred_net.load_model("all_models/minigrid_model",138000)
    kl_pred_net.eval()

    input_output_info_train=copy.deepcopy(input_output_info)
    input_output_info_train['to_static']=False
    pfc_train=PPO_GRU_Module(input_output_info_train)
    # pfc_train.load_model('all_models/pfc',80)

    "初始化"
    env_obs=env.reset()
    kl_pred_net.pred_reset()
    # env_obs=np.reshape(np.transpose(env_obs,[2,0,1]).astype(np.float32),[-1])/255
    # input_dict=from_env_obs_to_input_dict(env_obs_dict,0)
    input_dict={
        "env_vec":env_obs_to_paddle_vec(env_obs)
    }
    pfc_interact._reset()
    pfc_train._reset()
    tt_main=timer_tool("train pfc",_debug=True)

    avg_ppo_reward=moving_avg()


    for e in range(10000000000000000000):
        ppo_reward=0
        env_ep_time=0
        pfc_ep_time=0
        bg_ep_time=0

        pfc_interact.update(pfc_train.model.state_dict())

        tt_main.start()
        with paddle.no_grad():
            for j in range(hp.PPO_NUM_STEPS):
                t_j=timer_tool("j step",False)
                # tt.start()

                h_dict={
                    'actor_init_h':None,
                    'critic_init_h':None
                }
                # print(j)
                # print(input_dict.keys())
                pfc_output_dict=pfc_interact._input(input_dict,train=None,h_dict=h_dict)
                t_j.end_and_start("pfc")
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
                pfc_value,pfc_action,pfc_action_log_prob,pfc_actor_gru_h,pfc_critic_gru_h=[pfc_output_dict[key] for key in ['value','action','action_log_prob','actor_gru_h','critic_gru_h']]
                discrete_pfc_env_action=cd.to_discrete(np.tanh(pfc_action))
                # print(pfc_action.shape)
                # input()
                # pfc_action_env=cd.to_discrece(np.tanh(pfc_action))

                # print(pfc_action)
                # pfc_action_hpc=(np.tanh(pfc_action[2:])+1)/2
                "inner reward"
                real_kl,pred_kl=kl_pred_net.pred(input_dict['env_vec'].numpy(),one_hot(discrete_pfc_env_action,3))
                # print("env_action",pfc_action,one_hot(discrete_pfc_env_action,3))
                inner_reward=(pred_kl+real_kl)/10
                # print("kl",real_kl,pred_kl)

                # env.render()

                next_env_obs,reward,done=env.step(discrete_pfc_env_action)

                
                reward=reward*10+inner_reward
                


                next_input_dict={"env_vec":env_obs_to_paddle_vec(next_env_obs)}
                t_j.end_and_start("env")
                # tt.end_and_start("env")
                # print("reward and done",reward,done)
                

                if done:
                    next_input_dict={"env_vec":env_obs_to_paddle_vec(env.reset())}
                    kl_pred_net.pred_reset()
                    pfc_interact._reset()

                    # pfc_interact._reset()

                ppo_reward+=reward

                masks = paddle.to_tensor(
                    [[0.0]] if done else [[1.0]] , dtype='float32')
                bad_masks = paddle.to_tensor(
                    [[1.0]],
                    dtype='float32')
                pfc_train._rpm_collect({
                    # 'env_img':np.transpose(np.concatenate([next_env_obs_dict['ball_obs'],next_env_obs_dict['label_img']],axis=-1),[2,0,1]),
                    'env_vec':env_obs_to_paddle_vec(next_env_obs).numpy(),
                    # 'hpc_img':np.transpose(next_env_obs_dict['hpc_img'],[2,0,1]),
                    # 'hpc_vec':next_env_obs_dict['ask_site_norm'],
                    'action':pfc_action,
                    'action_log_prob':pfc_action_log_prob,
                    'value_pred':pfc_value,
                    'reward':reward,
                    'actor_gru_h':pfc_actor_gru_h,
                    # 'actor_gru2_h':pfc_actor_gru2_h,
                    'critic_gru_h':pfc_critic_gru_h,
                    'mask':masks,
                    'bad_mask':bad_masks,
                })
                input_dict=next_input_dict
                t_j.end_and_start("collect")
                t_j.analyze()
            
        avg_ppo_reward.update(ppo_reward)
        print(f"iter {e} ppo reward={avg_ppo_reward}")
        # print(tt.debug)
        tt_main.end_and_start("collect")

        pfc_train._learn()
        

        tt_main.end_and_start("learn")

        if e%20==0:
            pfc_train.save_model("all_models/pfc_model",e)



                
if __name__=="__main__":
    main()