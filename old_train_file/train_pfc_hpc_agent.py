import paddle
import numpy as np
import matplotlib
import time
import os 
import copy
from pfc.gru_pfc_env_hpc import PPO_GRU_Module
from env.env_with_label import multi_circle_env
from myf_ML_util import timer_tool
import hyperparam as hp

def from_env_obs_to_input_dict(env_obs_dict,reward):
    """
    input_dict={
        'env_img'
        'env_vec'
        'hpc_img'
        'hpc_vec'
        'action_env'
        'action_hpc'
    }
    """
   
    ball_obs=env_obs_dict['ball_obs']
    label_img=env_obs_dict['label_img']
    ball_site_norm=env_obs_dict['ball_site_norm']

    ask_label_img=env_obs_dict['ask_label_img']
    ask_site_norm=env_obs_dict['ask_site_norm']

    # print(ball_obs.shape,label_img.shape)

    input_dict={
        'env_img':paddle.to_tensor(np.transpose(np.expand_dims(np.concatenate([ball_obs,label_img],axis=-1),axis=[0,1]),[0,1,4,2,3])),
        'env_vec':paddle.to_tensor(np.expand_dims(np.concatenate([ball_site_norm,[np.float32(reward)]],axis=-1),[0,1])),
        'hpc_img':paddle.to_tensor(np.transpose(np.expand_dims(ask_label_img,axis=[0,1]),[0,1,4,2,3])),
        'hpc_vec':paddle.to_tensor(np.expand_dims(ask_site_norm,[0,1])),
    }
    return input_dict


def main():
    env=multi_circle_env()
    input_output_info={
        'to_static':True,
        'train_batchsize':4,
        'env_img_shape':[4,64,64],
        'env_img_conv_vec_dim':256,
        'env_vec_dim':3,
        'hpc_img_shape':[1,64,64],
        'hpc_img_conv_vec_dim':256,
        'hpc_vec_dim':2,
        'action_env_dim':2,
        'action_hpc_dim':2,
        'action_dim':4,
        'actor_gru1_to_gru2_vec_dim':256,
        'actor_gru1_hid_dim':256,
        'actor_gru2_hid_dim':256,
        'critic_gru_hid_dim':256
        }
    pfc_interact=PPO_GRU_Module(input_output_info)
    pfc_interact.model.eval() 

    input_output_info_train=copy.deepcopy(input_output_info)
    input_output_info_train['to_static']=False
    pfc_train=PPO_GRU_Module(input_output_info_train)
    # pfc_train.load_model('all_models/pfc',80)

    "初始化"
    env_obs_dict=env.reset()
    input_dict=from_env_obs_to_input_dict(env_obs_dict,0)
    pfc_interact._reset()
    pfc_train._reset()
    tt_main=timer_tool("train pfc",_debug=True)
    # print(tt.debug)
    # print(pfc_interact.model.state_dict().keys())
    # input()

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
                    'actor_init_h1':None,
                    'actor_init_h2':None,
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
                pfc_value,pfc_action,pfc_action_log_prob,pfc_actor_gru1_h,pfc_actor_gru2_h,pfc_critic_gru_h=[pfc_output_dict[key] for key in ['value','action','action_log_prob','actor_gru_h1','actor_gru_h2','critic_gru_h']]
                # print(pfc_action.shape)
                # input()
                pfc_action_env=np.tanh(pfc_action[:2])
                pfc_action_hpc=(np.tanh(pfc_action[2:])+1)/2

                next_env_obs_dict,reward,done=env.step(pfc_action_env,pfc_action_hpc)
                next_input_dict=from_env_obs_to_input_dict(next_env_obs_dict,reward)
                t_j.end_and_start("env")
                # tt.end_and_start("env")

                if done:
                    next_input_dict=from_env_obs_to_input_dict(env.reset(),0)
                    # pfc_interact._reset()

                ppo_reward+=reward

                masks = paddle.to_tensor(
                    [[1.0]], dtype='float32')
                bad_masks = paddle.to_tensor(
                    [[1.0]],
                    dtype='float32')
                pfc_train._rpm_collect({
                    'env_img':np.transpose(np.concatenate([next_env_obs_dict['ball_obs'],next_env_obs_dict['label_img']],axis=-1),[2,0,1]),
                    'env_vec':np.concatenate([next_env_obs_dict['ball_site_norm'],[np.float32(reward)]],axis=-1),
                    'hpc_img':np.transpose(next_env_obs_dict['hpc_img'],[2,0,1]),
                    'hpc_vec':next_env_obs_dict['ask_site_norm'],
                    'action':pfc_action,
                    'action_log_prob':pfc_action_log_prob,
                    'value_pred':pfc_value,
                    'reward':reward,
                    'actor_gru1_h':pfc_actor_gru1_h,
                    'actor_gru2_h':pfc_actor_gru2_h,
                    'critic_gru_h':pfc_critic_gru_h,
                    'mask':masks,
                    'bad_mask':bad_masks,
                })
                input_dict=next_input_dict
                t_j.end_and_start("collect")
                t_j.analyze()
            

        print(f"iter {e} ppo reward={ppo_reward}")
        # print(tt.debug)
        tt_main.end_and_start("collect")

        pfc_train._learn()
        

        tt_main.end_and_start("learn")

        if e%20==0:
            pfc_train.save_model("all_models/pfc_model",e)



                
if __name__=="__main__":
    main()
                



