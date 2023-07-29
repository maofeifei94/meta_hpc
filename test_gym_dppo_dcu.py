from multiprocessing import Queue,Process
from pfc.hyperparam.dppo_gru_hybrid_dcu_hp import DPPO_Hybrid_HyperParam as hp
import gym
import copy
import time
import cv2


import paddle
import numpy as np
from mlutils.env import EnvNeverDone
from pfc.ac_net.ppo_gru_hybrid_actor import Actor,Actor_Static
from pfc.ac_net.ppo_gru_hybrid_critic import Critic,Critic_Static
from pfc.ac_model.ppo_gru_hybrid_model import Model
from pfc.alg.ppo_gru_hybrid_dcu_alg import PPO_GRU_HYBRID
from pfc.agent.ppo_gru_hybrid_dcu_agent import Agent
from pfc.data_buffer.ppo_gru_hybrid_rolloutstorage import RolloutStorage



def env_obs_to_paddle_vec(obs):
    obs=paddle.to_tensor(np.reshape(obs,[1,1,-1]).astype(np.float32))
    return obs
"env"
env = gym.make(hp.GAME_NAME)
game_type="discrete" if isinstance(env.action_space,gym.spaces.discrete.Discrete) else "continuous"
input_output_info={
    'to_static':True,
    'env_vec_dim':env.observation_space.shape[0],
    'obs_dim':env.observation_space.shape[0],
    'action_env_discrete_dim':env.action_space.n if game_type=="discrete" else 2,
    'action_env_continuous_dim':env.action_space.shape[0] if game_type=="continuous" else 2,
    'hyperparam':hp
    }
print("info",input_output_info)
env = EnvNeverDone(env,hp.GAME_OBS_SCALE)
env_obs=env.reset()
input_dict={
    "env_vec":env_obs_to_paddle_vec(env_obs)
}





"agent"
ppo_model=Model(
    input_output_info,
    net_class_dict={'actor_class':Actor_Static,'critic_class':Critic_Static},
    # net_class_dict={'actor_class':Actor,'critic_class':Critic}
    )
ppo_alg=PPO_GRU_HYBRID(ppo_model,input_output_info)
ppo_agent=Agent(ppo_alg,input_output_info)
ppo_agent.load_model('all_models/pfc_model',hp.GAME_LOAD_MODEL)


"init"
for e in range(int(1e16)):
    ppo_reward=0
    with paddle.no_grad():
        for _s in range(int(1e16)):
            "agent step"
            agent_output_dict=ppo_agent.test(input_dict,False,{'actor_init_h':None,'critic_init_h':None})
            agent_value=agent_output_dict['value']
            agent_action_discrete,agent_action_discrete_log_prob=agent_output_dict['action_discrete'],agent_output_dict['action_discrete_log_probs']
            agent_action_continuous,agent_action_continuous_log_prob=agent_output_dict['action_continuous'] ,agent_output_dict['action_continuous_log_probs']
            agent_actor_gru_h,agent_critic_gru_h=agent_output_dict['actor_gru_h'] ,agent_output_dict['critic_gru_h']

            # print(agent_action_discrete,agent_action_discrete_log_prob)
            
            "eng step"
            if game_type=="discrete":
                if _s==0:
                    agent_action_discrete=1
                next_env_obs,reward,done,info=env.step(agent_action_discrete)
                # print(next_env_obs.shape)
            elif game_type=="continuous":
                agent_action_continuous_for_env=(np.clip(agent_action_continuous,-1,1)+1)/2*(env.action_space.high-env.action_space.low)+env.action_space.low
                next_env_obs,reward,done,info=env.step(agent_action_continuous_for_env)
            if 1:
                img=env.render('rgb_array')
                # print(img)
                cv2.imshow("img",cv2.resize(img[:,:,::-1],None,fx=3,fy=3,interpolation=cv2.INTER_AREA))
                cv2.waitKey(20)
            if reward!=0:
                print("reward:",reward)
            reward=np.maximum(reward,-10)
            ppo_reward+=reward
            next_input_dict={"env_vec":env_obs_to_paddle_vec(next_env_obs)}
            
            data_dict={
                'env_vec':env_obs_to_paddle_vec(next_env_obs).numpy(),
                'action_discrete':1,
                'action_discrete_log_prob':1,
                'action_continuous':agent_action_continuous,
                'action_continuous_log_prob':agent_action_continuous_log_prob,
                'value_pred':agent_value,
                'reward':reward,
                'actor_gru_h':agent_actor_gru_h,
                'critic_gru_h':agent_critic_gru_h,
                'mask':[[1.0]],
                'bad_mask':[[1.0]],
            }
            input_dict=next_input_dict
    





