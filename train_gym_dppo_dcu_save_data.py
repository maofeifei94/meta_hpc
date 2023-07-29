from multiprocessing import Queue,Process
from pfc.hyperparam.dppo_gru_hybrid_dcu_hp import DPPO_Hybrid_HyperParam as hp
import gym
import copy
import time


env = gym.make(hp.GAME_NAME)
game_type="discrete" if isinstance(env.action_space,gym.spaces.discrete.Discrete) else "continuous"
input_output_info={
    'to_static':False,
    'env_vec_dim':env.observation_space.shape[0],
    'obs_dim':env.observation_space.shape[0],
    'action_env_discrete_dim':env.action_space.n if game_type=="discrete" else 2,
    'action_env_continuous_dim':env.action_space.shape[0] if game_type=="continuous" else 2,
    'hyperparam':hp
    }


def dppo_learn_process(queue_dict):
    from pfc.ac_net.ppo_gru_hybrid_actor import Actor,Actor_Static
    from pfc.ac_net.ppo_gru_hybrid_critic import Critic,Critic_Static
    from pfc.ac_model.ppo_gru_hybrid_model import Model
    from pfc.alg.ppo_gru_hybrid_dcu_alg import PPO_GRU_HYBRID
    from pfc.agent.ppo_gru_hybrid_dcu_agent import Agent
    from pfc.data_buffer.ppo_gru_hybrid_rolloutstorage import RolloutStorage
    import logging

    "queue"
    queue_data=queue_dict['data']
    queue_state_dict=queue_dict['state_dict']
    queue_reward=queue_dict['reward']

 
    "agent"
    ppo_model=Model(
        input_output_info,
        net_class_dict={'actor_class':Actor,'critic_class':Critic}
        )
    ppo_alg=PPO_GRU_HYBRID(ppo_model,input_output_info)
    ppo_agent=Agent(ppo_alg,input_output_info)
    if hp.GAME_LOAD_MODEL is not None:
        ppo_agent.load_model('all_models/pfc_model',hp.GAME_LOAD_MODEL)

    "log"
    logging.basicConfig(filename="log/"+time.strftime("%Y_%m%d_%H%M%S",time.localtime(time.time()))+".txt", level=logging.INFO)
    "超参数写入log"
    for attr in dir(hp):
        if attr.startswith("__") and attr.endswith("__"):
            # print(None)
            pass
        else:
            print(f"{attr}={getattr(hp,attr)}")
            logging.info(f"{attr}={getattr(hp,attr)}")

    'main loop'
    for e in range(int(1e16)):
        t1=time.time()
        "update"
        for _p in range(hp.WORKER_NUM):
            queue_state_dict.put(ppo_agent.send_model_to_np())
        # print("send_state_dict success")

        "get_data"
        data_buffer_list=[]
        for _p in range(hp.WORKER_NUM):
            data_buffer_list.append(queue_data.get())
            # print(f"get data_buffer:",time.time())
        # print("get_all_data_buffer")
        "get_reward"
        ppo_reward=0
        for _p in range(hp.WORKER_NUM):
            ppo_reward+=queue_reward.get()
        ppo_reward/=hp.WORKER_NUM

        t2=time.time()
        

        "learn"
        learn_loss=ppo_agent.learn(data_buffer_list)
        value_loss_epoch,action_continuous_loss_epoch,action_continuous_dist_entropy_epoch,action_discrete_loss_epoch, action_discrete_dist_entropy_epoch,value_clip_mean_ratio_epoch=learn_loss
        t3=time.time()

        print(f"data {round(t2-t1,2)} | learn {round(t3-t2,2)}")
        "log and save model"
        print(f"iter {e} ppo reward={ppo_reward}")
        logging.info(
                f"|iter={e}"+
                f"|reward={ppo_reward}"+
                # f"|abs_action_mean={np.mean(np.abs(pfc_train.rpm.actions))}"+
                f"|v_loss={value_loss_epoch[0]}"+
                f"|action_discrete_loss={action_discrete_loss_epoch[0]}"+
                f"|action_discrete_dist_entropy={action_discrete_dist_entropy_epoch[0]}"+
                f"|action_continuous_loss={action_continuous_loss_epoch[0]}"+
                f"|action_continuous_dist_entropy={action_continuous_dist_entropy_epoch[0]}"+
                f"|value_clip_mean_ratio_epoch={value_clip_mean_ratio_epoch[0]}"
                )

        if e%100==0:
            ppo_agent.save_model("all_models/pfc_model",e)
        

def dppo_interact_process(queue_dict,process_n):
    import paddle
    import numpy as np
    from mlutils.env import EnvNeverDone
    from mlutils.ml import ReplayMemory,DataFormat
    from pfc.ac_net.ppo_gru_hybrid_actor import Actor,Actor_Static
    from pfc.ac_net.ppo_gru_hybrid_critic import Critic,Critic_Static
    from pfc.ac_model.ppo_gru_hybrid_model import Model
    from pfc.alg.ppo_gru_hybrid_dcu_alg import PPO_GRU_HYBRID
    from pfc.agent.ppo_gru_hybrid_dcu_agent import Agent
    from pfc.data_buffer.ppo_gru_hybrid_rolloutstorage import RolloutStorage

    if hp.WORKER_USE_CPU and process_n%hp.WORKER_USE_CPU_INTERVAL==0:
        paddle.device.set_device("cpu")
        print(f"process {process_n} use CPU")
    else:
        print(f"process {process_n} use GPU")
    # print("device",paddle.device.get_device())

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
    env = EnvNeverDone(env,hp.GAME_OBS_SCALE)
    env_obs=env.reset()
    input_dict={
        "env_vec":env_obs_to_paddle_vec(env_obs)
    }

    "queue"
    queue_data=queue_dict['data']
    queue_state_dict=queue_dict['state_dict']
    queue_reward=queue_dict['reward']

    "data"
    data_buffer=RolloutStorage(input_output_info)

    "agent"
    ppo_model=Model(
        input_output_info,
        net_class_dict={'actor_class':Actor_Static,'critic_class':Critic_Static},
        # net_class_dict={'actor_class':Actor,'critic_class':Critic}
        )
    ppo_alg=PPO_GRU_HYBRID(ppo_model,input_output_info)
    ppo_agent=Agent(ppo_alg,input_output_info)
    ppo_agent.update_model_from_np(queue_state_dict.get())

    "data saver"
    rpm=ReplayMemory([
        DataFormat("state",[input_output_info['obs_dim']],'float32'),
        DataFormat("reward",[1],'float32'),
        DataFormat('action',[input_output_info['action_env_continuous_dim']] if game_type=='continuous' else [1],'float32')
        ],max_size=int(hp.GAME_DATA_SAVE_SIZE))


    "init"
    for e in range(int(1e16)):
        ppo_reward=0
        with paddle.no_grad():
            for i in range(hp.PPO_NUM_STEPS):
                "agent step"
                agent_output_dict=ppo_agent.interact(input_dict,False,{'actor_init_h':None,'critic_init_h':None})
                agent_value=agent_output_dict['value']
                agent_action_discrete,agent_action_discrete_log_prob=agent_output_dict['action_discrete'],agent_output_dict['action_discrete_log_probs']
                agent_action_continuous,agent_action_continuous_log_prob=agent_output_dict['action_continuous'] ,agent_output_dict['action_continuous_log_probs']
                agent_actor_gru_h,agent_critic_gru_h=agent_output_dict['actor_gru_h'] ,agent_output_dict['critic_gru_h']
                
                "eng step"
                if game_type=="discrete":
                    next_env_obs,reward,done,info=env.step(agent_action_discrete)
                elif game_type=="continuous":
                    agent_action_continuous_for_env=(np.clip(agent_action_continuous,-1,1)+1)/2*(env.action_space.high-env.action_space.low)+env.action_space.low
                    next_env_obs,reward,done,info=env.step(agent_action_continuous_for_env)
                reward=np.maximum(reward,-10)
                ppo_reward+=reward
                next_input_dict={"env_vec":env_obs_to_paddle_vec(next_env_obs)}
                
                data_dict={
                    'env_vec':env_obs_to_paddle_vec(next_env_obs).numpy(),
                    'action_discrete':agent_action_discrete,
                    'action_discrete_log_prob':agent_action_discrete_log_prob,
                    'action_continuous':agent_action_continuous,
                    'action_continuous_log_prob':agent_action_continuous_log_prob,
                    'value_pred':agent_value,
                    'reward':reward,
                    'actor_gru_h':agent_actor_gru_h,
                    'critic_gru_h':agent_critic_gru_h,
                    'mask':[[1.0]],
                    'bad_mask':[[1.0]],
                }

                rpm.collect({
                    "state":env_obs,
                    "action":agent_action_continuous_for_env if game_type=='continuous' else [agent_action_discrete],
                    'reward':[reward],
                })
                if rpm._size>=rpm._max_size:
                    rpm.save(f"{hp.GAME_DATA_SAVE_DIR}/process_{process_n}",rpm.save_iter)
                    rpm.clear()


                data_buffer.append(data_dict)
                input_dict=next_input_dict
                env_obs=next_env_obs
                # print(f"process {process_n},i={i}")
        # print(f"process {process_n} send data_buffer:",time.time())
        queue_data.put(data_buffer)
        queue_reward.put(ppo_reward/hp.PPO_NUM_STEPS)

        ppo_agent.update_model_from_np(queue_state_dict.get())
        # print(f"process {process_n} update model success")
        data_buffer.after_update()




def main_process():

    queue_dict={
        'data':Queue(maxsize=1000),
        'state_dict':Queue(maxsize=1000),
        'reward':Queue(maxsize=1000)
    }
    p_list=[]

    p_list.append(Process(target=dppo_learn_process,args=(queue_dict,)))
    for i in range(hp.WORKER_NUM):
        p_list.append(Process(target=dppo_interact_process,args=(queue_dict,i)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    pass

if __name__=="__main__":
    main_process()
    





