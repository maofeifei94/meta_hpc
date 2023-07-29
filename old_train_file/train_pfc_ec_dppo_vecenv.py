from multiprocessing import Pipe,Queue,Process
import threading
import gym
from pfc.hyperparam.dppo_gru_ec_dcu_hp import DPPO_Ec_HyperParam as hp


class trainer():
    def __init__(self) -> None:
        queue_dict={
            "ppo_rolloutstorage":Queue(),
            "ppo_state_dict":Queue(),
            "reward":Queue(),
            "require_ec_sd":Queue(),
            "ec_state_dict":Queue(),
            "ec_data":Queue(),
        }
        p1=Process(target=self.dppo_learn_process,args=(queue_dict,))
        interact_process_list=[Process(target=self.dppo_interact_process,args=(queue_dict,i)) for i in range(hp.WORKER_NUM)]
        p3=Process(target=self.train_ec_process,args=(queue_dict,))
        p_list=[p1,*interact_process_list,p3]
        [p.start() for p in p_list]
        [p.join() for p in p_list]

    def dppo_learn_process(self,queue_dict):
        from pfc.ac_net.ppo_gru_hybrid_actor import Actor,Actor_Static
        from pfc.ac_net.ppo_gru_hybrid_critic import Critic,Critic_Static
        from pfc.ac_model.ppo_gru_hybrid_model import Model
        from pfc.alg.ppo_gru_hybrid_dcu_alg import PPO_GRU_HYBRID
        from pfc.agent.ppo_gru_hybrid_dcu_agent import Agent
        from pfc.data_buffer.ppo_gru_hybrid_rolloutstorage import RolloutStorage
        from pfc.hyperparam.dppo_gru_ec_dcu_hp import DPPO_Ec_HyperParam as hp
        from pfc.hyperparam.dppo_gru_ec_dcu_hp import trainer_info,input_output_info,svaepred_info
        import logging
        import time

        "queue"
        queue_ppo_rolloutstorage=queue_dict['ppo_rolloutstorage']
        queue_ppo_state_dict=queue_dict['ppo_state_dict']
        queue_reward=queue_dict['reward']
        queue_require_ec_sd=queue_dict["require_ec_sd"]
        queue_ec_state_dict=queue_dict["ec_state_dict"]
        queue_ec_data=queue_dict["ec_data"]

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
                # print(f"{attr}={getattr(hp,attr)}")
                logging.info(f"{attr}={getattr(hp,attr)}")

        'main loop'
        for e in range(int(1e16)):
            t1=time.time()
            "update"
            for _p in range(hp.WORKER_NUM):
                queue_ppo_state_dict.put(ppo_agent.send_model_to_np())
            print("ppo send_state_dict success")

            "get_data"
            data_buffer_list=[]
            for _p in range(hp.WORKER_NUM):
                data_buffer_list.append(queue_ppo_rolloutstorage.get())
                # print(f"get data_buffer:",time.time())
            print("ppo get_all_data_buffer")
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
                ppo_agent.save_model("all_models/pfc_model","newest")
            

    def dppo_interact_process(self,queue_dict,process_n):
        import paddle
        import numpy as np
        from mlutils.env import EnvNeverDone
        from ec.seq_vae_pred import SeqVaePred
        from pfc.ac_net.ppo_gru_hybrid_actor import Actor,Actor_Static
        from pfc.ac_net.ppo_gru_hybrid_critic import Critic,Critic_Static
        from pfc.ac_model.ppo_gru_hybrid_model import Model
        from pfc.alg.ppo_gru_hybrid_dcu_alg import PPO_GRU_HYBRID
        from pfc.agent.ppo_gru_hybrid_dcu_agent import Agent
        from pfc.data_buffer.ppo_gru_hybrid_rolloutstorage import RolloutStorage
        from pfc.hyperparam.dppo_gru_ec_dcu_hp import DPPO_Ec_HyperParam as hp
        from pfc.hyperparam.dppo_gru_ec_dcu_hp import trainer_info,input_output_info,svaepred_info,env_param

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
        env = EnvNeverDone(env)
        env_obs=env.reset()


        "queue"
        queue_ppo_rolloutstorage=queue_dict['ppo_rolloutstorage']
        queue_ppo_state_dict=queue_dict['ppo_state_dict']
        queue_reward=queue_dict['reward']
        queue_require_ec_sd=queue_dict["require_ec_sd"]
        queue_ec_state_dict=queue_dict["ec_state_dict"]
        queue_ec_data=queue_dict["ec_data"]

        "data"
        data_buffer=RolloutStorage(input_output_info)

        "agent"
        input_output_info['to_static']=True
        ppo_model=Model(
            input_output_info,
            net_class_dict={'actor_class':Actor_Static,'critic_class':Critic_Static},
            )
        ppo_alg=PPO_GRU_HYBRID(ppo_model,input_output_info)
        ppo_agent=Agent(ppo_alg,input_output_info)
        def update_ppo_agent():
            ppo_agent.update_model_from_np(queue_ppo_state_dict.get())
            print("ppo update model")
        update_ppo_agent()

        "ec"
        svaepred=SeqVaePred(svaepred_info)
        def update_ec():
            queue_require_ec_sd.put(True)
            print("ec put sd require")
            ec_state_dict=queue_ec_state_dict.get()
            svaepred.update_model_from_np(ec_state_dict)
            print("ec update model")
        update_ec()
        svaepred.eval()
        ec_obs=svaepred.reset()


        input_dict={
            "env_vec":paddle.concat([env_obs_to_paddle_vec(env_obs),env_obs_to_paddle_vec(ec_obs)],axis=-1)
        }



        "train"
        for e in range(int(1e16)):
            ppo_reward=0
            ec_data_list=[]
            with paddle.no_grad():
                for i in range(hp.PPO_NUM_STEPS):
                    # print(f"ppo collect {e} {i}")
                    "agent step"
                    agent_output_dict=ppo_agent.interact(input_dict,False,{'actor_init_h':None,'critic_init_h':None})
                    agent_value=agent_output_dict['value']
                    # agent_action_discrete,agent_action_discrete_log_prob=agent_output_dict['action_discrete'],agent_output_dict['action_discrete_log_probs']
                    agent_action_continuous,agent_action_continuous_log_prob=agent_output_dict['action_continuous'] ,agent_output_dict['action_continuous_log_probs']
                    agent_action_continuous_for_env=(np.clip(agent_action_continuous,-1,1)+1)/2*(env.action_space.high-env.action_space.low)+env.action_space.low
                    agent_actor_gru_h,agent_critic_gru_h=agent_output_dict['actor_gru_h'] ,agent_output_dict['critic_gru_h']

                    
                    
                    "env step"
                    next_env_obs,reward,done,info=env.step(agent_action_continuous_for_env)
                    reward=np.maximum(reward,-10)
                    ppo_reward+=reward
                    "ec step"
                    env_obs_with_reward=np.concatenate([env_obs,[reward]])
                    next_ec_obs=svaepred.forward(env_obs_with_reward if env_param.add_reward_to_ec else env_obs,agent_action_continuous_for_env)
                    
                    "next step"
                    env_obs=next_env_obs
                    ec_obs=next_ec_obs
                    next_input_dict={"env_vec":paddle.concat([env_obs_to_paddle_vec(next_env_obs),env_obs_to_paddle_vec(next_ec_obs)],axis=-1)}
                    input_dict=next_input_dict

                    "for queue"
                    data_dict={
                        'env_vec':next_input_dict['env_vec'].numpy(),
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
                    data_buffer.append(data_dict)

                    ec_data_list.append({
                        "state":np.reshape(env_obs_with_reward if env_param.add_reward_to_ec else env_obs,[-1]),
                        'action':np.reshape(agent_action_continuous_for_env,[-1]),
                        'glo_gru_h':svaepred.global_h.numpy().reshape([-1]),
                        'loc_gru_h':svaepred.local_h.numpy().reshape([-1]),
                    })



            "queue put"
            queue_ppo_rolloutstorage.put(data_buffer)
            queue_reward.put(ppo_reward/hp.PPO_NUM_STEPS)
            queue_ec_data.put(ec_data_list)


            "update model"
            update_ec()
            update_ppo_agent()

            "reset data buffer"
            data_buffer.after_update()
            
    def train_ec_process(self,queue_dict):

        from ec.seq_vae_pred import SeqVaePred
        from mlutils.ml import ReplayMemory,DataFormat
        from pfc.hyperparam.dppo_gru_ec_dcu_hp import DPPO_Ec_HyperParam as hp
        from pfc.hyperparam.dppo_gru_ec_dcu_hp import trainer_info,input_output_info,svaepred_info
        import numpy as np
        import time
        "queue"
        queue_ppo_rolloutstorage=queue_dict['ppo_rolloutstorage']
        queue_ppo_state_dict=queue_dict['ppo_state_dict']
        queue_reward=queue_dict['reward']
        queue_require_ec_sd=queue_dict["require_ec_sd"]
        queue_ec_state_dict=queue_dict["ec_state_dict"]
        queue_ec_data=queue_dict["ec_data"]

        "rpm"
        rpm=ReplayMemory(
            [
                DataFormat('state',[svaepred_info.state_vec_dim],'float32'),
                DataFormat('action',[svaepred_info.action_vec_dim],'float32'),
                DataFormat('glo_gru_h',[svaepred_info.gloinfo_gru_dim],'float32'),
                DataFormat('loc_gru_h',[svaepred_info.locinfo_gru_dim],'float32'),
            ],
            max_size=trainer_info.ec_rpm_size
        )
        def collect_data():
            data_list=queue_ec_data.get()
            for data in data_list:
                rpm.collect(data)
        
        "svaepred"
        svaepred=SeqVaePred(svaepred_info)
        if trainer_info.ec_svaepred_load_model is not None:
            svaepred.load_model(trainer_info.ec_svaepred_model_dir,trainer_info.ec_svaepred_load_model)



        "train"
        for train_iter in range(int(1e16)):
            "send_sd"
            # print("ec",train_iter)
            if queue_require_ec_sd.qsize()>0:
                queue_require_ec_sd.get()
                print("ec get sd require")
                queue_ec_state_dict.put(svaepred.send_model_to_np())
                print("ec send model")
            
            "collect data"
            if queue_ec_data.qsize()>0:
                "数据格式是list[dict]"
                collect_data()
                print("ec rpm size=",rpm._size)
            
            if trainer_info.ec_svaepred_train and rpm._size>=trainer_info.ec_train_min_rpm_size:
                "train"
                avg_loss=0
                for _ in range(trainer_info.ec_train_freq):
                    sample_len=np.random.randint(trainer_info.ec_svaepred_train_history_len//2,trainer_info.ec_svaepred_train_history_len)
                    train_data=rpm.sample_batch_seq(trainer_info.ec_svaepred_train_batch_size,sample_len+1)
                    # print(np.shape(train_data['state']),np.shape(train_data['glo_gru_h']))
                    loss=svaepred.train(
                        train_data['state'][:,1:],
                        train_data['action'][:,1:],
                        np.transpose(train_data['glo_gru_h'][:,0:1],[1,0,2]),
                        np.transpose(train_data['loc_gru_h'][:,0:1],[1,0,2]),
                        )
                    avg_loss+=np.array(loss)
                avg_loss/=trainer_info.ec_train_freq
                print("svaepred loss=",avg_loss)

                "save model"
                if train_iter%500==0:
                    svaepred.save_model(trainer_info.ec_svaepred_model_dir,train_iter)
                    svaepred.save_model(trainer_info.ec_svaepred_model_dir,"newest")
            else:
                time.sleep(0.2)



                

        
if __name__=="__main__":
    trainer()
        




