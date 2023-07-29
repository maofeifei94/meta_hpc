import numpy as np
import time
from mlutils.ml import ModelSaver,one_hot
from mlutils.multiprocess import ProcessFilter,TaskFlow,Task,Cache
from mlutils.ml import DataFormat,ReplayMemory
from multiprocessing import Queue,Process,Manager
from multiprocessing import Lock as process_Lock
import threading
import cv2
import os
import paddle
from paddle import nn
from paddle.jit import to_static
from paddle.static import InputSpec
#env
# from env.env_animalai import Animal
# from env.env_procgen_jumper import Jumper
from env.env_guess_box import GuessBox
#brain
from brain.hyperparam.brain_guess_box_hp import brain_hyperparam
#pfc
from pfc.ac_net.ppo_gru_hybrid_actor import Actor,Actor_Static
from pfc.ac_net.ppo_gru_hybrid_critic import Critic,Critic_Static
from pfc.ac_model.ppo_gru_hybrid_model import Model
from pfc.agent.ppo_gru_hybrid_dcu_agent import Agent
from pfc.alg.ppo_gru_hybrid_dcu_alg import PPO_GRU_HYBRID
from pfc.data_buffer.ppo_gru_hybrid_rolloutstorage import RolloutStorage
from mlutils.logger import mllogger
from mlutils.data import LMDBRecoder

"""
定义各个组件的来源
"""
def import_ec():
    from ec.ec_vae import SeqVaePred
    return SeqVaePred
def import_cae():
    from cae.cae_focus_8x8 import Conv_Auto_Encoder
    return Conv_Auto_Encoder
"""
环境交互的数据到flowgenerator中，flowgenerator创建datafilter收集数据的flow
"""


class FlowGenerator():
    """
    任务产生器
    起到调节任务比例的作用。
    比如某一个模块训练的比较好了，就根据它的Loss值调节训练间隔，将计算资源分配给训练得不好的模块
    """
    def __init__(self,queue_dict,name,share_info,lock) -> None:
        self.queue_dict=queue_dict
        self.name=name
        self.share_info=share_info
        self.lock=lock
        # self.env_data_queue=self.queue_dict[self.name]
        # self._run_system()
        # self._run_train_cae()
        # self._run_collect()
        self._run_train_ec()
        # self._run_train_ppo()
    def _send(self,flow:TaskFlow):
        """
        根据flow的信息,将数据发送到下一个进程专属的Queue中
        """
        if flow.left_steps>0:
            self.queue_dict[flow.current_task.filter_name][flow.priority].put(flow)
    def _run_system(self):
        """
        interact and train
        """
        steps=0
        time.sleep(5)
        interact_flow=self.generate_env_interact_flow()
        self._send(interact_flow)
        self._send(self.generate_h_update_flow({
            "pre_glo_h":np.zeros([1,1,brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32),
            "pre_loc_h":np.zeros([1,1,brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32),
            "pre_klpred_h":np.zeros([1,1,brain_hyperparam.ec_info.klpred_gru_dim],dtype=np.float32),
            'start_index':0,
            'seq_len':2048,
            'default_seq_len':2048,
            'finish_first_loop':False
        }))


        def thread_sys_loop():
            finish_first_loop=False
            while 1:
                #收集env数据
                time.sleep(0.01)
                if self.queue_dict['flowgenerator'].qsize()>0:
                    # print("send new interact flow")
                    for _ in range(self.queue_dict['flowgenerator'].qsize()):
                        info=self.queue_dict['flowgenerator'].get()
                        if info['name']=="h_update":
                            if info['finish_first_loop']:
                                if not finish_first_loop:
                                    finish_first_loop=True
                                    self._send(self.generate_clear_buffer_flow())
                            if not finish_first_loop:
                                print("h_update_start_index=",info['start_index'])
                            h_update_flow=self.generate_h_update_flow(info)
                            self._send(h_update_flow)
                        elif info['name']=="interact":
                            interact_flow=self.generate_env_interact_flow()
                            self._send(interact_flow)
                        else:
                            raise KeyError(f"info name={info['name']} not match [h_update,interact]")
        def thread_train():
            while 1:
                train_flow=self.generate_online_train_flow()
                self._send(train_flow)
                time.sleep(0.01)
        t1=threading.Thread(target=thread_sys_loop)
        t2=threading.Thread(target=thread_train)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    def _run_collect(self):
        for i in range(200):
            self._send(self.generate_collect_folw())
            print(f"send collect flow {i}")
            
    def _run_train_cae(self):
        self._send(self.generate_clear_buffer_flow())
        while 1:
            self._send(self.generate_cae_train_flow())
            time.sleep(0.01)
    def _run_train_ec(self):
        """
        interact and train
        """
        steps=0
        time.sleep(5)
        self._send(self.generate_h_update_flow({
            "pre_glo_h":np.zeros([1,1,brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32),
            "pre_loc_h":np.zeros([1,1,brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32),
            "pre_klpred_h":np.zeros([1,1,brain_hyperparam.ec_info.klpred_gru_dim],dtype=np.float32),
            'start_index':0,
            'seq_len':2048,
            'default_seq_len':2048,
            'finish_first_loop':False
        }))
        self._send(self.generate_clear_buffer_flow())
        self._send(self.generate_clear_buffer_flow())
        def thread_sys_loop():
            finish_first_loop=False
            while 1:
                #收集env数据
                time.sleep(0.01)
                if self.queue_dict['flowgenerator'].qsize()>0:
                    # print("send new interact flow")
                    for _ in range(self.queue_dict['flowgenerator'].qsize()):
                        info=self.queue_dict['flowgenerator'].get()
                        if info['name']=="h_update":
                            if info['finish_first_loop']:
                                if not finish_first_loop:
                                    finish_first_loop=True
                                    self._send(self.generate_clear_buffer_flow())
                            if not finish_first_loop:
                                print("h_update_start_index=",info['start_index'])
                            h_update_flow=self.generate_h_update_flow(info)
                            self._send(h_update_flow)
                        else:
                            raise KeyError(f"info name={info['name']} not match [h_update,interact]")
        def thread_train():
            while 1:
                train_flow=self.generate_ec_train_flow()
                self._send(train_flow)
                time.sleep(0.01)
        t1=threading.Thread(target=thread_sys_loop)
        t2=threading.Thread(target=thread_train)
        t1.start()
        t2.start()
        t1.join()
        t2.join()





    def _run_train_brain(self):
        """
        only train
        """
        steps=0
        time.sleep(5)
        self._send(self.generate_h_update_flow({
            "pre_glo_h":np.zeros([1,1,brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32),
            "pre_loc_h":np.zeros([1,1,brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32),
            "pre_klpred_h":np.zeros([1,1,brain_hyperparam.ec_info.klpred_gru_dim],dtype=np.float32),
            'start_index':0,
            'seq_len':2048,
            'default_seq_len':2048,
            'finish_first_loop':False
        }))

        finish_first_loop=False
        while 1:
            if self.queue_dict['flowgenerator'].qsize()>0:
                info=self.queue_dict['flowgenerator'].get()
                if info['name']=="h_update":
                    if info['finish_first_loop']:
                        if not finish_first_loop:
                            finish_first_loop=True
                            self._send(self.generate_clear_buffer_flow())
                    if not finish_first_loop:
                        print(info['start_index'])
                    h_update_flow=self.generate_h_update_flow(info)
                    self._send(h_update_flow)

                    # print("next info",info)
            # train_flow=self.generate_train_flow()
            # train_flow=self.generate_cae_train_flow()
            if finish_first_loop:
                train_flow=self.generate_offline_train_flow()
                self._send(train_flow)
                time.sleep(0.01)

    def _run_train_ppo(self):
        """
        only train ppo
        """
        steps=0
        time.sleep(5)
        interact_flow=self.generate_ppo_train_flow()
        self._send(interact_flow)
        while 1:
            #收集env数据
            if self.queue_dict['flowgenerator'].qsize()>0:
                # print("send new interact flow")
                self.queue_dict['flowgenerator'].get()
                interact_flow=self.generate_ppo_train_flow()
                self._send(interact_flow)
            time.sleep(1)
    def generate_collect_folw(self):
        tasklist=[
            #获取数据
            Task(filter_name='envfilter',func_name='randact_collect'),
            #训练模型
            Task(filter_name='datafilter',func_name='collect',data_key="env_randactcollect"),
        ]
        
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='high')
    def generate_env_interact_flow(self):
        tasklist=[
            #更新参数
            Task(filter_name='ppofilter',func_name='get_agent_param'),
            Task(filter_name='ecfilter',func_name='get_ec_param'),
            Task(filter_name='envfilter',func_name='update_all_param'),
            #获取数据
            Task(filter_name='envfilter',func_name='interact'),
            #训练模型
            Task(filter_name='datafilter',func_name='collect',data_key="env_interact_fordata"),
            Task(filter_name='ppofilter',func_name='train'),
            Task(filter_name='ppofilter',func_name='save_model',save_interval=20,save_dir=brain_hyperparam.DPPO_Hybrid_HyperParam.MODEL_SAVE_DIR),
            Task(filter_name='finishfilter',func_name='finish_interact')
        ]
        
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='high')
    def generate_ppo_train_flow(self):
        tasklist=[
            #更新参数
            Task(filter_name='ppofilter',func_name='get_agent_param'),
            Task(filter_name='ecfilter',func_name='get_ec_param'),
            Task(filter_name='envfilter',func_name='update_all_param'),
            #获取数据
            Task(filter_name='envfilter',func_name='interact'),
            #训练模型
            Task(filter_name='ppofilter',func_name='train'),
            Task(filter_name='ppofilter',func_name='save_model',save_interval=20,save_dir=brain_hyperparam.DPPO_Hybrid_HyperParam.MODEL_SAVE_DIR),
            Task(filter_name='finishfilter',func_name='finish_interact')
        ]
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='high')
    def generate_online_train_flow(self):
        tasklist=[
            Task(filter_name='datafilter',func_name='sample_data',batch_size=brain_hyperparam.train_batch_size,seq_len=brain_hyperparam.train_seq_len,use_recent_rpm=True),
            Task(filter_name='ecfilter',func_name='train'),
            Task(filter_name='ecfilter',func_name='save_model',save_interval=1000,save_dir=brain_hyperparam.ec_info.model_dir)
        ]
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='low')
    def generate_offline_train_flow(self):
        tasklist=[
            Task(filter_name='datafilter',func_name='sample_data',batch_size=brain_hyperparam.train_batch_size,seq_len=brain_hyperparam.train_seq_len,use_recent_rpm=False),
    
            Task(filter_name='ecfilter',func_name='train'),
            Task(filter_name='ecfilter',func_name='save_model',save_interval=1000,save_dir=brain_hyperparam.ec_info.model_dir)
        ]
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='low')
    def generate_ec_train_flow(self):
        tasklist=[
            Task(filter_name='datafilter',func_name='sample_data',batch_size=brain_hyperparam.train_batch_size,seq_len=brain_hyperparam.train_seq_len,use_recent_rpm=False),
            Task(filter_name='ecfilter',func_name='train'),
            Task(filter_name='ecfilter',func_name='save_model',save_interval=1000,save_dir=brain_hyperparam.ec_info.model_dir)
        ]
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='low')
    def generate_h_update_flow(self,info_dict):
        # info_dict={
        #         "pre_glo_h":pre_glo_h,
        #         "pre_loc_h":pre_loc_h,
        #         "start_index":start_index,
        #         "seq_len":seq_len
        #     }
        tasklist=[
            Task(filter_name="datafilter",func_name='sample_h_update_seq'),
            Task(filter_name='ecfilter',func_name='pred_h',cache_data_key='data_samplehupdateseq'),
            Task(filter_name='datafilter',func_name='h_update'),
            Task(filter_name='finishfilter',func_name='finish_h_update')
        ]
        cache=Cache(self.share_info,self.lock)
        cache.update({
            "info":info_dict
            })
        return TaskFlow(tasklist=tasklist,cache=cache,priority='mid')
    def generate_clear_buffer_flow(self):
        tasklist=[
            Task(filter_name="datafilter",func_name='clear_buffer'),
        ]
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='high')
    
class FinishFilter(ProcessFilter):
    def finish_interact(self,cache,**kwargs):
        self.queue_dict['flowgenerator'].put({'name':'interact'})
        cache_update={}
        return cache_update
    def finish_h_update(self,cache,**kwargs):
        if cache['next_info'] is None:
            next_info=cache['info']
            next_info.update({'name':'h_update'})

        else:
            next_start_index=cache['next_info']['start_index']
            next_pre_glo_h=cache['ec_predh']['next_glo_h']
            next_pre_loc_h=cache['ec_predh']['next_loc_h']
            next_pre_klpred_h=cache['ec_predh']['next_klpred_h']
            if next_start_index<cache['info']['start_index']:
                next_pre_glo_h*=0
                next_pre_loc_h*=0
                next_pre_klpred_h*=0
                print("h_update finish one loop")
                finish_first_loop=True
            else:
                finish_first_loop=False
            next_seq_len=cache['next_info']['seq_len']
            default_seq_len=cache['info']['default_seq_len']

            next_info={
                'name':'h_update',
                'pre_glo_h':next_pre_glo_h,
                'pre_loc_h':next_pre_loc_h,
                'pre_klpred_h':next_pre_klpred_h,
                'start_index':next_start_index,
                'seq_len':next_seq_len,
                'default_seq_len':default_seq_len,
                'finish_first_loop':finish_first_loop
            }
        self.queue_dict['flowgenerator'].put(next_info)
        cache_update={}
        return cache_update
class EnvFilter(ProcessFilter):
    def __init__(self, queue_dict,name='envfilter') -> None:
        paddle.device.set_device("cpu")
        #env
        # self.env=Animal(play=False,config_file="env/animalai_env/configs/competition/01-28-01.yaml")
        self.env=GuessBox()
        #cae
        SeqVaePred=import_ec()
        self.ec=SeqVaePred(brain_hyperparam.ec_info)
        
        # ppo_param
        self.ppo_input_output_info=brain_hyperparam.PPO_input_output_info
        self.ppo_hp=self.ppo_input_output_info['hyperparam']
        self.ppo_env_vec_dim=self.ppo_input_output_info['env_vec_dim']
        self.ppo_num_steps= self.ppo_hp.PPO_NUM_STEPS
        #agent
        self.agent=Agent(
            PPO_GRU_HYBRID(
                Model(self.ppo_input_output_info,{'actor_class':Actor,'critic_class':Critic}),self.ppo_input_output_info
            ),
            self.ppo_input_output_info
        )
        #reset
        self._reset()

        super().__init__(queue_dict,name)


    def _reset(self):
        env_obs=self.env.reset()
        self.env_img=env_obs
        self.input_dict,self.pre_h_dict,inner_reward=self._env_obs_to_input_dict_and_inner_reward(env_obs,one_hot([0],brain_hyperparam.env_action_discrete_num))
    
    def _env_obs_to_input_dict_and_inner_reward(self,env_obs,action_env_discrete_onehot):
        env_obs_paddle=paddle.to_tensor(env_obs,dtype='float32').reshape([1,1,-1])
        action_paddle=paddle.to_tensor(action_env_discrete_onehot,'float32').reshape([1,1,-1])

        # cae_vec,ec_vec,pre_glo_h,pre_loc_h,inner_reward_ori=self.cae_ec(env_obs_paddle,action_env_discrete_onehot)
        ec_vec,pre_glo_h,pre_loc_h,pre_klpred_h,inner_reward_ori=self.ec.pred_env(env_obs_paddle,action_paddle)
        # input_dict={
        #     'env_vec':paddle.concat([env_obs_paddle,ec_vec*0],axis=-1),
        #             }
        # pre_h_dict={
        #     'pre_glo_h':pre_glo_h.numpy().reshape([1,-1])*0,
        #     'pre_loc_h':pre_loc_h.numpy().reshape([1,-1])*0,
        # }
        input_dict={
            'env_vec':paddle.concat([env_obs_paddle,ec_vec],axis=-1),
                    }
        pre_h_dict={
            'pre_glo_h':pre_glo_h.numpy().reshape([1,-1]),
            'pre_loc_h':pre_loc_h.numpy().reshape([1,-1]),
            'pre_klpred_h':pre_klpred_h.numpy().reshape([1,-1]),
        }
        return input_dict,pre_h_dict,inner_reward_ori*brain_hyperparam.inner_reward_ratio
    def randact_collect(self,acache,**kwargs):
        rpm=ReplayMemory(
            data_format_list=[
                DataFormat(name='obs',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32,key_head=0),
                DataFormat(name='action_env',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32,key_head=1),
                DataFormat(name='action_ec',shape=[brain_hyperparam.ec_action_continuous_dim],dtype=np.float32,key_head=2),
                DataFormat(name='ec_glo_gru',shape=[brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32,key_head=3),
                DataFormat(name='ec_loc_gru',shape=[brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32,key_head=4),
                DataFormat(name='ec_klpred_gru',shape=[brain_hyperparam.ec_info.klpred_gru_dim],dtype=np.float32,key_head=5),
                DataFormat(name='reward',shape=[1],dtype=np.float32,key_head=6),
            ],max_size=self.ppo_hp.PPO_NUM_STEPS
        )
        t1=time.time()
        with paddle.no_grad():
            for _iter in range(self.ppo_hp.PPO_NUM_STEPS):
                # print("iter=",_iter)
                #agent step
                rand_action=np.random.randint(0,brain_hyperparam.env_action_discrete_num)
                agent_action_discrete_onehot=one_hot([rand_action],brain_hyperparam.env_action_discrete_num)

                #env_step
                next_env_obs,env_reward,done,info=self.env.step(rand_action)

                rpm.collect({
                    'obs':np.reshape(next_env_obs,[-1]),#下一步图像
                    'action_env':np.reshape(agent_action_discrete_onehot,[-1]),#当前动作
                    'action_ec':np.zeros([brain_hyperparam.ec_action_continuous_dim],np.float32),
                    'ec_glo_gru':np.zeros([brain_hyperparam.ec_info.gloinfo_gru_dim],np.float32),
                    'ec_loc_gru':np.zeros([brain_hyperparam.ec_info.locinfo_gru_dim],np.float32),
                    'ec_klpred_gru':np.zeros([brain_hyperparam.ec_info.klpred_gru_dim],np.float32),
                    'reward':[env_reward]
                }) 

        print(f"collect cost {time.time()-t1}")
            
        cache_update={
            'env_randactcollect':rpm.get_data_dict(),
            }
        return cache_update

    def interact(self,cache,**kwargs):
        ppo_data_buffer=RolloutStorage(self.ppo_input_output_info)
        rpm=ReplayMemory(
            data_format_list=[
                DataFormat(name='obs',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32,key_head=0),
                DataFormat(name='action_env',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32,key_head=1),
                DataFormat(name='action_ec',shape=[brain_hyperparam.ec_action_continuous_dim],dtype=np.float32,key_head=2),
                DataFormat(name='ec_glo_gru',shape=[brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32,key_head=3),
                DataFormat(name='ec_loc_gru',shape=[brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32,key_head=4),
                DataFormat(name='ec_klpred_gru',shape=[brain_hyperparam.ec_info.klpred_gru_dim],dtype=np.float32,key_head=5),
                DataFormat(name='reward',shape=[1],dtype=np.float32,key_head=6),
                ],max_size=self.ppo_hp.PPO_NUM_STEPS
        )
        agent_time=0
        env_time=0
        ec_time=0
        collect_time=0
        ppo_reward=0
        ppo_inner_reward=0
        ppo_env_reward=0

        with paddle.no_grad():
            for _iter in range(self.ppo_hp.PPO_NUM_STEPS):
                # print("iter=",_iter)
                #agent step
                t1=time.time()
                agent_output_dict=self.agent.interact(self.input_dict,False,{'actor_init_h':None,'critic_init_h':None})
                t2=time.time()
                agent_time+=t2-t1
                agent_value=agent_output_dict['value']
                agent_action_discrete,agent_action_discrete_log_prob=agent_output_dict['action_discrete'],agent_output_dict['action_discrete_log_probs']
                agent_action_discrete_onehot=one_hot([int(agent_action_discrete)],brain_hyperparam.env_action_discrete_num)
                agent_action_continuous,agent_action_continuous_log_prob=agent_output_dict['action_continuous'] ,agent_output_dict['action_continuous_log_probs']
                agent_actor_gru_h,agent_critic_gru_h=agent_output_dict['actor_gru_h'] ,agent_output_dict['critic_gru_h']
                    
                #env_step
                next_env_obs,env_reward,done,info=self.env.step(int(agent_action_discrete))
                t3=time.time()
                env_time+=(t3-t2)
                # data_step
                self.input_dict,self.pre_h_dict,inner_reward=self._env_obs_to_input_dict_and_inner_reward(next_env_obs,paddle.to_tensor(agent_action_discrete_onehot,'float32'))

                t4=time.time()
                ec_time+=(t4-t3)
                reward=env_reward+inner_reward
                rpm.collect({
                    'obs':np.reshape(next_env_obs,[-1]),#下一步图像
                    'action_env':np.reshape(agent_action_discrete_onehot,[-1]),#当前动作
                    'action_ec':np.reshape(agent_action_continuous,[-1]),
                    'ec_glo_gru':np.reshape(self.pre_h_dict['pre_glo_h'],[-1]),
                    'ec_loc_gru':np.reshape(self.pre_h_dict['pre_loc_h'],[-1]),
                    'ec_klpred_gru':np.reshape(self.pre_h_dict['pre_klpred_h'],[-1]),
                    'reward':[reward]
                }) 
                data_buffer_data={
                        'env_vec':self.input_dict['env_vec'].numpy().reshape([1,-1]),
                        'action_discrete':np.array([agent_action_discrete]),
                        'action_discrete_log_prob':np.array([agent_action_discrete_log_prob]),
                        'action_continuous':np.array(agent_action_continuous),
                        'action_continuous_log_prob':np.array([agent_action_continuous_log_prob]),
                        'value_pred':np.array([agent_value]),
                        'reward':np.array([reward]),
                        'actor_gru_h':np.array(agent_actor_gru_h),
                        'critic_gru_h':np.array(agent_critic_gru_h),
                        'mask':np.array([[1.0]]),
                        'bad_mask':np.array([[1.0]]),
                    }
                # print({key:np.shape(data_buffer_data[key]) for key in data_buffer_data.keys()})
                ppo_data_buffer.append(data_buffer_data)
                t5=time.time()
                collect_time+=t5-t4
                self.env_img=next_env_obs

                ppo_reward+=reward
                ppo_env_reward+=env_reward
                ppo_inner_reward+=inner_reward
        print(f"step{self.ppo_hp.PPO_NUM_STEPS},agent cost {agent_time},env cost {env_time},ec cost {ec_time},collect cost {t5-t4}")
            
        cache_update={
            "env_interact_forppo":ppo_data_buffer.get_data_dict(),
            'env_interact_fordata':rpm.get_data_dict(),
            'env_interact_reward':ppo_reward,
            'env_interact_inner_reward':ppo_inner_reward,
            'env_interact_env_reward':ppo_env_reward,
            }
        return cache_update
    def update_all_param(self,cache,**kwargs):
        self.agent.update_model_from_np(cache['ppo_getagentparam'])
        self.ec.update_model_from_np(cache['ec_getecparam'])
        cache_update={}
        return cache_update

class PPOFilter(ProcessFilter):
    def __init__(self, queue_dict,name='ppofilter') -> None:
        input_output_info=brain_hyperparam.PPO_input_output_info
        self.agent=Agent(
            PPO_GRU_HYBRID(
                Model(input_output_info,{'actor_class':Actor,'critic_class':Critic}),input_output_info
            ),
            input_output_info
        )
        self.agent.load_model(brain_hyperparam.DPPO_Hybrid_HyperParam.MODEL_SAVE_DIR,"newest")
        self.data_buffer_list=[RolloutStorage(input_output_info)]

        self._logger=mllogger(log_dir='log/ppolog')
        self._train_iter=0
        super().__init__(queue_dict,name)
    def get_agent_param(self,cache,**kwargs):
        param=self.agent.send_model_to_np()
        cache_update={'ppo_getagentparam':param}
        return cache_update
    def save_model(self,cache,**kwargs):
        if self._train_iter%kwargs['save_interval']==0:
            self.agent.save_model(kwargs['save_dir'],'newest')
        return {}
    def train(self,cache,**kwargs):
        "收集数据并train"
        self.data_buffer_list[0].update_data_dict(cache['env_interact_forppo'])
        learn_loss=self.agent.learn(self.data_buffer_list)
        value_loss_epoch,action_continuous_loss_epoch,action_continuous_dist_entropy_epoch,action_discrete_loss_epoch, action_discrete_dist_entropy_epoch,value_clip_mean_ratio_epoch=learn_loss
        
        self._logger.log_dict({
            'iter':self._train_iter,
            'reward':cache['env_interact_reward'],
            'inner_reward':cache['env_interact_inner_reward'],
            'env_reward':cache['env_interact_env_reward'],
            'v_loss':value_loss_epoch[0],
            'action_discrete_loss':action_discrete_loss_epoch[0],
            'action_discrete_dist_entropy':action_discrete_dist_entropy_epoch[0],
            'action_continuous_loss':action_continuous_loss_epoch[0],
            'action_continuous_dist_entropy':action_continuous_dist_entropy_epoch[0],
        })
        self._train_iter+=1

        
        [data_buffer.after_update() for data_buffer in self.data_buffer_list]
        cache_update={}
        return cache_update
    

class DataFilter(ProcessFilter):
    def __init__(self, queue_dict,name='datafilter') -> None:
        self.rpm=LMDBRecoder(
            data_format_list=[
                DataFormat(name='obs',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32,key_head=0),
                DataFormat(name='action_env',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32,key_head=1),
                DataFormat(name='action_ec',shape=[brain_hyperparam.ec_action_continuous_dim],dtype=np.float32,key_head=2),
                DataFormat(name='ec_glo_gru',shape=[brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32,key_head=3),
                DataFormat(name='ec_loc_gru',shape=[brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32,key_head=4),
                DataFormat(name='ec_klpred_gru',shape=[brain_hyperparam.ec_info.klpred_gru_dim],dtype=np.float32,key_head=5),
                DataFormat(name='reward',shape=[1],dtype=np.float32,key_head=6),
                ],rpm_buffer_size=brain_hyperparam.rpm_buffer_size,lmdb_dir="./lmdbdata",
                map_size=2**40
        )
        super().__init__(queue_dict,name)
    def collect(self,cache,**kwargs):
        env_data=cache[kwargs['data_key']]
        self.rpm.collect_dict_of_batch(env_data)
        cache_update={}
        return cache_update
    def sample_data(self,cache,**kwargs):
        batch_size=kwargs['batch_size']
        seq_len=kwargs['seq_len']
        if len(self.rpm)<brain_hyperparam.rpm_start_train_size or len(self.rpm.rpm)<self.rpm.rpm_buffer_size:
            return {'data_sampledata':None}
        else:
            ori_data=self.rpm.sample_batch_seq(batch_size=batch_size,seq_len=seq_len,use_recent_rpm=kwargs['use_recent_rpm'])
            # print(ori_data.keys())
            cache_update={'data_sampledata':ori_data}
            return cache_update
    def sample_h_update_seq(self,cache,**kwargs):
        if len(self.rpm)<brain_hyperparam.rpm_start_train_size:
            return {'data_samplehupdateseq':None,'next_info':None}
        else:
            start_index=cache['info']['start_index']
            seq_len=cache['info']['seq_len']
            sample=self.rpm.sample_h_update_seq(start_index,seq_len)

            next_start_index=start_index+seq_len
            next_start_index=next_start_index if next_start_index<self.rpm.max_index+1 else 0
            next_seq_len=min(cache['info']['default_seq_len'],self.rpm.max_index+1-next_start_index)
            cache_update={
                'data_samplehupdateseq':sample,
                'next_info':{
                    "start_index":next_start_index,
                    "seq_len":next_seq_len
                }
            }
            return cache_update
    def h_update(self,cache,**kwargs):
        if cache['data_samplehupdateseq'] is None:
            return {}
        else:
            glo_h=cache['ec_predh']['glo_h']
            loc_h=cache['ec_predh']['loc_h']
            klpred_h=cache['ec_predh']['klpred_h']
            index=cache['data_samplehupdateseq']['index']
            self.rpm.h_update(glo_h,loc_h,klpred_h,index)
            cache_update={}
            return cache_update
    def clear_buffer(self,cache,**kwargs):
        self.rpm.clear_buffer()

        return {}




class EcFilter(ProcessFilter):
    def __init__(self, queue_dict,name='ecfilter') -> None:
        SeqVaePred=import_ec()
        self.ec=SeqVaePred(brain_hyperparam.ec_info)
        self.ec.load_model(brain_hyperparam.ec_info.model_dir,'newest')
        self.train_steps=1
        self._logger=mllogger('log/eclog')
        super().__init__(queue_dict,name)
    def pred_h(self,cache,**kwargs):
        s_history=cache[kwargs['cache_data_key']]['obs']
        if s_history is None:
            return {
                'ec_predh':None
            }
        else:
            a_history=cache[kwargs['cache_data_key']]['action_env']
            pre_glo_h=cache['info']['pre_glo_h']
            pre_loc_h=cache['info']['pre_loc_h']
            pre_klpred_h=cache['info']['pre_klpred_h']
            pred_glo_h,pred_loc_h,pred_klpred_h=self.ec.pred_h(s_history,a_history,pre_glo_h,pre_loc_h,pre_klpred_h)
            cache_update={
                'ec_predh':{
                    "glo_h":np.concatenate([pre_glo_h,pred_glo_h[:,:-1]],axis=1),
                    "loc_h":np.concatenate([pre_loc_h,pred_loc_h[:,:-1]],axis=1),
                    "klpred_h":np.concatenate([pre_klpred_h,pred_klpred_h[:,:-1]],axis=1),
                    "next_glo_h":pred_glo_h[:,-1:],
                    "next_loc_h":pred_loc_h[:,-1:],
                    "next_klpred_h":pred_klpred_h[:,-1:]
                }
            }
            return cache_update

    def train(self,cache,**kwargs):
        data_sampledata=cache['data_sampledata']
        if data_sampledata is None:
            return {}
        else:
            self.train_steps+=1
            action_history=data_sampledata['action_env']
            state_history=data_sampledata['obs']
            global_h_history=data_sampledata['ec_glo_gru']
            local_h_history=data_sampledata['ec_loc_gru']
            klpred_h_history=data_sampledata['ec_klpred_gru']

            loss_info_dict=self.ec.train_with_warmup(state_history,action_history,global_h_history[:,:1,:],local_h_history[:,:1,:],klpred_h_history[:,:1,:])
            if self.train_steps%100==0:
                self._logger.log_dict(loss_info_dict)
            
            cache_update={}
            return cache_update


    def get_ec_param(self,cache,**kwargs):
        cache_update={"ec_getecparam":self.ec.send_model_to_np()}
        return cache_update
    def save_model(self,cache,**kwargs):
        if self.train_steps%kwargs['save_interval']==0:
            # self.ec.save_model(kwargs['save_dir'],self.train_steps)
            self.ec.save_model(kwargs['save_dir'],'newest')
        # print("cae save model finish")
        return {}

class EchpredFilter():
    def __init__(self, queue_dict) -> None:
        SeqVaePred=import_ec()
        self.ec=SeqVaePred(brain_hyperparam.ec_info)
        super().__init__(queue_dict)
    # def pred

class HpcFilter(ProcessFilter):
    def __init__(self, queue_dict) -> None:
        super().__init__(queue_dict)
    def pred(self,**kwargs):
        pass
    def train(self,**kwargs):
        pass
    def train_with_pred(self,**kwargs):
        pass