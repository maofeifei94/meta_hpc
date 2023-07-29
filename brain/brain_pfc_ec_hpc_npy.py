import numpy as np
import time
from mlutils.ml import ModelSaver,one_hot
from mlutils.multiprocess import ProcessFilter,TaskFlow,Task,Cache
from mlutils.ml import DataFormat,ReplayMemory
from multiprocessing import Queue,Process,Manager
from multiprocessing import Lock as process_Lock
import cv2
import os
import paddle
from paddle import nn
#env
from env.env_animalai import Animal
#brain
from brain.hyperparam.brain_pfc_ec_hpc_hp import brain_hyperparam
#pfc
from pfc.ac_net.ppo_gru_hybrid_actor import Actor,Actor_Static
from pfc.ac_net.ppo_gru_hybrid_critic import Critic,Critic_Static
from pfc.ac_model.ppo_gru_hybrid_model import Model
from pfc.agent.ppo_gru_hybrid_dcu_agent import Agent
from pfc.alg.ppo_gru_hybrid_dcu_alg import PPO_GRU_HYBRID
from pfc.data_buffer.ppo_gru_hybrid_rolloutstorage import RolloutStorage
from mlutils.logger import mllogger
from mlutils.data import DataRecorder

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
        self._run()
    def _send(self,flow:TaskFlow):
        """
        根据flow的信息,将数据发送到下一个进程专属的Queue中
        """
        if flow.left_steps>0:
            self.queue_dict[flow.current_task.filter_name][flow.priority].put(flow)
    def _run(self):
        """
        interact and train
        """
        steps=0
        time.sleep(5)
        interact_flow=self.generate_env_interact_flow()
        self._send(interact_flow)
        while 1:
            #收集env数据
            if self.queue_dict['flowgenerator'].qsize()>0:
                # print("send new interact flow")
                self.queue_dict['flowgenerator'].get()
                interact_flow=self.generate_env_interact_flow()
                self._send(interact_flow)

            train_flow=self.generate_train_flow()
            self._send(train_flow)
            steps+=1
            time.sleep(0.1)
    # def _run(self):
    #     """
    #     only train
    #     """
    #     steps=0
    #     time.sleep(5)
    #     while 1:
    #         train_flow=self.generate_train_flow()
    #         self._send(train_flow)
    #         steps+=1
    #         time.sleep(0.1)
    def generate_env_interact_flow(self):
        tasklist=[
            #更新参数
            Task(filter_name='ppofilter',func_name='get_agent_param'),
            Task(filter_name='caefilter',func_name='get_cae_param'),
            Task(filter_name='ecfilter',func_name='get_ec_param'),
            Task(filter_name='envfilter',func_name='update_all_param'),
            #获取数据
            Task(filter_name='envfilter',func_name='interact'),
            #训练模型
            Task(filter_name='datafilter',func_name='collect'),
            Task(filter_name='ppofilter',func_name='train'),
            Task(filter_name='ppofilter',func_name='save_model',save_interval=20,save_dir=brain_hyperparam.DPPO_Hybrid_HyperParam.MODEL_SAVE_DIR),
            Task(filter_name='finishfilter',func_name='finish_interact')
        ]
        
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='high')
    def generate_train_flow(self):
        tasklist=[
            Task(filter_name='datafilter',func_name='sample_data',batch_size=brain_hyperparam.train_batch_size,seq_len=brain_hyperparam.tran_seq_len),
            Task(filter_name='caefilter',func_name='train_with_pred',img_save_dir="all_data/cae_img"),
            Task(filter_name='caefilter',func_name='save_model',save_interval=1000,save_dir=brain_hyperparam.cae_hyperparam.model_dir),
            Task(filter_name='ecfilter',func_name='train'),
            Task(filter_name='ecfilter',func_name='save_model',save_interval=1000,save_dir=brain_hyperparam.ec_info.model_dir)
        ]
        cache=Cache(self.share_info,self.lock)
        return TaskFlow(tasklist=tasklist,cache=cache,priority='low')
class FinishFilter(ProcessFilter):
    def finish_interact(self,cache,**kwargs):
        self.queue_dict['flowgenerator'].put(None)
        cache_update={}
        return cache_update
class EnvFilter(ProcessFilter):
    def __init__(self, queue_dict,name='envfilter') -> None:
        #env
        self.env=Animal()
        #cae
        Conv_Auto_Encoder=import_cae()
        self.cae=Conv_Auto_Encoder(brain_hyperparam.cae_hyperparam)
        # ec
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
        self.input_dict,self.pre_h_dict=self._env_obs_to_input_dict(env_obs,one_hot([0],brain_hyperparam.env_action_discrete_num))

    def _env_obs_to_input_dict(self,env_obs,action_env_discrete_onehot):
        env_obs_bchw=np.expand_dims(np.transpose(env_obs,[2,0,1]),axis=0).astype(np.float32)/255
        cae_vec=self.cae.encode(paddle.to_tensor(env_obs_bchw)).reshape([1,1,-1])
        ec_vec,pre_glo_h,pre_loc_h=self.ec.pred_env(cae_vec.reshape([1,1,-1]),action_env_discrete_onehot.reshape([1,1,-1]))

        input_dict={
            'env_vec':paddle.concat([cae_vec,ec_vec],axis=-1),
                    }
        pre_h_dict={
            'pre_glo_h':pre_glo_h.numpy().reshape([1,-1]),
            'pre_loc_h':pre_loc_h.numpy().reshape([1,-1]),
        }
        # print("encode_vec.shape=",encode_vec.shape)
        return input_dict,pre_h_dict

    def interact(self,cache,**kwargs):
        ppo_data_buffer=RolloutStorage(self.ppo_input_output_info)
        rpm=ReplayMemory(
            {
                DataFormat(name='img',shape=[3,64,64],dtype=np.uint8),
                DataFormat(name='action_env',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32),
                DataFormat(name='action_ec',shape=[brain_hyperparam.ec_action_continuous_dim],dtype=np.float32),
                DataFormat(name='ec_glo_gru',shape=[brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32),
                DataFormat(name='ec_loc_gru',shape=[brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32),
                DataFormat(name='reward',shape=[1],dtype=np.float32),
            },max_size=self.ppo_hp.PPO_NUM_STEPS
        )

        agent_time=0
        env_time=0
        ppo_reward=0

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
                next_env_obs,reward,done,info=self.env.step(int(agent_action_discrete))
                
                # data_step
                self.input_dict,self.pre_h_dict=self._env_obs_to_input_dict(next_env_obs,agent_action_discrete_onehot)
                rpm.collect({
                    'img':np.transpose(next_env_obs,[2,0,1]),#下一步图像
                    'action_env':np.reshape(agent_action_discrete_onehot,[-1]),#当前动作
                    'action_ec':np.reshape(agent_action_continuous,[-1]),
                    'ec_glo_gru':np.reshape(self.pre_h_dict['pre_glo_h'],[-1]),
                    'ec_loc_gru':np.reshape(self.pre_h_dict['pre_loc_h'],[-1]),
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
                self.env_img=next_env_obs
                t3=time.time()
                env_time+=(t3-t2)
                ppo_reward+=reward
        print(f"step{self.ppo_hp.PPO_NUM_STEPS},agent cost {agent_time},env cost {env_time}")
            
        cache_update={
            "env_interact_forppo":ppo_data_buffer.get_data_dict(),
            'env_interact_fordata':rpm.get_data_dict(),
            'env_interact_reward':ppo_reward,
            }
        return cache_update
    def update_all_param(self,cache,**kwargs):
        self.agent.update_model_from_np(cache['ppo_getagentparam'])
        self.cae.update_model_from_np(cache['cae_getcaeparam'])
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
        # self.rpm=ReplayMemory(
        #     [
        #         DataFormat(name='img',shape=[3,64,64],dtype=np.uint8),
        #         DataFormat(name='action_env',shape=[1],dtype=np.int32),
        #         DataFormat(name='action_ec',shape=[brain_hyperparam.ec_action_continuous_dim],dtype=np.float32),
        #         DataFormat(name='reward',shape=[1],dtype=np.float32),
        #     ],max_size=brain_hyperparam.rpm_max_size
        # )
        self.rpm=DataRecorder(
            save_steps=2048*10,
            buffer_max_steps=2048*50,
            max_save_rpm_num=100,
            data_format_list=[
                DataFormat(name='img',shape=[3,64,64],dtype=np.uint8),
                DataFormat(name='action_env',shape=[brain_hyperparam.env_action_discrete_num],dtype=np.float32),
                DataFormat(name='action_ec',shape=[brain_hyperparam.ec_action_continuous_dim],dtype=np.float32),
                DataFormat(name='ec_glo_gru',shape=[brain_hyperparam.ec_info.gloinfo_gru_dim],dtype=np.float32),
                DataFormat(name='ec_loc_gru',shape=[brain_hyperparam.ec_info.locinfo_gru_dim],dtype=np.float32),
                DataFormat(name='reward',shape=[1],dtype=np.float32),
                ],
            data_dir="all_data/ppo",
            current_rpm_sample_ratio=0.0
        )
        super().__init__(queue_dict,name)
    def collect(self,cache,**kwargs):
        env_data=cache['env_interact_fordata']
        self.rpm.collect_dict_of_batch(env_data)
        cache_update={}
        return cache_update
    def sample_data(self,cache,**kwargs):
        batch_size=kwargs['batch_size']
        seq_len=kwargs['seq_len']
        if len(self.rpm)<brain_hyperparam.rpm_start_train_size:
            time.sleep(2)
            return {}
        else:
            ori_data=self.rpm.sample_batch_seq(batch_size=batch_size,seq_len=seq_len)
            print(ori_data.keys())
            cache_update={'data_sampledata':ori_data}
            return cache_update
    def save_local_data(self,cache,**kwargs):
        img_data=cache['env_interact_fordata']['img']
        print(np.shape(img_data),img_data.dtype)
        return {}
class CaeFilter(ProcessFilter):
    def __init__(self,queue_dict,name='caefilter') -> None:
        Conv_Auto_Encoder=import_cae()
        from mlutils.logger import mllogger
        self.cae=Conv_Auto_Encoder(brain_hyperparam.cae_hyperparam)
        self.cae.load_model(brain_hyperparam.cae_hyperparam.model_dir,brain_hyperparam.cae_hyperparam.model_load)
        self.train_steps=1
        self._logger=mllogger('log/caelog')
        # print("cae init done")
        super().__init__(queue_dict,name)

        
    # def pred(self,**kwargs):
    #     cache_update={}
    #     return cache_update
    def _save_recon_img(self,ori_img,recon_img,img_dir,img_num):

        def paddle_to_np(img):
            return np.transpose((img).astype(np.uint8),[1,2,0])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        ori_img=paddle_to_np(ori_img[0][0])
        recon_img=paddle_to_np(recon_img[0].numpy()*255)
        # print(np.shape(ori_img),np.shape(recon_img))
        combine_img=cv2.resize(np.concatenate([ori_img,recon_img],axis=1),dsize=None,fx=4,fy=4,interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{img_dir}/{img_num}.jpg",combine_img)
        # print("write img finish")

    def get_cae_param(self,cache,**kwargs):
        cache_update={"cae_getcaeparam":self.cae.send_model_to_np()}
        return cache_update
    def train_with_pred(self,cache,**kwargs):
        if 'data_sampledata' in cache.keys():
            batch_seq_img=cache['data_sampledata']['img']
            img_cae_mean,recon_img=self.cae.train_with_pred(batch_seq_img)
            if self.train_steps%10==0:
                self._save_recon_img(batch_seq_img,recon_img,kwargs['img_save_dir'],"newest")
            if self.train_steps%100==0:
                self._logger.log_dict({
                    "loss_kl":self.cae.avg_loss_kl,
                    "loss_recon":self.cae.avg_loss_recon
                })
            cache_update={"cae_trainwithpred":img_cae_mean}
            self.train_steps+=1
            return cache_update
        else:
            return {}
        
    def save_model(self,cache,**kwargs):
        # print("cae save model")
        if self.train_steps%kwargs['save_interval']==0:
            self.cae.save_model(kwargs['save_dir'],self.train_steps)
            self.cae.save_model(kwargs['save_dir'],'newest')
        # print("cae save model finish")
        return {}



class EcFilter(ProcessFilter):
    def __init__(self, queue_dict,name='ecfilter') -> None:
        SeqVaePred=import_ec()
        self.ec=SeqVaePred(brain_hyperparam.ec_info)
        self.ec.load_model(brain_hyperparam.ec_info.model_dir,'newest')
        self.train_steps=1
        super().__init__(queue_dict,name)

    def train(self,cache,**kwargs):
        if 'data_sampledata' in cache.keys():
            self.train_steps+=1
            action_history=cache['data_sampledata']['action_env']
            state_history=cache['cae_trainwithpred']
            global_h_history=cache['data_sampledata']['ec_glo_gru']
            local_h_history=cache['data_sampledata']['ec_loc_gru']
            loss_list=self.ec.train_with_warmup(state_history,action_history,global_h_history[:,:1,:],local_h_history[:,:1,:])
            cache_update={}
            return cache_update
        else:
            return {}

    def get_ec_param(self,cache,**kwargs):
        cache_update={"ec_getecparam":self.ec.send_model_to_np()}
        return cache_update
    def save_model(self,cache,**kwargs):
        if self.train_steps%kwargs['save_interval']==0:
            self.ec.save_model(kwargs['save_dir'],self.train_steps)
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