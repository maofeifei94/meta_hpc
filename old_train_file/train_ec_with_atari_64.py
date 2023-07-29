from multiprocessing import Pipe,Process
import threading
import hyperparam as hp
class trainer_info():
    env_name='BreakoutDeterministic-v4'
    env_img_shape=[3,64,64]
    env_img_rand_part_times=10

    ec_rpm_size=int(8e5)
    ec_rpm_start_train_size=int(1e5)
    ec_train_batchsize=64

    ec_use_cvae=True
    ec_use_svae=True

    "cvae"
    ec_cvae_load_newest_model=False
    ec_cvae_model_dir="all_models/ec_cvae64_model"
    ec_cvae_train_sleep_ratio=0.0 #cvae休息与训练时间比例
    ec_cvae_vec_dim=256
    "svae"
    ec_svae_load_newest_model=False
    ec_svae_model_dir="all_models/ec_svae_model"
    ec_svae_vec_dim=256

    "pfc"
    pfc_start_train_epoch=int(ec_rpm_size/(hp.PPO_NUM_STEPS*env_img_rand_part_times))+1
    pfc_let_ec_train_time=3600*3 #cvae预热训练时间 单位s

    # pfc_start_train_epoch=0
    # pfc_let_ec_train_time=0
    


class trainer():
    def __init__(self) -> None:
        pipe_name_list=[
            "env_data_env","env_data_ec",
            "query_state_dict_env","query_state_dict_ec",

            "state_dict_ec","state_dict_env",
            ]
        pipe_dict={}
        for i in range(len(pipe_name_list)//2):
            pipe_send,pipe_recv=Pipe()
            pipe_dict[pipe_name_list[i*2]]=pipe_send
            pipe_dict[pipe_name_list[i*2+1]]=pipe_recv
        
        p1=Process(target=self.agent_env_process,args=(pipe_dict,))
        p2=Process(target=self.train_ec_process,args=(pipe_dict,))
        p_list=[p1,p2]
        [p.start() for p in p_list]
        [p.join() for p in p_list]
        

    def agent_env_process(self,pipe_dict):
        from mlutils.multiprocess import sd_to_np,set_sd_from_np
        from mlutils.env import EnvNeverDone
        from pfc.gru_pfc_gym_discrete import PPO_GRU_Module
        from ec.conv_VAE_64 import ConvVAE
        from ec.seq_VAE import SeqVAE
        from ec.seq_pred import SeqPred
        import gym,time,logging,copy,paddle
        import numpy as np
        import hyperparam as hp

        #******env
        env = gym.make(trainer_info.env_name)
        env=EnvNeverDone(env)
        logging.basicConfig(filename="log/"+time.strftime("%Y_%m%d_%H%M%S",time.localtime(time.time()))+".txt", level=logging.INFO)
        #******ConvVAE
        cvae=ConvVAE()
        cvae.eval()

        #******pfc ppo module
        ppo_input_output_info={
            'to_static':True,
            'env_vec_dim':cvae.info.vae_gauss_dim,
            'obs_dim':cvae.info.vae_gauss_dim,
            'action_env_discrete_dim':env.action_space.n,
            'action_env_continuous_dim':2,
            'action_dim':env.action_space.n,
            'actor_gru_hid_dim':256,
            'critic_gru_hid_dim':256
        }
        pfc_interact=PPO_GRU_Module(ppo_input_output_info)
        pfc_interact.model.eval() 
        ppo_input_output_info_train=copy.deepcopy(ppo_input_output_info)
        ppo_input_output_info_train['to_static']=False
        pfc_train=PPO_GRU_Module(ppo_input_output_info_train)
        pfc_interact.update(pfc_train.model.state_dict())
        pfc_interact._reset()
        pfc_train._reset()

        def update_cvae():
            pipe_dict['query_state_dict_env'].send(True)
            # print("send need sd success")
            sd_np_dict=pipe_dict['state_dict_env'].recv()
            set_sd_from_np(cvae.state_dict(),sd_np_dict['ConvVAE'])
            # print("update cvae success")
        def update_pfc():
            pfc_interact.update(pfc_train.model.state_dict())
        def cvae_encode(img):
            # print("cvae_encode 1")
            img_paddle=paddle.cast(paddle.to_tensor(img,'uint8'),'float32')
            # print("cvae_encode 2")
            img_paddle=paddle.expand(img_paddle,[1,*img_paddle.shape])/255
            # print("cvae_encode 3")
            gauss_mean=cvae.pred_gauss_mean(img_paddle)
            # print("cvae_encode 4")
            return paddle.reshape(gauss_mean,[1,1,-1])

        def RL_loop():
            # print("rl loop 1")
            env_obs = env.reset()
            # print("rl loop 1.1")
            env_obs=np.transpose(env_obs,[2,0,1])
            # print("rl loop 1.2")
            # print(np.shape(env_obs))
            ppo_input_dict={
                'env_vec':cvae_encode(env_obs),
            }


            #warm up
            for _e in range(trainer_info.pfc_start_train_epoch):
                env_data_dict_list=[]
                for i in range(hp.PPO_NUM_STEPS):
                    for j in range(trainer_info.env_img_rand_part_times):
                        rand_slice_y=np.random.randint(0,np.shape(env_obs)[1]-trainer_info.env_img_shape[1])
                        rand_slice_x=np.random.randint(0,np.shape(env_obs)[2]-trainer_info.env_img_shape[2])
                        env_data_dict_list.append({
                            "env_img":env_obs[:,rand_slice_y:rand_slice_y+trainer_info.env_img_shape[1],rand_slice_x:rand_slice_x+trainer_info.env_img_shape[2]],
                        })
                    next_env_obs,reward,done,info= env.step(env.action_space.sample())
                    next_env_obs=np.transpose(next_env_obs,[2,0,1])
                    #next iter
                    env_obs=next_env_obs
                pipe_dict['env_data_env'].send(env_data_dict_list)
                print(_e,"random action")
            time.sleep(trainer_info.pfc_let_ec_train_time)
            for e in range(int(1e10)):
                update_cvae()
                update_pfc()
                env_data_dict_list=[]
                ppo_reward=0
                collect_time=time.time()
                with paddle.no_grad():
                    for i in range(hp.PPO_NUM_STEPS):
                        #pfc action
                        h_dict={
                            'actor_init_h':None,
                            'critic_init_h':None
                        }
                        pfc_output_dict=pfc_interact._input(ppo_input_dict,train=False,h_dict=h_dict)
                        pfc_value,pfc_action,pfc_action_log_prob,pfc_actor_gru_h,pfc_critic_gru_h=[pfc_output_dict[key] for key in ['value','action','action_log_prob','actor_gru_h','critic_gru_h']]
                        #ec data
                        env_data_dict_list.append({
                            "env_img":env_obs,
                        })
                        #env step
                        next_env_obs,reward,done,info= env.step(pfc_action)
                        next_env_obs=np.transpose(next_env_obs,[2,0,1])
                        ppo_reward+=reward
                        
                        #pfc collect
                        cvae_next_env_obs=cvae_encode(next_env_obs)
                        next_ppo_input_dict={
                            'env_vec':cvae_next_env_obs,
                        }
                        masks = paddle.to_tensor([[1.0]] , dtype='float32')
                        bad_masks = paddle.to_tensor([[1.0]],dtype='float32')
                        pfc_train._rpm_collect({
                            'env_vec':cvae_next_env_obs.numpy().reshape([-1]),
                            "model_probs":pfc_output_dict['model_probs'],
                            'action':pfc_action,
                            'action_log_prob':pfc_action_log_prob,
                            'value_pred':pfc_value,
                            'reward':reward,
                            'actor_gru_h':pfc_actor_gru_h,
                            'critic_gru_h':pfc_critic_gru_h,
                            'mask':masks,
                            'bad_mask':bad_masks,
                        })

                        #next iter
                        ppo_input_dict=next_ppo_input_dict
                        env_obs=next_env_obs


                
                pipe_dict['env_data_env'].send(env_data_dict_list)
                print(f"iter {e} ppo reward={ppo_reward/hp.PPO_NUM_STEPS}")
                print(f"collect data cost {time.time()-collect_time}")
                value_loss,action_loss,dist_entropy=pfc_train._learn()
                logging.info(
                    f"|iter={e}"+
                    # f"|inner_reward={ppo_inner_reward/hp.PPO_NUM_STEPS}"+
                    # f"|env_reward={ppo_env_reward/hp.PPO_NUM_STEPS}"+
                    f"|reward={ppo_reward/hp.PPO_NUM_STEPS}"+
                    f"|abs_action_mean={np.mean(np.abs(pfc_train.rpm.actions))}"+
                    f"|v_loss={value_loss[0]}"+
                    f"|action_loss={action_loss[0]}"+
                    f"|dist_entropy={dist_entropy[0]}")
                if e%20==0:
                    pfc_train.save_model("all_models/pfc_model",e)
                    
                    cvae.save_model(trainer_info.ec_cvae_model_dir,e)
                    cvae.save_model(trainer_info.ec_cvae_model_dir,"newest")

        RL_loop()
            
    def train_ec_process(self,pipe_dict):
        from mlutils.multiprocess import sd_to_np,set_sd_from_np
        from ec.conv_VAE_64 import ConvVAE
        from ec.seq_VAE import SeqVAE
        from ec.seq_pred import SeqPred
        from mlutils.ml import ReplayMemory,DataFormat
        import numpy as np
        import time

        dataformat_list=[DataFormat("env_img",trainer_info.env_img_shape,np.uint8)]
        if trainer_info.ec_use_svae:
            dataformat_list.append(DataFormat("cvae_vec",trainer_info.ec_cvae_vec_dim,np.float32))
            dataformat_list.append(DataFormat("svae_vec",trainer_info.ec_svae_vec_dim,np.float32))
        rpm=ReplayMemory(
            dataformat_list,max_size=trainer_info.ec_rpm_size
        )
        cvae=ConvVAE()
        if trainer_info.ec_cvae_load_newest_model:
            cvae.load_model(trainer_info.ec_cvae_model_dir,"newest")

        def thread_send_state_dict():
            while 1:
                need_sd=pipe_dict['query_state_dict_ec'].recv()
                # print("get need sd")
                if need_sd:
                    pipe_dict['state_dict_ec'].send({
                        'ConvVAE':sd_to_np(cvae.state_dict())
                    })
                # print("send sd success")
        def thread_collect_data():
            while 1:
                env_data_list=pipe_dict['env_data_ec'].recv()
                for env_data in env_data_list:
                    "env_data只包含env_img"
                    if trainer_info.ec_use_svae:
                        env_data["cvae_vec"]=
                    rpm.collect(env_data)
        def thread_train_cvae():
            train_steps=0
            while 1:
                # print(rpm._size)
                if rpm._size>=trainer_info.ec_rpm_start_train_size:
                    t1=time.time()
                    train_dict=rpm.sample_batch(trainer_info.ec_train_batchsize)
                    cvae_loss=cvae.train(cvae.chw_uint8_to_paddle(train_dict['env_img']))
                    cost_time=time.time()-t1
                    # print(f"ec train cost {time.time()-t1}")
                    train_steps+=1
                    time.sleep(cost_time*trainer_info.ec_cvae_train_sleep_ratio)
                    if train_steps%1000==0:
                        
                        print(f"ec train_steps{train_steps},loss={cvae_loss}")
                    if train_steps%10000==0:
                        cvae.save_model(trainer_info.ec_cvae_model_dir,train_steps)
                        cvae.save_model(trainer_info.ec_cvae_model_dir,"newest")
                else:
                    print(f"rpm size {rpm._size} < {trainer_info.ec_rpm_start_train_size}, not train")
                    time.sleep(5.0)
        




        thread1=threading.Thread(target=thread_send_state_dict)
        thread2=threading.Thread(target=thread_collect_data)

        thread3=threading.Thread(target=thread_train_cvae)
        thread_list=[thread1,thread2,thread3]
        [t.start() for t in thread_list]
        [t.join() for t in thread_list]

        
        


if __name__=="__main__":
    trainer()
        




