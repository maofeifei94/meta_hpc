from multiprocessing import Pipe,Process
import threading
import hyperparam as hp
class trainer_info():
    env_name='BreakoutDeterministic-v4'
    env_img_shape=[3,224,160]

    
    ec_rpm_size=int(1e5)
    ec_rpm_start_train_size=int(1e4)
    ec_train_batchsize=128
    ec_cvae_train_sleep_ratio=0.5

    ec_cvae_model_num='newest'

    # pfc_start_train_epoch=int(ec_rpm_size/hp.PPO_NUM_STEPS)+1
    # pfc_let_ec_train_time=300


    
from mlutils.multiprocess import sd_to_np,set_sd_from_np
from mlutils.env import EnvNeverDone
from pfc.gru_pfc_gym_discrete import PPO_GRU_Module
from ec.conv_VAE import ConvVAE
from ec.seq_VAE import SeqVAE
from ec.seq_pred import SeqPred
import gym,time,logging,copy,paddle
import numpy as np
import hyperparam as hp
import cv2


#******env
env = gym.make(trainer_info.env_name)
env=EnvNeverDone(env)
#******ConvVAE
cvae=ConvVAE()
cvae.eval()
cvae.load_model("all_models/ec_cvae_model",trainer_info.ec_cvae_model_num)

#******pfc ppo module
ppo_input_output_info={
    'to_static':True,
    'env_vec_dim':cvae.info.vae_gauss_dim,
    'obs_dim':cvae.info.vae_gauss_dim,
    'action_env_dim':env.action_space.n,
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


def update_pfc():
    pfc_interact.update(pfc_train.model.state_dict())
def cvae_encode(img):
    img_paddle=paddle.cast(paddle.to_tensor(img,'uint8'),'float32')
    img_paddle=paddle.expand(img_paddle,[1,*img_paddle.shape])/255
    gauss_mean=cvae.pred_gauss_mean(img_paddle)
    return paddle.reshape(gauss_mean,[1,1,-1])
def cvae_recon(img):
    img_paddle=paddle.cast(paddle.to_tensor(img,'uint8'),'float32')
    img_paddle=paddle.expand(img_paddle,[1,*img_paddle.shape])/255
    recon_img=cvae.pred_recon(img_paddle)
    return paddle.cast(recon_img*255,'uint8').numpy()[0]

def process_env_data_to_rpm(x):
    return np.transpose(np.pad(x,((7,7),(0,0),(0,0)),'constant'),(2,0,1))
def RL_loop():
    # print("rl loop 1")
    env_obs = env.reset()
    # print("rl loop 1.1")
    print(np.shape(env_obs))
    env_obs=process_env_data_to_rpm(env_obs)

    ppo_input_dict={
        'env_vec':cvae_encode(env_obs),
    }
    for e in range(int(1e10)):
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
                next_env_obs=process_env_data_to_rpm(next_env_obs)
                ppo_reward+=reward
                recon_next_env_obs=cvae_recon(next_env_obs)

                cv2.imshow("ori_obs",cv2.resize(np.transpose(next_env_obs,[1,2,0]),None,fx=3,fy=3,interpolation=cv2.INTER_AREA))
                cv2.imshow("recon_obs",cv2.resize(np.transpose(recon_next_env_obs,[1,2,0]),None,fx=3,fy=3,interpolation=cv2.INTER_AREA))
                cv2.waitKey()
                

                #pfc collect
                cvae_next_env_obs=cvae_encode(next_env_obs)
                next_ppo_input_dict={
                    'env_vec':cvae_next_env_obs,
                }
                #next iter
                ppo_input_dict=next_ppo_input_dict
                env_obs=next_env_obs


RL_loop()


if __name__=="__main__":
    RL_loop()
        




