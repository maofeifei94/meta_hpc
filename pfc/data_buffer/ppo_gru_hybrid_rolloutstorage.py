import numpy as np

class RolloutStorage(object):
    def __init__(self, input_output_info):

        self.hp=input_output_info['hyperparam']
        self.env_vec_dim=input_output_info['env_vec_dim']
        self.num_steps=num_steps= self.hp.PPO_NUM_STEPS
        # self.action_dim=input_output_info['action_dim']


        # self.env_img=np.zeros((num_steps + 1, *input_output_info['env_img_shape']), dtype='float32')
        self.env_vec = np.zeros((num_steps + 1, self.env_vec_dim), dtype='float32')
        # self.hpc_img=np.zeros((num_steps + 1, *input_output_info['hpc_img_shape']), dtype='float32')
        # self.hpc_vec = np.zeros((num_steps + 1, input_output_info['hpc_vec_dim']), dtype='float32')

        # self.model_probs=np.zeros((num_steps, input_output_info['action_dim']), dtype='float32')

        ###  hybrid  ###
        self.action_discrete = np.zeros((num_steps, 1), dtype='int64')
        self.action_discrete_log_prob = np.zeros((num_steps, ), dtype='float32')
        self.action_continuous = np.zeros((num_steps, input_output_info['action_env_continuous_dim']), dtype='float32')
        self.action_continuous_log_prob = np.zeros((num_steps, ), dtype='float32')

        self.value_pred = np.zeros((num_steps+1, ), dtype='float32')
        self.returns = np.zeros((num_steps + 1, ), dtype='float32')
        # self.advantage=np.zeros((num_steps),dtype='float32')
        
        self.reward = np.zeros((num_steps, ), dtype='float32')

        self.actor_gru_h=np.zeros((num_steps+1,self.hp.PPO_GRU_HID_DIM),dtype='float32')
        # self.actor_gru2_h=np.zeros((num_steps,input_output_info['actor_gru2_hid_dim']),dtype='float32')
        self.critic_gru_h=np.zeros((num_steps+1,self.hp.PPO_GRU_HID_DIM),dtype='float32')

        self.mask = np.ones((num_steps + 1, ), dtype='bool')
        self.bad_mask = np.ones((num_steps + 1, ), dtype='bool')

        self.step = 0
    def get_data_dict(self):
        dict_key_list=[
            'env_vec','action_discrete','action_discrete_log_prob',
            'action_continuous','action_continuous_log_prob','value_pred',
            'reward','actor_gru_h','critic_gru_h',
            'mask','bad_mask']
        data_dict={}
        for key in dict_key_list:
            data=getattr(self,key)
            data_dict.update({key:data[len(data)-self.num_steps:]})
        return data_dict
    def update_data_dict(self,data_dict):
        self.append(data_dict)
    def append(self,collect_dict):
        collect_env_vec=collect_dict['env_vec']

        if len(np.shape(collect_env_vec))==len(np.shape(self.env_vec)):
            collect_size=len(collect_env_vec)
        elif len(np.shape(collect_env_vec))==len(np.shape(self.env_vec))-1:
            collect_size=1
        else:
            raise ValueError(f"rolloutstorage collect data of shape{np.shape(collect_env_vec)} not in [{np.shape(self.env_vec)},{np.shape(self.env_vec)[1:]}]")
        
        if self.step+collect_size>self.num_steps:
            raise ValueError(f"rolloutstorage step+collect_size={self.step}+{collect_size}>num_steps {self.num_steps}")
        
        self.env_vec[self.step + 1:self.step+1+collect_size] = collect_dict['env_vec']
        # self.hpc_img[self.step + 1] = collect_dict['hpc_img']
        # self.hpc_vec[self.step + 1] = collect_dict['hpc_vec']
        
        # self.model_probs[self.step] = collect_dict['model_probs']

        ###  hybrid  ###
        self.action_discrete[self.step:self.step+collect_size] = collect_dict['action_discrete']
        self.action_discrete_log_prob[self.step:self.step+collect_size] = collect_dict['action_discrete_log_prob']
        self.action_continuous[self.step:self.step+collect_size] = collect_dict['action_continuous']
        self.action_continuous_log_prob[self.step:self.step+collect_size] = collect_dict['action_continuous_log_prob']

        self.value_pred[self.step:self.step+collect_size] = collect_dict['value_pred']
        self.reward[self.step:self.step+collect_size] = collect_dict['reward']

        # self.actor_gru1_h[self.step]=collect_dict['actor_gru1_h']
        self.actor_gru_h[self.step+1:self.step+1+collect_size]=collect_dict['actor_gru_h']
        self.critic_gru_h[self.step+1:self.step+1+collect_size]=collect_dict['critic_gru_h']
        self.mask[self.step + 1:self.step+1+collect_size] = collect_dict['mask']
        self.bad_mask[self.step + 1:self.step+1+collect_size] = collect_dict['bad_mask']
        
        self.step = (self.step + collect_size) % self.num_steps

    def append_single(self, collect_dict):
        
        # self.env_img[self.step + 1] = collect_dict['env_img']
        self.env_vec[self.step + 1] = collect_dict['env_vec']
        # self.hpc_img[self.step + 1] = collect_dict['hpc_img']
        # self.hpc_vec[self.step + 1] = collect_dict['hpc_vec']
        
        # self.model_probs[self.step] = collect_dict['model_probs']

        ###  hybrid  ###
        self.actions_discrete[self.step] = collect_dict['action_discrete']
        self.action_discrete_log_probs[self.step] = collect_dict['action_discrete_log_prob']
        self.actions_continuous[self.step] = collect_dict['action_continuous']
        self.action_continuous_log_probs[self.step] = collect_dict['action_continuous_log_prob']


        self.value_preds[self.step] = collect_dict['value_pred']
        self.rewards[self.step] = collect_dict['reward']

        # self.actor_gru1_h[self.step]=collect_dict['actor_gru1_h']
        self.actor_gru_h[self.step+1]=collect_dict['actor_gru_h']
        self.critic_gru_h[self.step+1]=collect_dict['critic_gru_h']
        self.masks[self.step + 1] = collect_dict['mask']
        self.bad_masks[self.step + 1] = collect_dict['bad_mask']
        
        self.step = (self.step + 1) % self.num_steps
        # print(f"roll out storage step={self.step}")
        # print("step:",self.step)

    def after_update(self):
        # self.env_img[0]=np.copy(self.env_img[-1])
        self.env_vec[0]=np.copy(self.env_vec[-1])
        # self.hpc_img[0]=np.copy(self.hpc_img[-1])
        # self.hpc_vec[0]=np.copy(self.hpc_vec[-1])

        self.mask[0] = np.copy(self.mask[-1])
        self.bad_mask[0] = np.copy(self.bad_mask[-1])

        self.actor_gru_h[0]=np.copy(self.actor_gru_h[-1])
        self.critic_gru_h[0]=np.copy(self.critic_gru_h[-1])

    def _pad_vec(self,x,max_len):
        x_shape=np.shape(x)
        if len(x)==max_len:
            return x
        elif len(x_shape)==1:
            out=np.zeros([max_len],dtype=x.dtype)
            out[:len(x)]=x
            return out
        else:
            # print(max_len,x_shape)
            out=np.zeros([max_len,*x_shape[1:]],dtype=x.dtype)
            out[:len(x)]=x
            return out
    def gru_indices(self):
        self.train_history_len=self.hp.PPO_TRAIN_HISTORY_LEN
        self.train_control=[0.0 if i<self.train_history_len else 1.0 for i in range(self.train_history_len*2)]
        train_start_idx=np.random.randint(-self.train_history_len+2,self.num_steps-1)
        # train_start_idx=128
        read_start_idx=train_start_idx-self.train_history_len
        end_idx=train_start_idx+self.train_history_len
        history_slice=slice(max(read_start_idx,0),min(end_idx,self.num_steps))
        train_control=np.reshape(np.array(self.train_control[max(0,-read_start_idx):2*self.train_history_len+min(self.num_steps-end_idx,0)]),[-1,1])
        return history_slice,train_control
    def sample_batch(self):
            
        indices,train_control=self.gru_indices()
        
        train_control=self._pad_vec(train_control,self.train_history_len*2)
        actor_init_h=self.actor_gru_h[indices.start]
        actor_init_h=np.reshape(actor_init_h,[1,1,-1])
        # actor_init_h2=np.zeros_like(self.actor_gru2_h[0]) if indices.start==0 else self.actor_gru2_h[indices.start-1]
        # actor_init_h2=np.reshape(actor_init_h2,[1,1,-1])
        critic_init_h=self.critic_gru_h[indices.start]
        critic_init_h=np.reshape(critic_init_h,[1,1,-1])
        # indices=slice(0,self.num_steps)
        # train_control=np.ones([self.num_steps])
        # env_img_batch=self._pad_vec(self.env_img[:-1][indices],self.train_history_len*2)
        env_vec_batch=self._pad_vec(self.env_vec[:-1][indices],self.train_history_len*2)
        # hpc_img_batch=self._pad_vec(self.hpc_img[:-1][indices],self.train_history_len*2)
        # hpc_vec_batch=self._pad_vec(self.hpc_vec[:-1][indices],self.train_history_len*2)

        ###  hybrid  ###
        actions_discrete_batch=self._pad_vec(self.action_discrete[indices],self.train_history_len*2)
        actions_continuous_batch=self._pad_vec(self.action_continuous[indices],self.train_history_len*2)

        old_action_discrete_log_probs_batch = self._pad_vec(self.action_discrete_log_prob[indices],self.train_history_len*2)
        old_action_continuous_log_probs_batch = self._pad_vec(self.action_continuous_log_prob[indices],self.train_history_len*2)

        # model_probs_batch = self._pad_vec(self.model_probs[indices],self.train_history_len*2)
        
        value_preds_batch = self._pad_vec(self.value_pred[:-1][indices],self.train_history_len*2)
        returns_batch = self._pad_vec(self.returns[:-1][indices],self.train_history_len*2)
        adv_targ = self._pad_vec(self.advantages[indices],self.train_history_len*2)

        "reshape"
        value_preds_batch = value_preds_batch.reshape(-1, 1)
        returns_batch = returns_batch.reshape(-1, 1)
        # model_probs_batch=model_probs_batch.reshape(-1,self.action_dim)
        old_action_discrete_log_probs_batch=old_action_discrete_log_probs_batch.reshape(-1, 1)
        old_action_continuous_log_probs_batch=old_action_continuous_log_probs_batch.reshape(-1, 1)

        adv_targ = adv_targ.reshape(-1, 1)

        sample_dict={
            "indices":[indices.start,indices.stop],
            "train_control":train_control,
            "actor_init_h":actor_init_h,
            # "actor_init_h2":actor_init_h2,
        
            "critic_init_h":critic_init_h,

            # "env_img":env_img_batch,
            "env_vec":env_vec_batch,
            # "hpc_img":hpc_img_batch,
            # "hpc_vec":hpc_vec_batch,
            # "model_probs_batch":model_probs_batch,

            "actions_discrete_batch":actions_discrete_batch,
            "old_action_discrete_log_probs_batch":old_action_discrete_log_probs_batch,

            "actions_continuous_batch":actions_continuous_batch,
            "old_action_continuous_log_probs_batch":old_action_continuous_log_probs_batch,

            "value_preds_batch":value_preds_batch,
            "returns_batch":returns_batch,
            "adv_targ":adv_targ,
        }
        return sample_dict



    def compute_returns(self, next_value, gamma, gae_lambda):

        # input()
        self.value_pred[-1] = next_value
        gae = 0
        for step in reversed(range(self.reward.size)):
            delta = self.reward[step] + gamma * self.value_pred[
                step + 1] * self.mask[step + 1] - self.value_pred[step]

            gae = delta + gamma * gae_lambda * self.mask[step + 1] * gae

            gae = gae * self.bad_mask[step + 1]
            
            self.returns[step] = gae + self.value_pred[step]

        advantages = self.returns[:-1] - self.value_pred[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.advantages=advantages