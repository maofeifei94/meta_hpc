import parl
import paddle
import numpy as np
import os

class MujocoAgent(parl.Agent):
    def __init__(self, algorithm):
        super(MujocoAgent, self).__init__(algorithm)

        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def sample(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action, _ = self.alg.sample(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.alg.model.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        
    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        params_state = paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams")
        self.alg.model.set_state_dict(params_state)
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")
    def update_model(self,state_dict):
        self.alg.model.set_state_dict(state_dict)
        print("update param success")
    def update_model_from_np(self,state_dict_np):
        set_state_dict_from_np(self.alg.model.state_dict(),state_dict_np)
    def send_model_to_np(self):
        return state_dict_to_np(self.alg.model.state_dict())