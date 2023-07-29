from gym_unity.envs import UnityToGymWrapper
from env.animalai_env.animalai.envs.environment import AnimalAIEnvironment

class Animal():
    def __init__(self,play=False,config_file="") -> None:
        print("start env")
        aai_env = AnimalAIEnvironment(
            additional_args=[] if play else ["-batchmode"],
            seed = 123,
            file_name="env/animalai_env/env/AAI_v3.0.1_build_linux_090422.x86_64",
            arenas_configurations=config_file,
            play=play,
            base_port=5000,
            inference=False,
            useCamera=True,
            resolution=64,
            useRayCasts=False,
            # raysPerSide=1,
            # rayMaxDegrees = 30,
            # no_graphics=True,
        )
        print("env start success")
        env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
        # print("action_size",env.action_space)
        # input()
        self.env=env
        self.max_ep_steps=500
    def reset(self):
        self._ep_steps=0
        return self._inner_reset()
    def _inner_reset(self):
        self._ep_steps=0
        return self.env.reset()
    def step(self,action):
        """
        action
        0不动 1左转 2右转
        3前进 4前进+左转 5前进+右转
        6后退 7后退+左转 8后退+右转
        """
        obs,reward,done,info=self.env.step(action)
        self._ep_steps+=1
        if done or self._ep_steps>=self.max_ep_steps:
            obs=self._inner_reset()

        return obs,reward,done,info
class AnimalPlay():
    def __init__(self) -> None:
        aai_env = AnimalAIEnvironment(
            additional_args=[
            # "-batchmode",
            ],
            seed = 123,
            file_name="env/animalai_env/env/AAI_v3.0.1_build_linux_090422.x86_64",
            arenas_configurations="",
            play=False,
            base_port=5000,
            inference=False,
            useCamera=True,
            resolution=256,
            useRayCasts=False,
            # raysPerSide=1,
            # rayMaxDegrees = 30,
            # no_graphics=True,
        )
        env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
        # print("action_size",env.action_space)
        # input()
        self.env=env
    def reset(self):
        return self._inner_reset()
    def _inner_reset(self):
        return self.env.reset()
    def step(self,action):
        obs,reward,done,info=self.env.step(action)
        if done:
            obs=self._inner_reset()
        return obs,reward,done,info
if __name__=="__main__":
    env=Animal()
    env.reset()
    while 1:
        env.step(1)
