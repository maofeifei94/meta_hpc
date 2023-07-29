import gym
import argparse
import numpy as np
from parl.utils import  ReplayMemory
from parl.env.continuous_wrappers import ActionMappingWrapper
from pfc.ac_model.sac_model import MujocoModel
from pfc.agent.sac_agent import MujocoAgent
from parl.algorithms import SAC

model_num=1460535
mode='test'

WARMUP_STEPS = 1e4
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4


# Run episode for training
def run_train_episode(agent, env):
    action_dim = env.action_space.shape[0]
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    while not done:
        episode_steps += 1
        # Select action randomly or according to policy

        action = agent.sample(obs)

        # Perform action
        next_obs, reward, done, _ = env.step(action)
        env.render()
        terminal = float(done) if episode_steps < env._max_episode_steps else 0

        # Store data in replay memory
        # rpm.append(obs, action, reward, next_obs, terminal)

        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data


    return episode_reward, episode_steps


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            env.render()
    avg_reward /= eval_episodes
    return avg_reward


def main():
    # logger.info("------------------- SAC ---------------------")
    # logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    # logger.info("---------------------------------------------")
    # logger.set_dir('./{}_{}'.format(args.env, args.seed))

    env = gym.make(args.env)
    env.seed(args.seed)
    env = ActionMappingWrapper(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent, replay_memory
    model = MujocoModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=args.alpha,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm)
    agent.load_model("all_models/sac_model",model_num)
    while 1:
        if mode=='train':
            print("train")
            episode_reward, episode_steps = run_train_episode(agent, env)
        else:
            print("test")
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
        # episode_reward, episode_steps = run_train_episode(agent, env)
        

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="BipedalWalkerHardcore-v3", help='Mujoco gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym seed')
    parser.add_argument(
        "--train_total_steps",
        default=3e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')
    parser.add_argument(
        "--alpha",
        default=0.2,
        type=float,
        help=
        'Determines the relative importance of entropy term against the reward'
    )
    args = parser.parse_args()

    main()