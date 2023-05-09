import torch as th
import gym
from gym.wrappers import TimeLimit
import numpy as np
from pathlib import Path
from seals.util import AutoResetWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.ppo import CnnPolicy

from imitation.algorithms import preference_comparisons
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import CnnRewardNet
from imitation.util import logger as imit_logger
from imitation.scripts.train_preference_comparisons import save_checkpoint

# seed = 0
rng = np.random.default_rng(0)

teacher_temp = 0
max_queue_size = None
batch_size = 128
checkpoint_interval = 1
fragment_length = 100
n_envs = 8

n_prefs_per_round = 50
num_feedback_rounds = 1
initial_comparison_frac = 1/(num_feedback_rounds + 1)
total_comparisons = int((num_feedback_rounds*n_prefs_per_round)/(1-initial_comparison_frac))

print (total_comparisons)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def constant_length_episode(env_name, num_steps):
    atari_env = gym.make(env_name,full_action_space=True)
    preprocessed_env = AtariWrapper(atari_env)
    endless_env = AutoResetWrapper(preprocessed_env)
    limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(limited_env)

venv = make_vec_env(constant_length_episode, n_envs=n_envs, env_kwargs={"env_name":"AssaultNoFrameskip-v4","num_steps": 1000})
venv = VecFrameStack(venv, n_stack=4)
reward_net = CnnRewardNet(
    venv.observation_space,
    venv.action_space,
).to(device)

fragmenter = preference_comparisons.RandomFragmenter(
    warning_threshold=0,
    rng=rng
)
gatherer = preference_comparisons.SyntheticGatherer(temperature=teacher_temp,rng=rng)
preference_model = preference_comparisons.PreferenceModel(reward_net)
reward_trainer = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=100,
    batch_size = batch_size,
    # custom_logger = logger,
    rng=rng,
    lr=0.003
)
dataset =None

#AssaultNoFrameskip: 4,8
#AirRaidNoFrameskip: 1,9
#BeamRiderNoFrameskip: 1,3
#DemonAttackNoFrameskip: 2,10
#PhoenixNoFrameskip: 2,8

policy_fps = ["AssaultNoFrameskip-v4_logs/AssaultNoFrameskip-v4_expert_4000000_steps.zip",
              "AirRaidNoFrameskip-v4_logs/AirRaidNoFrameskip-v4_expert_1000000_steps.zip",
              "BeamRiderNoFrameskip-v4_logs/BeamRiderNoFrameskip-v4_expert_1000000_steps.zip",
              "DemonAttackNoFrameskip-v4_logs/DemonAttackNoFrameskip-v4_expert_2000000_steps.zip",
              "PhoenixNoFrameskip-v4_logs/PhoenixNoFrameskip-v4_expert_2000000_steps.zip",
              "AssaultNoFrameskip-v4_logs/AssaultNoFrameskip-v4_expert_8000000_steps.zip",
              "AirRaidNoFrameskip-v4_logs/AirRaidNoFrameskip-v4_expert_9000000_steps.zip",
              "BeamRiderNoFrameskip-v4_logs/BeamRiderNoFrameskip-v4_expert_3000000_steps.zip",
              "DemonAttackNoFrameskip-v4_logs/DemonAttackNoFrameskip-v4_expert_10000000_steps.zip",
              "PhoenixNoFrameskip-v4_logs/PhoenixNoFrameskip-v4_expert_8000000_steps.zip"]
for policy_fp in policy_fps:

    logger_fp = policy_fp.split("/")[1].split(".")[0] + "_pref_learning"
    logger = imit_logger.configure(logger_fp,["stdout", "log", "csv", "tensorboard"])

    # agent = PPO.load(policy_fp)
    env_name = policy_fp.split("_")[0]
    print ("On environment:",env_name)

    venv = make_vec_env(constant_length_episode, n_envs=n_envs, env_kwargs={"env_name":env_name,"num_steps": 1000})
    venv = VecFrameStack(venv, n_stack=4)

    agent = PPO.load(policy_fp)
    # agent = PPO("CnnPolicy",venv)


    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0, #fraction of the trajectories that will be generated partially randomly rather than only by the agent when sampling.
        custom_logger = logger,
        rng=rng,
        train_agent=False
    )

    def save_callback(iteration_num):
        if checkpoint_interval > 0 and iteration_num % checkpoint_interval == 0:
            save_checkpoint(
                trainer=pref_comparisons,
                save_path= Path(logger.default_logger.dir + "/checkpoints/" + str(iteration_num)),
                allow_save_policy=True,
            )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=num_feedback_rounds, #note this was 1 previously # number of times to train the agent against the reward model and then train the reward model against newly gathered preferences
        fragmenter=fragmenter, # takes in a set of trajectories and returns pairs of fragments for which preferences will be gathered.
        preference_gatherer=gatherer, # how to get preferences between trajectory fragments. Default (and currently the only option) is to use synthetic preferences based on ground-truth rewards.
        reward_trainer=reward_trainer,
        comparison_queue_size = max_queue_size, #the maximum number of comparisons to keep in the queue for training the reward model. If None, the queue will grow without bound as new comparisons are added.
        fragment_length=fragment_length,
        transition_oversampling=1,
        initial_comparison_frac=initial_comparison_frac, #fraction of the total_comparisons argument to train() that will be sampled before the rest of training begins (using a randomly initialized agent).
        initial_epoch_multiplier=1,
        query_schedule = "constant", #representing a fraction of the total number of timesteps elapsed up to some time T, and returns a potentially unnormalized probability indicating the fraction of total_comparisons that should be queried at that iteration.
        custom_logger = logger,
        dataset = dataset,
    )
    pref_comparisons.train(
        total_timesteps=1000,  # For good performance this should be 1_000_000
        total_comparisons=total_comparisons,  # For good performance this should be 5_000
        callback=save_callback
    )
    dataset = pref_comparisons.dataset
