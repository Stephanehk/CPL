from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

#AirRaidDeterministic-v4, AssaultNoFrameskip-v4, BeamRiderNoFrameskip-v4, PhoenixNoFrameskip-v4, DemonAttackNoFrameskip-v4
#
env_name  = "AirRaidNoFrameskip-v4"
n_envs = 8
env = make_atari_env(env_name, n_envs=n_envs,monitor_dir=env_name + "_monitor_files",env_kwargs={"full_action_space":True})
env = VecFrameStack(env, n_stack=4)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=int(100000/n_envs), #TODO: check if this is correct
  save_path="./"+env_name+"_logs/",
  name_prefix= env_name + "_expert",
)

#PPO params: https://huggingface.co/sb3/ppo-BreakoutNoFrameskip-v4
model = PPO("CnnPolicy",
            env,
            verbose=1,
            batch_size=256,
            ent_coef=0.01,
            n_epochs=4,
            n_steps=128,
            vf_coef=0.5,
            )

model.learn(total_timesteps=10000000,callback=checkpoint_callback)
obs = env.reset()
#model = A2C.load("A2C_breakout") #uncomment to load saved model
model.save("PPO_breakout")
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
