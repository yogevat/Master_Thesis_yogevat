import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from Envronment_RL_Push import RobotModelEnv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            self.save_path = os.path.join(log_dir, "best_model"+str(self.n_calls))
            self.model.save(self.save_path)

        return True


log_dir = r"path\to\log\dir"
env = RobotModelEnv(action_type='discrete')
env = Monitor(env, log_dir)
print("Action space:", env.action_space)
print("Type:", type(env.action_space))
print("create a new model")
policy_kwargs = dict(net_arch=[512, 512, 512, 512])
model = PPO(policy="MlpPolicy", env=env, learning_rate=0.001, verbose=True, batch_size=512,
            policy_kwargs=policy_kwargs)
####################### uncomment to learn the agent ######################
# checkpoint_callback = CheckpointCallback(save_freq=500, save_path=log_dir)
# callback_best_reward = SaveOnBestTrainingRewardCallback(check_freq=250, log_dir=log_dir)
# callback = CallbackList([checkpoint_callback, callback_best_reward])
# model.learn(total_timesteps=1e6, callback=callback, progress_bar=True)
# checkpoint_callback = CheckpointCallback(save_freq=500, save_path=log_dir)
# callback_best_reward = SaveOnBestTrainingRewardCallback(check_freq=250, log_dir=log_dir)
# callback = CallbackList([checkpoint_callback, callback_best_reward])
# model.learn(total_timesteps=1e6, callback=callback, progress_bar=True)

####################### load learnd Agent ######################
path = r"path\to\the\zip\of\the\train\agent\Agent_YYY.zip"
loaded_model = PPO.load(path=path, env=env)
loaded_model1 = PPO.load(path=path, env=env)
loaded_model2 = PPO.load(path=path, env=env)
loaded_model3 = PPO.load(path=path, env=env)

####################### Finish ######################

