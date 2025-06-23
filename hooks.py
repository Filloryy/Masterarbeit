import numpy as np
import torch
from torchrl.trainers import TrainerHookBase, OptimizerHook, LogReward
from torchrl.envs.utils import ExplorationType, set_exploration_type
import logging #??
from collections import defaultdict
from typing import Union, Dict
from sim_environment.hubert import create_new_hfield



class CustomProcessBatchHook(TrainerHookBase):
    def __init__(self, advantage_module, replay_buffer, sub_batch_size, device):
        self.advantage_module = advantage_module
        self.replay_buffer = replay_buffer
        self.sub_batch_size = sub_batch_size
        self.device = device

    def __call__(self, batch):
        # Compute advantages
        self.advantage_module(batch)
        # Flatten the batch and extend the replay buffer
        data_view = batch.reshape(-1)
        self.replay_buffer.extend(data_view.cpu())
        return batch
    
class CustomProcessOptimBatchHook(TrainerHookBase):
    def __init__(self, replay_buffer, sub_batch_size, device):
        self.replay_buffer = replay_buffer
        self.sub_batch_size = sub_batch_size
        self.device = device

    def __call__(self, batch):
        # Sample a sub-batch
        sub_batch = self.replay_buffer.sample(self.sub_batch_size)
        return sub_batch
    
class LearningRateSchedulerHook(TrainerHookBase):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self):
        self.scheduler.step()
    
    def register(self, trainer, name="scheduler"):
        trainer.register_op("post_steps", self)
        trainer.register_module(name, self)

class CumulativeLoggingHook(TrainerHookBase):
    def __init__(self, logname, env, policy_module):
        self.logname = logname
        self.counter = 0
        self.env = env
        self.policy_module = policy_module

    def __call__(self, batch):
        if self.counter % 10 == 0:
            self.counter += 1
            logs = defaultdict(list)
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = self.env.rollout(1000, self.policy_module)
                out = eval_rollout["next", "reward"].sum().item()
            return {self.logname: out
            }
        else:
            self.counter += 1
            return None
        
    def register(self, trainer, name):
        trainer.register_module(name, self)
        trainer.register_op("post_steps_log", self)

#logging the weights
class WeightWatcherHook(TrainerHookBase):
    def __init__(self, module):
        self.module = module
        self.counter = 0
        self.log_interval = 10
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("WeightLogger")

    def __call__(self, batch):
        if self.counter % self.log_interval == 0:
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    self.logger.info(f"Step {self.counter}: {name} - {param.data}")
        self.counter += 1

    def register(self, trainer, name="weight_logger"):
        trainer.register_op("post_steps", self)
        trainer.register_module(name, self)

class VideoRecorderHook(TrainerHookBase):
    def __init__(self, env, policy_module, file_path, interval=100):
        self.env = env
        self.policy_module = policy_module
        self.file_path = file_path
        self.interval = interval
        self.counter = 0

    def __call__(self, *args, **kwargs):
        if self.counter % self.interval == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                rollout = self.env.rollout(1000, self.policy_module, auto_reset=True, break_when_any_done=True)
                if "action" in rollout:
                    actions = rollout["action"].cpu().numpy()
                    np.savetxt(f"{self.file_path}/video_actions_step{self.counter}.txt", actions, fmt="%.6f", delimiter=",")
                    print(f"Actions saved to {self.file_path}/video_actions_step{self.counter}.txt")
                if "observation" in rollout:
                    observations = rollout["observation"].cpu().numpy()
                    np.savetxt(f"{self.file_path}/video_observations_step{self.counter}.txt", observations, fmt="%.6f", delimiter=",")
                    print(f"Observations saved to {self.file_path}/video_observations_step{self.counter}.txt")
                if "terminated" in rollout:
                    terminated = rollout["terminated"].cpu().numpy()
                    np.savetxt(f"{self.file_path}/video_terminated_step{self.counter}.txt", terminated, fmt="%d", delimiter=",")
                    print(f"Terminated states saved to {self.file_path}/video_terminated_step{self.counter}.txt")
                if hasattr(self.env, 'transform') and hasattr(self.env.transform[-1], 'dump'):
                    self.env.transform[-1].dump()
        self.counter += 1

class hfield_update_hook(TrainerHookBase):
    def __init__(self, env, smoothness_range=(0.1, 1)):
        self.env = env
        self.smoothness_range = smoothness_range

    def __call__(self, *args, **kwargs):
        # Randomize smoothness each time
        new_smoothness = np.random.uniform(*self.smoothness_range)
        self.env.unwrapped.update_hfield(smoothness=new_smoothness)