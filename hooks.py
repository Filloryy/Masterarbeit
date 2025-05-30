import torch
from torchrl.trainers import TrainerHookBase, OptimizerHook, LogReward
from torchrl.envs.utils import ExplorationType, set_exploration_type
import logging #??
from collections import defaultdict
from typing import Union, Dict
from sim_environment.hubert import create_new_hfield
import random

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

class videohook(TrainerHookBase):
    def __init__(self, trainer, save_trainer_interval, base_path):
        self.save_trainer_interval = save_trainer_interval
        self.base_path = base_path
        self.last_save = -1
        self.trainer = trainer

    def __call__(self):
        if (self.trainer.collected_frames - self.last_save) > self.save_trainer_interval:
            self.last_save = self.trainer.collected_frames
            torch.save(self.state_dict(), self.base_path + self.trainer.collected_frames + ".pt")

    def register(self, trainer, name="trainer_saver"):
        trainer.register_op("post_steps", self)
        trainer.register_module(name, self)

class hfield_update_hook(TrainerHookBase):
    def __init__(self, env):
        self.env = env

    def __call__(self):
        smoothness = random.uniform(0, 1)
        create_new_hfield(self.env.unwrapped.model, smoothness)