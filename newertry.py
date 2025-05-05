import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.record import CSVLogger, VideoRecorder
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    #ObsToGraph,
    check_env_specs,
)
from torchrl.trainers import OptimizerHook, LogReward
from torchrl.trainers import Trainer
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from sim_environment.hubert import QuantrupedEnv
from hooks import CustomProcessBatchHook, CustomProcessOptimBatchHook, LearningRateSchedulerHook, CumulativeLoggingHook
from torch_geometric.nn import Linear, Sequential, GCNConv, GraphConv, global_mean_pool, avg_pool
from torch_geometric.data import Data, Batch
from customtransform import ObsToGraph, OneNode, torsoleftright, fullbodygraph
"""
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
"""

device = torch.device("cpu")  #using CPU until we use larger matrices

num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000

total_frames = 5_000_000

sub_batch_size = 10 # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss
)
#parameters are chosen as in Nervenet, works for NN.
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

#working environment
base_env = GymEnv("hubert") #hubert is a slightly changed ant-v4 environment

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        fullbodygraph(in_keys=["observation"], out_keys=["graph"]),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0) #initial stats for observation normalization


#record environment
path = "./Quant4/training_loop"
logger = CSVLogger(exp_name="PPO", log_dir=path, video_format="mp4")
video_recorder = VideoRecorder(logger, tag="video")
"""
record_env = TransformedEnv(
    GymEnv("hubert", from_pixels=True, pixels_only=False),
                          Compose(
                            ObservationNorm(in_keys=["observation"]),
                            DoubleToFloat(),
                            OneNode(in_keys=["observation"], out_keys=["graph"]),
                            StepCounter(),
                            video_recorder,
                        )
)
record_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
"""
class single_node_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index, batch", [
            (Linear(27, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (Linear(64, 16), "x -> x"),
            nn.Tanh(),
            NormalParamExtractor(),
        ])


    #this is an example forward function, tried with different but somewhat similar functions    
    def forward(self, data):
        #propagation model does not take a list so we unpack the list in a batch as explained in torch_geometric
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            loc, scale = self.propagation_model(batch.x, batch.edge_index)  
        else:
            loc, scale = self.propagation_model(data.x, data.edge_index)
            loc = loc.t().squeeze(-1)
            scale = scale.t().squeeze(-1)
        return loc, scale

class torso_left_right_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index, batch", [
            (Linear(11, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (Linear(64, 16), "x -> x"),
            nn.Tanh(),
            (global_mean_pool, "x, batch -> x"),
            NormalParamExtractor(),
        ])


    def forward(self, data):
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            loc, scale = self.propagation_model(batch.x, batch.edge_index, batch.batch)  
        else:
            batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            loc, scale = self.propagation_model(data.x, data.edge_index, batch_vec)
            loc = loc.t().squeeze(-1)
            scale = scale.t().squeeze(-1)
        return loc, scale


actor_net = torso_left_right_actor().to(device)#comment this and out uncomment following actor_net line to switch to "normal" NN, change in policy module in_key from graph to observation (line 143)


"""
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)
"""

policy_module = TensorDictModule(
    actor_net, in_keys=["graph"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)


collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

#init dummy batch, needed for LazyLinear
observation_shape = (27,)
dummy_batch = torch.zeros((1, *observation_shape), device=device, dtype=torch.float32)
with torch.no_grad():
    value_module(dummy_batch)



advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

#initializing optimizers
optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)
#initializing hooks
process_batch_hook = CustomProcessBatchHook(advantage_module, replay_buffer, sub_batch_size, device)
process_optim_batch_hook = CustomProcessOptimBatchHook(replay_buffer, sub_batch_size, device)
optimizerHook = OptimizerHook(optimizer=optim, loss_components=["loss_objective", "loss_critic", "loss_entropy"])
log_reward = LogReward(logname="r_training" , log_pbar=True, reward_key=("next", "reward"))
lrscheduler_hook = LearningRateSchedulerHook(scheduler)
cum_reward = CumulativeLoggingHook(logname="Cumulative Reward", env=env, policy_module=policy_module)

#setting up trainer
trainer = Trainer(
    collector=collector,
    total_frames=total_frames,
    frame_skip=1,
    loss_module=loss_module,
    logger=logger,
    optim_steps_per_batch=num_epochs,
    clip_grad_norm=True,
    clip_norm=max_grad_norm,
    progress_bar=True,
    save_trainer_interval=10000,
    log_interval=10000,
    save_trainer_file="/home/joshua/Desktop/WorkBase/Quant4/training_loop/GNN_PPO/Trainer/trainer.pt",
)
#registering hooks, defines the training loop for ppo learning, adapted from torchrl tutorial
trainer.register_op("batch_process", process_batch_hook)
trainer.register_op("process_optim_batch", process_optim_batch_hook)
trainer.register_op("optimizer", optimizerHook)
trainer.register_op("pre_steps_log", log_reward)
trainer.register_op("post_steps", lrscheduler_hook)
trainer.register_op("post_steps_log", cum_reward)


trainer.train()
#trainer.load_from_file("/home/joshua/Desktop/WorkBase/Quant4/training_loop/GNN_PPO/Trainer/trainer.pt")

#rendering video
"""
with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
    video_rollout = record_env.rollout(1000, policy_module)
    video_recorder.dump()
    del video_rollout
"""
