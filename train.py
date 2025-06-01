import torch
from torch import nn

from tensordict.nn import TensorDictModule
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
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger, VideoRecorder
from torchrl.trainers import OptimizerHook, LogReward, Trainer

from sim_environment.hubert import QuantrupedEnv
from hooks import (
    CustomProcessBatchHook,
    CustomProcessOptimBatchHook,
    LearningRateSchedulerHook,
    CumulativeLoggingHook,
    hfield_update_hook,
)
from customtransform import OneNode, torsoleftright, fullbodygraph, heterograph, Notransform
import actors
from config import (
    hetero_config,
    single_config,
    left_right_config,
    fully_distributed_config,
    mlp_config,
)

def run_experiment(config):
    device = torch.device("cpu")
    terrain = config.terrain
    experiment = config.experiment
    lr = config.lr
    max_grad_norm = config.max_grad_norm
    frames_per_batch = config.frames_per_batch
    total_frames = config.total_frames
    sub_batch_size = config.sub_batch_size
    num_epochs = config.num_epochs
    clip_epsilon = config.clip_epsilon
    gamma = config.gamma
    lmbda = config.lmbda
    entropy_eps = config.entropy_eps
    file_path = config.file_path
    video = config.video

    # Select transform dynamically
    transform_map = {
        "heterograph": heterograph,
        "OneNode": OneNode,
        "torsoleftright": torsoleftright,
        "fullbodygraph": fullbodygraph,
        "Notransform": Notransform,
    }

    actor_map = {
        "hetero_actor": actors.hetero_actor,
        "single_node_actor": actors.single_node_actor,
        "distributed_actor": actors.distributed_actor,
        "left_right_actor": actors.left_right_actor,
        "mlp_actor": lambda: actors.mlp_actor(num_cells=256, action_dim=8, device=device),
    }

    selected_transform = transform_map.get(config.transform)
    if selected_transform is None and config.transform is not None:
        raise ValueError(f"Unknown transform: {config.transform}")

    actor_fn = actor_map.get(config.actor)
    if actor_fn is None:
        raise ValueError(f"Unknown actor: {config.actor}")
    actor_net = actor_fn().to(device)

    #working environment
    base_env = GymEnv("hubert") #hubert is a slightly changed ant-v4 environment

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            selected_transform(in_keys=["observation"], out_keys=["graph"]),
            StepCounter(),
        ),
    )

    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0) #initial stats for observation normalization

    #record environment
    path = "Logs"
    logger = CSVLogger(exp_name=experiment, log_dir=path, video_format="mp4")
    video_recorder = VideoRecorder(logger, tag="video")
    record_env = TransformedEnv(
        GymEnv("hubert", from_pixels=True, pixels_only=False),
                            Compose(
                                ObservationNorm(in_keys=["observation"]),
                                DoubleToFloat(),
                                selected_transform(in_keys=["observation"], out_keys=["graph"]),
                                StepCounter(),
                                video_recorder,
                            )
    )
    record_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    policy_in_keys = ["graph"] if selected_transform is not Notransform else ["observation"]
    policy_module = TensorDictModule(
        actor_net, in_keys=policy_in_keys, out_keys=["loc", "scale"]
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
        nn.LazyLinear(256, device=device),
        nn.Tanh(),
        nn.LazyLinear(256, device=device),
        nn.Tanh(),
        nn.LazyLinear(256, device=device),
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
    hfield_updater= hfield_update_hook(env)

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
        save_trainer_interval=1000000,
        log_interval=10000,
        save_trainer_file=f"{file_path}/trainer.pt",
    )
    #registering hooks, defines the training loop for ppo learning, adapted from torchrl tutorial
    trainer.register_op("batch_process", process_batch_hook)
    trainer.register_op("process_optim_batch", process_optim_batch_hook)
    trainer.register_op("optimizer", optimizerHook)
    trainer.register_op("pre_steps_log", log_reward)
    trainer.register_op("post_steps", lrscheduler_hook)
    trainer.register_op("post_steps_log", cum_reward)
    if terrain == "hills": trainer.register_op("post_steps", hfield_updater)
    
    if video:
        trainer.load_from_file(f"{file_path}/trainer.pt")

        #rendering video
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            video_rollout = record_env.rollout(1000, policy_module)
            video_recorder.dump()
            del video_rollout
    else:
        trainer.train()

experiment_list = [hetero_config, single_config, left_right_config, fully_distributed_config, mlp_config]
for cfg in experiment_list:
    run_experiment(cfg)