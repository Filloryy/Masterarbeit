class ExperimentConfig:
    def __init__(self, experiment, actor, transform, terrain, live_recording=False, video=False, contact_forces=False, total_frames=1_000_000, lr=3e-4, max_grad_norm=1,
                 frames_per_batch=1000, sub_batch_size=10, num_epochs=10,
                 clip_epsilon=0.2, gamma=0.99, lmbda=0.95, entropy_eps=1e-4):
        self.experiment = experiment
        self.actor = actor
        self.transform = transform
        self.terrain = terrain
        self.video = video
        self.contact_forces = contact_forces
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.file_path = f"Logs/{experiment}"
        self.live_recording = live_recording

hetero_config_old = ExperimentConfig(
    experiment= "hetero_old",
    actor = "hetero_actor",
    transform = "heterograph_old",
    terrain = "flat",
    total_frames = 2_000_000)

hetero_config = ExperimentConfig(
    experiment= "hetero",
    actor = "hetero_actor",
    transform = "heterograph",
    terrain = "flat",
    total_frames = 2_000_000)

hetero_full_info_config = ExperimentConfig(
    experiment= "hetero_full_info",
    actor = "hetero_full_info_actor",
    transform = "heterograph_full_info",
    terrain = "flat",
    total_frames = 2_000_000,
    live_recording=True)

contact_config = ExperimentConfig(
    experiment = "contact",
    actor = "contact_actor",
    transform = "full_contact",
    terrain = "flat",
    total_frames = 2_000_000,
    live_recording=False,
    contact_forces=True)

single_config = ExperimentConfig(
    experiment = "single_node",
    actor= "single_node_actor",
    transform = "OneNode",
    terrain = "flat",
    total_frames = 2_000_000)

left_right_config = ExperimentConfig(
    experiment = "left_right",
    actor = "left_right_actor",
    transform = "torsoleftright",
    terrain = "flat",
    total_frames = 2_000_000)

fully_distributed_config = ExperimentConfig(
    experiment = "fully_distributed",
    actor = "distributed_actor",
    transform = "fullbodygraph",
    terrain = "flat",
    total_frames = 2_000_000)

no_batching_config = ExperimentConfig(
    experiment = "no_batching",
    actor = "no_batching_actor",
    transform = "fullbodygraph",
    terrain = "flat",
    total_frames = 2_000_000)

mlp_config = ExperimentConfig(
    experiment = "mlp",
    actor = "mlp_actor",
    transform = "Notransform",
    terrain = "flat",
    total_frames = 2_000_000)

investigactor_config = ExperimentConfig(
    experiment = "investigactor",
    actor = "investigactor",
    transform = "Notransform",
    terrain = "flat",
    total_frames = 10_000,
    video=True)