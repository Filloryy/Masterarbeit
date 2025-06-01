class ExperimentConfig:
    def __init__(self, experiment, actor, transform, terrain, video=False, total_frames=1_000_000, lr=3e-4, max_grad_norm=1,
                 frames_per_batch=1000, sub_batch_size=10, num_epochs=10,
                 clip_epsilon=0.2, gamma=0.99, lmbda=0.95, entropy_eps=1e-4):
        self.experiment = experiment
        self.actor = actor
        self.transform = transform
        self.terrain = terrain
        self.video = video
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

hetero_config = ExperimentConfig(
    experiment= "hetero",
    actor = "hetero_actor",
    transform = "heterograph",
    terrain = "flat",
    total_frames = 1_000_000,)

single_config = ExperimentConfig(
    experiment = "single_node",
    actor= "single_node_actor",
    transform = "OneNode",
    terrain = "flat",
    total_frames = 1_000_000,)

left_right_config = ExperimentConfig(
    experiment = "left_right",
    actor = "left_right_actor",
    transform = "torsoleftright",
    terrain = "flat",
    total_frames = 1_000_000,)

fully_distributed_config = ExperimentConfig(
    experiment = "fully_distributed",
    actor = "distributed_actor",
    transform = "fullbodygraph",
    terrain = "flat",
    total_frames = 1_000_000,)

mlp_config = ExperimentConfig(
    experiment = "mlp",
    actor = "mlp_actor",
    transform = "Notransform",
    terrain = "flat",
    total_frames = 1_000_000,)