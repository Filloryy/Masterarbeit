from gymnasium.envs.registration import register
from sim_environment.hubert import QuantrupedEnv



# Register Gym environment.
register(
    id='hubert',
    entry_point='sim_environment.hubert:QuantrupedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,

)

print("Base environments registered!")