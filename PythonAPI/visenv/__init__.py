from gym.envs.registration import register
from visenv.base import VisualWorldEnv

register(
    id='VisualWorldEnv-v0',
    entry_point='visenv:VisualWorldEnv',
    max_episode_steps = 100,
    kwargs={
        'goal_visible':False
    }
)