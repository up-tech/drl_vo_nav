from gym.envs.registration import register

# Register drl_nav env 
register(
  id='drl-nav-v0',
  entry_point='turtlebot_gym.envs:DRLNavEnv'
  )

register(
  id='drl-nav-v1',
  entry_point='turtlebot_gym.envs:DRLNavDojoEnv'
)

register(
  id='drl-nav-v2',
  entry_point='turtlebot_gym.envs:DRLNavDojoPlusPlusEnv'
)
