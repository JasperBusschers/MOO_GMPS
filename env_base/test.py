from env_base.continious import continious_test
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

from rllab.envs.mujoco.ant_env_rand_goal_ring import AntEnvRandGoalRing

env = continious_test()
print(env.spec._observation_space)
print(env.spec.action_space)