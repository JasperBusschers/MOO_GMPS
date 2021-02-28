import collections
from rllab import spaces
from rllab.envs.base import Env

import numpy as np
from rllab.core.serializable import Serializable

_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])

class continious_test(Serializable, Env):
    def __init__(self, num_goals=40, train=True, *args, **kwargs):
        thetas = np.linspace(0, np.pi / 2, 40)
        self.goals = np.array([[2 * np.cos(theta), 2 * np.sin(theta)] for theta in thetas])
        self.goal = None
        self.num_goals = num_goals
        self.sparse = False
        self.info_logKeys = ['goal_dist']
        super(continious_test, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)


    def get_current_obs(self):
        return np.zeros([1])

    def viewer_setup(self):
        pass

    def sample_goals(self, num_goals):
        return self.goals[np.arange(num_goals)]


    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_idx = reset_args
        if goal_idx is not None:
            self.goal = self.goals[goal_idx]
        elif self.goal is None:
            self.goal = self.goals[0]

        obs = self.get_current_obs()
        return obs



    def step(self, action):
        reward = 100#goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.get_current_obs() #self._state
        notdone = False
        done = not notdone
        ob = self.get_current_obs()
        infos = {'goal_dist': np.linalg.norm(reward- self.goal)}
        return _Step(ob, float(reward), done, infos)


    def log_diagnostics(self, paths, prefix='', logger=None):
        pass
        from rllab.misc import logger
        #if type(paths[0]) == dict:
            #for key in self.info_logKeys:
                #logger.record_tabular(prefix + 'last_' + key, np.mean([path['env_infos'][key][-1] for path in paths]))

        #else:
        #    raise NotImplementedError

    @property
    def action_space(self):
        init = np.zeros([1]).shape
        ub = np.array([5])
        lb = np.zeros_like(ub)
        return spaces.Box(lb, ub)

    @property
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = 1e6 * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def action_bounds(self):
        return self.action_space.bounds