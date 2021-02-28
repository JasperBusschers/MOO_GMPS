import collections
from rllab import spaces
from rllab.envs.base import Env
from gym.utils import seeding

import numpy as np
from rllab.core.serializable import Serializable

_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class dsth(Serializable, Env):
    def __init__(self, num_goals=40, train=True,width=10, *args, **kwargs):
        thetas = np.linspace(0, np.pi / 2, 40)
        self.goals = [[124,-19],[1,-1],[2,-3], [3,-5] , [5,-7] ,[8,-8], [16,-9] , [24,-13], [50,-14], [74,-17] ]
        self.goal = [124,-19]
        self.num_goals = num_goals
        self.sparse = False
        self.info_logKeys = ['goal_dist']
        super(dsth, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.shape = (width + 1, width)
        self.start_state_index = 0

        nS = np.prod(self.shape)
        nA = 4


        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (0, 0)
        self.steps = 0
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA
        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)
        self.total_reward = np.zeros([2])

    def get_current_obs(self):
        return np.zeros([1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def viewer_setup(self):
        pass

    def sample_goals(self, num_goals):
        return self.goals[np.arange(num_goals)]


    def reset(self, init_state=None, reset_args=None, **kwargs):
        print(reset_args)
        goal_idx = reset_args
        self.steps = 0
        if goal_idx is not None:
            self.goal = self.goals[goal_idx]
        elif self.goal is None:
            self.goal = self.goals[0]
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        obs = self.get_current_obs()
        print("Achieved reward after episode : " +str(self.total_reward))
        self.total_reward = np.zeros([2])
        return self.s



    def step(self, action):
        action = int(np.round(action))
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = action
        r_= r
        self.total_reward += r
        r = self.chebychev()#0.99*r[0]+0.01*r[1]


        self.steps+=1
        return (s, float(r), d, {"prob": p})

    def chebychev(self):
        return - np.max([np.abs(self.total_reward[0] - self.goal[0]),
                          np.abs(self.total_reward[1] - self.goal[1])])

    def log_diagnostics(self, paths, prefix='', logger=None):
        pass
        from rllab.misc import logger
        #if type(paths[0]) == dict:
            #for key in self.info_logKeys:
                #logger.record_tabular(prefix + 'last_' + key, np.mean([path['env_infos'][key][-1] for path in paths]))

        #else:
        #    raise NotImplementedError

    def _treasures(self):

        if self.shape[1] > 10:
            raise ValueError('Default Deep Sea Treasure only supports a grid-size of max 10')

        return {(1, 0): 1,
                (2, 1): 2,
                (3, 2): 3,
                (4, 3): 5,
                (4, 4): 8,
                (4, 5): 16,
                (7, 6): 24,
                (7, 7): 50,
                (9, 8): 74,
                (10, 9): 124}

    def _unreachable_positions(self):
        u = []
        treasures = self._treasures()
        for p in treasures.keys():
            for i in range(p[0]+1, self.shape[0]):
                u.append((i, p[1]))
        return u

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):

        unreachable = self._unreachable_positions()
        treasures = self._treasures()
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_position = tuple(new_position)
        if new_position in unreachable:
            new_position = tuple(current)
        new_state = np.ravel_multi_index(new_position, self.shape)

        if new_position in treasures:
            reward = [treasures[new_position], -1]
            done = True
        else:
            reward = [0, -1]
            done = False
        return [(1., new_state, np.array(reward), done)]



    @property
    def action_space(self):
        init = np.zeros([0]).shape
        ub = np.array([3])
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