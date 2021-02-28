import collections
from rllab import spaces
from rllab.envs.base import Env
from gym.utils import seeding
import numpy as np
import os.path as osp

from rllab import spaces
from rllab.envs.base import Env
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc import logger
from gym.utils import seeding

import theano
import tempfile
import os
import mako.template
import mako.lookup
import numpy as np
from rllab.core.serializable import Serializable

_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])

from rllab import spaces
from rllab.core.serializable import Serializable


class DamEnv(Serializable,Env):
    """ A Water reservoir environment.
        The agent executes a continuous action, corresponding to the amount of water
        released by the dam.
        There are up to 4 rewards:
         - cost due to excess level wrt a flooding threshold (upstream)
         - deficit in the water supply wrt the water demand
         - deficit in hydroelectric supply wrt hydroelectric demand
         - cost due to excess level wrt a flooding threshold (downstream)

         Code from:
         https://gitlab.ai.vub.ac.be/mreymond/dam
         Ported from:
         https://github.com/sparisi/mips
    """

    S = 1.0 # Reservoir surface
    W_IRR = 50. # Water demand
    H_FLO_U = 50. # Flooding threshold (upstream, i.e. height of dam)
    S_MIN_REL = 100. # Release threshold (i.e. max capacity)
    DAM_INFLOW_MEAN = 40. # Random inflow (e.g. rain)
    DAM_INFLOW_STD = 0.#10.
    Q_MEF = 0.
    GAMMA_H2O = 1000. # water density
    W_HYD = 4.36 # Hydroelectric demand
    Q_FLO_D = 30. # Flooding threshold (downstream, i.e. releasing too much water)
    ETA = 1. # Turbine efficiency
    G = 9.81 # Gravity

    utopia = {2: [-.5, -9], 3: [-.5, -9, -0.0001], 4: [-0.5, -9, -0.001, -9]}
    antiutopia = {2: [-2.5, -11], 3: [-65, -12, -0.7], 4: [-65, -12, -0.7, -12]}
    s_init = [9.6855361e+01,
              5.8046026e+01,
              1.1615767e+02,
              2.0164311e+01,
              7.9191000e+01,
              1.4013098e+02,
              1.3101816e+02,
              4.4351321e+01,
              1.3185943e+01,
              7.3508622e+01,]

    s_init = [7.9191000e+01]
    
    def __init__(self,w1=0.5, nO=2, penalize=False, *args, **kwargs):
        super(DamEnv).__init__()
        self.tasks =[0.05,0.25,0.64,0.73,0.99] #[0.25,0.3,0.4,0.5,0.6,0.8,0.85]#[i/100 for i in range(0,100,5)]#[0.15,0.2,0.85]#[0.15,0.2,0.3,0.45,0.85]#[0.15,0.2,0.25,0.3,0.35,0.45,0.5, 0.6, 0.85]#[i/100 for i in range(0,100,5)]
        self.w1 = w1
        self.nO = nO
        self.penalize = penalize
        self.curr_step = 0
        low = 0#-np.ones(nO)*np.inf # DamEnv.antiutopia[nO]
        high = 1# np.zeros(nO) # DamEnv.utopia[nO]
        self.info_logKeys = ['goal_dist']
        super(DamEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def reset(self, init_state=None, reset_args=None, **kwargs):
        if reset_args is not None:
            self.w1 = self.tasks[reset_args]
        if not self.penalize:
            state = np.random.choice(DamEnv.s_init, size=1)
        else:
            state = np.random.randint(0, 160, size=1)
        self.curr_step=0
        self.state = state
        return self.state

    def step(self, a):
        action = np.copy(a)[0]*80#]*20+40
        # bound the action
        actionLB = np.clip(self.state - DamEnv.S_MIN_REL, 0, None)
        actionUB = self.state
        done = False
        self.curr_step+=1
        if self.curr_step >= 10:
            done = True
        # Penalty proportional to the violation
        bounded_action = np.clip(action, actionLB, actionUB)
        penalty = 0#-self.penalize*np.abs(bounded_action - action)

        # transition dynamic
        action = bounded_action
        dam_inflow = np.random.normal(DamEnv.DAM_INFLOW_MEAN, DamEnv.DAM_INFLOW_STD, len(self.state))
        # small chance dam_inflow < 0
        n_state = np.clip(self.state + dam_inflow - action, 0, None)

        # cost due to excess level wrt a flooding threshold (upstream)
        r0 = -np.clip(n_state/DamEnv.S - DamEnv.H_FLO_U, 0, None) + penalty
        # deficit in the water supply wrt the water demand
        r1 = -np.clip(DamEnv.W_IRR - action, 0, None) + penalty
        
        q = np.clip(action - DamEnv.Q_MEF, 0, None)
        p_hyd = DamEnv.ETA * DamEnv.G * DamEnv.GAMMA_H2O * n_state / DamEnv.S * q / 3.6e6

        # deficit in hydroelectric supply wrt hydroelectric demand
        r2 = -np.clip(DamEnv.W_HYD - p_hyd, 0, None) + penalty
        # cost due to excess level wrt a flooding threshold (downstream)
        r3 = -np.clip(action - DamEnv.Q_FLO_D, 0, None) + penalty

        reward = np.array([r0, r1, r2, r3], dtype=np.float32)[:self.nO].flatten()
        reward = self.w1 * r0/3 + (1-self.w1)*r1
        self.state = n_state
        return n_state, reward[0], done, {"r0":r0 , "r1" : r1}

    @property
    @overrides
    def action_space(self):
        bounds = np.array([[0.0,1.0]])
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb, ub)

    @property
    @overrides
    def observation_space(self):
        ub = np.zeros([1])
        return spaces.Box(ub , (ub+80))

    @overrides
    @property
    def action_bounds(self):
        return self.action_space.bounds

    def log_diagnostics(self, paths, prefix='', logger=None):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def viewer_setup(self):
        pass

    def sample_goals(self, num_goals):
        return self.goals[np.arange(num_goals)]