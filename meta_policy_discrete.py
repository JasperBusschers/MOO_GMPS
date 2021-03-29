from env_base.dsth import dsth
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.maml_npo import MAMLNPO
from sandbox.rocky.tf.algos.maml_vpg import MAMLVPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaptivestep_biastransform import MAMLGaussianMLPPolicy as fullAda_Bias_policy
import matplotlib.pyplot as plt
from sandbox.rocky.tf.algos.vpg import VPG as vpg_basic, VPG
from sandbox.rocky.tf.algos.vpg_biasADA import VPG as vpg_biasADA
from sandbox.rocky.tf.algos.vpg_fullADA import VPG as vpg_fullADA
from sandbox.rocky.tf.algos.vpg_conv import VPG as vpg_conv
from rllab.algos import ppo
# from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaptivestep_biastransform import MAMLGaussianMLPPolicy as fullAda_Bias_policy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_biasonlyadaptivestep_biastransform import MAMLGaussianMLPPolicy as biasAda_Bias_policy



#from rllab.envs.mujoco.ant_env_rand_goal_ring import AntEnvRandGoalRing
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

import pickle
import argparse
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf
import joblib

def experiment(variant):

    log_dir = variant['log_dir']
    init_file = variant['init_file'] ; taskIndex = variant['taskIndex']
    n_itr = variant['n_itr'] ; default_step = variant['default_step']
    max_path_length = variant['max_path_length']
    env = TfEnv(normalize(dsth(task=taskIndex)))

    # baseline = LinearFeatureBaseline(env_spec = env.spec)
    batch_size = variant['batch_size']
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    use_meta = True
    if use_meta:

        algo = MAMLNPO(
            env=env,
            use_maml=False,
            policy=None,
            load_policy=init_file,
            baseline=baseline,
            batch_size=batch_size,  # 2x
            max_path_length=max_path_length,
            n_itr=n_itr,
            # noise_opt = True,
            default_step=default_step,
            sampler_cls=VectorizedSampler,  # added by RK 6/19
            sampler_args=dict(n_envs=1),

            # reset_arg=np.asscalar(taskIndex),
            reset_arg=taskIndex,
            log_dir=log_dir
        )
    else:
        policy = CategoricalMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(20, 20)
        )
        algo = TRPO(
            env=env ,
            policy=policy ,
            baseline=baseline ,
            batch_size=batch_size ,  # 2x
            max_path_length=max_path_length ,
            n_itr=n_itr ,
            # noise_opt = True,
            default_step=default_step ,
            sampler_cls=VectorizedSampler ,  # added by RK 6/19
            sampler_args=dict(n_envs=1) ,

            # reset_arg=np.asscalar(taskIndex),
            reset_arg=taskIndex ,
            log_dir=log_dir
        )
    test = algo.train()
    for i in test:
        print(i['env_infos'])
        r0 = np.sum(i['env_infos']['r0'])
        r1 = np.sum(i['env_infos']['r1'])
        break

    return r0,r1
####################### Example Testing script for Pushing ####################################
#envType = 'Push' ; max_path_length = 50 ; tasksFile = 'push_v4_val'



envType = 'Ant' ; annotation = 'v2-40tasks' ; tasksFile = 'rad2_quat_v2' ; max_path_length = 25
policyType = 'fullAda_Bias'
initFile = 'logs/dam/itr_151.pkl'

initFlr = 0.05 ; seed = 1
batch_size = 64
OUTPUT_DIR ='/data/local/'


# Provide the meta-trained file which will be used for testing

expPrefix = 'Test/Ant/'


#n_itr = 2
tasks = [0,1,2,3,4,5,6,7,8]
R0 , R1 = [] , []
for index in tasks:
    for n_itr in [200]:
        expPrefix_numItr = expPrefix+'/Task_'+str(index)+'/'

        # for n_itr in range(1,6):

        tf.reset_default_graph()
        expName = expPrefix_numItr+ 'Itr_'+str(n_itr)
        variant = {'taskIndex':index, 'init_file': initFile,  'n_parallel' : 1 ,   'log_dir':OUTPUT_DIR+expName+'/', 'seed' : seed  , 'tasksFile' : tasksFile , 'batch_size' : batch_size,
                        'policyType' : policyType ,  'n_itr' : n_itr , 'default_step' : initFlr , 'envType' : envType , 'max_path_length' : max_path_length}
        fig, ax = plt.subplots()

        r0,r1 = experiment(variant)
        R0.append(r0)
        R1.append(r1)

ax.scatter(R0, R1, color="green")
for i, txt in enumerate(tasks):
    ax.annotate(str(txt), (R0[i], R1[i]))
ax.ticklabel_format(useOffset=False)
plt.savefig(str(index)+"-meta_discrete"+str(n_itr))

