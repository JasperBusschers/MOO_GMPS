import itertools
import pickle
import joblib

import gym
import numpy as np
import tqdm
from env_base.dam import DamEnv
from env_base.dsth import dsth
from tqdm.contrib.itertools import product
env = 'Dam2Objectives-v0'
env_ = dsth()#gym.make(env)

output = []
dict_output = {'rewards': [], 'agent_infos': [],  'actions': [], 'observations': [] , 'mo_rewards': []}


#samples = itertools.combinations(list(range(4)),25)
samples = itertools.combinations(range(4), 3)
print(list(samples))
samples = itertools.product([ i for i in range(4)],repeat=25)
rewards = []
for seq in samples:
    print(seq)
    observations = []
    actions = []
    Moo_rew = []
    s= env_.reset()
    r0 = 0
    r1 =0
    for i, a in enumerate(seq):
        ns, r, done, rewa = env_.step(a)
        r0 += 0.99**i* rewa['r0']
        r1 += 0.99**i*rewa['r1']
        observations.append(s)
        s=ns
        action = [0,0,0,0]
        action[a] = 1
        actions.append([action])
        Moo_rew.append(rewa)
    dict_output['mo_rewards'].append(np.array(Moo_rew))
    dict_output['observations'].append(observations)
    #dict_output['expert_actions'].append(actions)
    dict_output['actions'].append(actions)
    rewards.append([r0,r1])

def pareto_filter(costs, minimize=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    from https://stackoverflow.com/a/40239615
    """
    costs_copy = np.copy(costs) if minimize else -np.copy(costs)
    is_efficient = np.arange(costs_copy.shape[0])
    n_points = costs_copy.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs_copy):
        nondominated_point_mask = np.any(
            costs_copy < costs_copy[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs_copy = costs_copy[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1
    return [costs[i] for i in is_efficient], is_efficient

print(rewards)
rewards , is_efficient = pareto_filter(rewards)
with open("dam_brute_rewards"+ '.pkl', 'wb') as handle:
    pickle.dump(rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
dict_output['observations'] = [dict_output['observations'][i] for i in is_efficient]
dict_output['actions']= [dict_output['actions'][i] for i in is_efficient]
dict_output['mo_rewards'] = [dict_output['mo_rewards'][i] for i in is_efficient]
#dict_output['expert_actions'] = [dict_output['expert_actions'][i] for i in is_efficient]
IDS = []
weights = []
print(rewards)
for i in range(101):
    w1 = i/100
    w2 = 1-w1
    best_score = -3000000
    best_id = 0
    id = 0
    for R0, R1 in rewards:
        r= w1*R0 + w2 *R1
        if r> best_score:
            best_id = id
            best_score = r
        id+=1
    if best_id not in IDS:
        IDS.append(best_id)
        weights.append(w1)
    print("for weight ", w1 , " best trajectory = " , best_id,[rewards[best_id]])

rewards  = [rewards[i] for i in IDS]
dict_output['observations'] = [dict_output['observations'][i] for i in IDS]
dict_output['actions']= [dict_output['actions'][i] for i in IDS]
dict_output['mo_rewards'] = [dict_output['mo_rewards'][i] for i in IDS]
#dict_output['expert_actions'] = [dict_output['expert_actions'][i] for i in IDS]
Rewards = []
print(weights)
for seq ,w1 in zip(dict_output['actions'],weights):
    s= env_.reset()
    Rewards = []
    r0 = 0
    r1 =0
    for a in seq:
        ns, r, done, rewa = env_.step(np.argmax(a))
        r0 += rewa['r0']
        r1 += rewa['r1']
        r= w1*r0 + (1-w1) *r1

        Rewards.append(r)
    dict_output['rewards'].append(Rewards)


j = 0
for r,a,o,mo ,w in zip(dict_output['rewards'],dict_output['actions'],dict_output['observations'],dict_output['mo_rewards'],weights):
    output = []
    dict_output2 = {'rewards': [], 'agent_infos': [], 'actions': [], 'observations': [],
                    'mo_rewards': []}
    print(r)
    dict_output2['rewards']=np.array(r)
    dict_output2['mo_rewards'].append(mo)
    print("------------------------------------")

    o = [[tmp] for tmp in o]
    print(o)
    dict_output2['observations'].extend(o)
    dict_output2['observations'] = np.asarray(dict_output2['observations'])
    #dict_output2['expert_actions'].extend(ea)
    dict_output2['actions'].extend(a[0])
    print(a)
    dict_output2['actions'] = np.asarray(dict_output2['actions'])
    output.append(dict_output2)
    joblib.dump(output,"C:\\Users\\jasper\\Documents\\AI\\MOO_GMPS\\moo_envs\expert_traj\\dsth_discrete2\\"
                +str(j)+ '.pkl' )
    #with open("/home/jasper/Documents/master/GMPS-master/moo_envs/expert_traj/dam10/" +str(j)+ '.pkl',
    #          'wb') as handle:
    #    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    j+=1

