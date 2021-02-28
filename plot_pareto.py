import joblib as joblib
import numpy as np
import pickle
import matplotlib.pyplot as plt


flatten = lambda t: [item for sublist in t for item in sublist]
x_real = []
y_real = []
x_pred = []
y_pred = []
costs = []
paths = np.load("logs/dam/flr=0.0005/2-0.npy",allow_pickle=True).tolist()
#paths = np.load("/home/jasper/Documents/master/GMPS-master/launchers/266    paths10.npy",allow_pickle=True).tolist()
print(len(paths))
for i in range(6):
    weight = i * 0.01
    w = paths[i][0]
    moo_rewards = w["env_infos"]
    x_pred.append(sum(flatten(moo_rewards["r0"])))
    y_pred.append(sum(flatten(moo_rewards["r1"])))
for i in range(6):
    
            original = joblib.load(open("moo_envs/expert_traj/dam10/"+str(i)+".pkl", 'rb'))

            original_r0 = 0
            original_r1 = 0
            for rewards in original[0]["mo_rewards"][0]:
                original_r0 +=  rewards['r0'][0]
                original_r1 += rewards['r1'][0]
            x_real.append(original_r0)
            y_real.append(original_r1)
            costs.append([original_r0,original_r1])
            #print("weight = ", weight, "predicted r1 ",sum(flatten(moo_rewards["r1"])))
            #print("weight = ", weight, "original r1 ",original_r1)
            #print("weight = ", weight, "predicted r0 ",sum(flatten(moo_rewards["r0"])))
            #print("weight = ", weight, "original r0 ",original_r0)

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



def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

costs2 , ie= pareto_filter(np.array(costs))
costs3 = is_pareto_efficient_dumb(np.array(costs))
print(costs)
print(ie)
r0 = np.array(costs)[:,0]
r1 = np.array(costs)[:,1]
fig, ax = plt.subplots()
#ax.scatter(x_real[50:55], y_real[50:55])
n=[0.05,0.2,0.47,0.56,0.65,0.8]#[i/100 for i in range(0,100,5)]#[0.15,0.2,0.25,0.3,0.35,0.45,0.5, 0.6, 0.85]#[0.15,0.2,0.3,0.45,0.85]#[0.15,0.2,0.25,0.3,0.35,0.45,0.5, 0.6, 0.85]
ax.scatter(x_pred, y_pred)
for i, txt in enumerate(n):
    ax.annotate(txt, (x_pred[i], y_pred[i]))
ax.scatter(r0, r1 ,color="green")
for i, txt in enumerate(n):
    ax.annotate(txt, (r0[i], r1[i]))
ax.ticklabel_format(useOffset=False)
plt.show()