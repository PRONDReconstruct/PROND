import time
import numpy as np
from sklearn.cluster import KMeans
import math
import copy
from sklearn.metrics.pairwise import pairwise_kernels


def hsic_construct_list(results_list, sample_times, metric,permutation_times,prune_choice,**kwargs):
    network_list = []
    for i in range(sample_times):
        constructed_network = hsic_construct_original(results_list[i], metric, permutation_times,prune_choice, **kwargs)
        network_list.append(constructed_network)

    return network_list


def fix_kmeans(data, max_iter=300):
    center_1 = np.min(data)
    center_2 = np.max(data)
    data_size = data.shape[0]
    label = -1*np.ones(data_size)

    last_center_2 = -1
    iter_cnt = 1
    while (not center_2==last_center_2) and (iter_cnt<=max_iter):
        for i in range(data_size):
            dist_1 = abs(data[i]-center_1)
            dist_2 = abs(data[i]-center_2)
            if dist_1<dist_2:
                label[i]=0
            else:
                label[i]=1

        last_center_2 = center_2
        center_2 = np.mean(data[np.where(label==1)])
        iter_cnt+=1

    return label


def prune_mi(record_states, mi_choice):
    # prune with (infection) mutual information
    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    IMI = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if j >= k:
                continue
            state_mat = np.zeros((2, 2))
            for result_index in range(results_num):
                state_mat[int(record_states[result_index, j]), int(record_states[result_index, k])] += 1

            epsilon = 1e-5
            M00 = state_mat[0, 0] / results_num * math.log(
                state_mat[0, 0] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M01 = state_mat[0, 1] / results_num * math.log(
                state_mat[0, 1] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)
            M10 = state_mat[1, 0] / results_num * math.log(
                state_mat[1, 0] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M11 = state_mat[1, 1] / results_num * math.log(
                state_mat[1, 1] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)

            if mi_choice == 0:
                IMI[j, k] = M00 + M11 + M10 + M01
            else:
                IMI[j, k] = M00 + M11 - abs(M10) - abs(M01)

            IMI[k, j] = IMI[j, k]

    IMI[np.where(IMI<0)] = 0
    tmp_IMI = IMI.reshape((-1, 1))
    tmp_IMI = tmp_IMI[np.where(tmp_IMI>0)].reshape((-1,1))
    label_pred = fix_kmeans(tmp_IMI)
    temp_0 = tmp_IMI[label_pred == 0]
    temp_1 = tmp_IMI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(IMI>tau)] = 1

    return prune_network


def cal_k_matrix2(x):
    # calculate kernel matrix
    data_size = x.shape[0]
    Kx = np.zeros((data_size, data_size))
    for i in range(data_size):
        for j in range(data_size):
            Kx[i, j] = np.dot(x[i], x[j])
    return Kx


def cal_hsic_matrix(all_data):
    kx_list = []
    for i in range(all_data.shape[1]):
        kx = cal_k_matrix2(all_data[:, i])
        kx_list.append(kx)
    dimension = len(kx_list)
    hsic_matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            if i > j:
                continue
            hsic_matrix[i, j] = cal_hsic(kx_list[i], kx_list[j])
            hsic_matrix[j, i] = hsic_matrix[i, j]

    return hsic_matrix



def cal_pruning_matrix(data,metric,per_time):
    print("begin calculating HSIC matrix")
    cov = cal_hsic_matrix(data)
    print("done")
    cov2 = copy.deepcopy(cov)
    nodes_num=cov2.shape[0]

    for i in range(cov2.shape[0]):
        for j in range(cov2.shape[0]):
            score=cal_score(data[:,i].reshape((-1,1)), data[:,j].reshape((-1,1)),metric,per_time)
            cov2[i,j]=score      # hsic/hsic-bias

    min_v = np.min(cov2)
    for i in range(cov2.shape[0]):
        cov2[i, i] = min_v

    cov2 = np.reshape(cov2, (-1, 1))

    tmp_val = cov2.reshape((-1, 1))
    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_val)
    label_pred = estimator.labels_
    temp_0 = tmp_val[label_pred == 0]
    temp_1 = tmp_val[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)
    cov2=cov2.reshape((nodes_num,nodes_num))
    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(cov2 > tau)] = 1

    return prune_network, cov



def hsic_construct_original(diffusion_result,metric, permutation_times, prune_choice, **kwargs):
    pruning_start = time.time()
    print("begin pruning")

    # prune_choice: decide which prune strategy to use
    # prune choice='hsic'/'mi'/'imi (infection mutual information)'
    if prune_choice=='hsic':
        pruning_matrix, _ = cal_pruning_matrix(diffusion_result,metric,permutation_times)
    elif prune_choice=='mi':
        pruning_matrix = prune_mi(diffusion_result, 0)
    elif prune_choice=='imi':
        pruning_matrix = prune_mi(diffusion_result, 1)
    # parents_num = sum(pruning_matrix,axis=0)
    # print("parents number: ", parents_num)
    print("pruning done")
    elapsed = (time.time() - pruning_start)
    print("pruning time:", elapsed)

    node_num = diffusion_result.shape[1]
    est_matrix = np.zeros((node_num, node_num))
    assess_start = time.time()
    print("begin assessing")
    n_count = 0
    p_count = 0
    for i in range(node_num):
        y_data = diffusion_result[:, i]
        data_size = diffusion_result.shape[0]
        x_data = np.zeros((data_size, 1))
        node_index = [i]

        for parent in range(diffusion_result.shape[1]):
            if pruning_matrix[parent, i] == 1:
                x_data = np.hstack((x_data, np.reshape(diffusion_result[:, parent], (-1, 1))))
                node_index.append(parent)
        # print("node_index ", i, ": ", node_index)
        if x_data.shape[1] == 1:
            # print("no parents for ", i)
            # print("continue next loop")
            continue
        x_data = x_data[:, 1:]
        y_data = np.reshape(y_data, (-1, 1))
        all_data = np.hstack((y_data, x_data))

        parents, best_score, history_parents, node_count, pruning_count = branch_and_bound(all_data, target_node=0,
                                                                                           score=-1, metric=metric,
                                                                                           permutation_times=permutation_times,
                                                                                           **kwargs)
        n_count += node_count
        p_count += pruning_count

        for parent in parents:
            est_matrix[node_index[parent], i] = 1

    assess_elapsed = (time.time() - assess_start)
    # print("p_count:", p_count)
    # print("n_count", n_count)
    print("assessing time:", assess_elapsed)
    print("\n")
    return est_matrix


class Node:
    """
    me: node_id, int
    parents: current selected parents , list
    score：with current selected parents，upper bound or certain value of the score, float
    size：nodes num, int
    next_branch: next branch to explore

    """
    def __init__(self, me, parents, score, size, next_branch):
        self.me = me
        self.parents = parents
        self.score = score
        self.size = size
        self.next_branch = next_branch

    def get_children(self, scores):
        """
        :param scores:list, includes two elements, the first is the upper bound of score when the next branch select a certain
        node; the second is the upper bound of score when a certain node is not selected at next branch
        :return:list，includes two elements, the first is a certain node is selected, the second is not
        """
        children = []
        next_branch = self.next_branch
        if next_branch == self.me:
            next_branch += 1
        if next_branch < self.size:
            child1 = Node(self.me, self.parents+[next_branch], scores[0], self.size, next_branch+1)
            child2 = Node(self.me, self.parents, scores[1], self.size, next_branch+1)
            children.append(child1)
            children.append(child2)
        return children

    def is_leaf_node(self):
        if self.next_branch < self.size-1:
            return False
        elif self.next_branch == self.size-1 and self.me < self.size-1:
            return False
        else:
            return True


def cal_hsic(kx, ky):
    n = kx.shape[0]
    kxy = np.dot(kx, ky)
    h = np.trace(kxy) / n ** 2 + np.mean(kx) * np.mean(ky) - 2 * np.mean(kxy) / n
    hsic_value = h * n ** 2 / (n - 1) ** 2
    return hsic_value


def cal_bias(x, y, metric, n=1, **kwargs):
    """
    :param x: parents nodes
    :param y: target node
    :param n: number of permutations times of y
    :return: bias
    """

    if x.shape[1]==0:
        return 0

    # np.random.seed(1)
    y_temp = copy.deepcopy(y)
    bias = 0
    for i in range(n):
        y_temp = np.random.permutation(y_temp)
        kx = cal_k_matrix(x,metric, **kwargs)
        ky = cal_k_matrix(y_temp,metric, **kwargs)
        bias += cal_hsic(kx, ky)/cal_hsic(ky, ky)
    return bias/n


def cal_k_matrix(x, metric, **kwargs):
    """
    :param x: n*m shaped array，n is the number of diffusion processes，m is the network size
    :return: kernel matrix
    """

    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))

    if x.shape[1] == 0:
        Kx = np.zeros((x.shape[0],x.shape[0]))
        return Kx

    Kx = pairwise_kernels(x,metric=metric, **kwargs)

    return Kx


def cal_score(x, y, metric, permutation_times, **kwargs):
    kx = cal_k_matrix(x, metric,**kwargs)
    ky = cal_k_matrix(y, metric,**kwargs)
    bias = cal_bias(x, y, metric, permutation_times,**kwargs)
    left = cal_hsic(kx, ky) / cal_hsic(ky, ky)

    score = left-bias

    return score



def cal_upper_bound2(xmax, y, metric, permutation_times, xparent, **kwargs):
    return cal_score(xmax, y, metric, permutation_times, **kwargs) - cal_bias(xparent, y, metric, permutation_times,**kwargs)

def branch_and_bound(data, target_node, score, metric, permutation_times, **kwargs):
    """
    Branch and Bound strategy to search the best parents for each node
    """
    best_score = score
    best_parents = []
    history_parents = {}
    history_upper_bound = {}
    root_parents = []
    next_branch = 0
    node_count = 1
    pruning_count = 0
    if target_node == 0:
        next_branch = 1
    root_node = Node(target_node, root_parents, 1, data.shape[1], next_branch)
    expand_set = [root_node]
    while expand_set:
        selected_node = expand_set.pop()
        children = selected_node.get_children([1, 1])
        node_count += 2
        # calculate upper bound for every child nodes
        # first situation, i.e. choose a certain node
        y_data = data[:, target_node]
        data_size = data.shape[0]

        # print("children 0 parents :",children[0].parents)
        x_data = np.zeros((data_size, 1))
        for parent in children[0].parents:
            x_data = np.hstack((x_data, np.reshape(data[:, parent], (-1, 1))))
        x_data = x_data[:, 1:]

        xmax_data = data[:, 1:]
        upper_bound1 = cal_upper_bound2(xmax_data, y_data, metric, permutation_times, x_data[:,0:-1],**kwargs)
        if upper_bound1 >= best_score:
            if children[0].is_leaf_node():
                score1 = cal_score(x_data, y_data, metric, permutation_times, **kwargs)
                if score1 >= best_score:
                    history_parents[best_score] = best_parents
                    history_upper_bound[best_score] = upper_bound1
                    best_parents = children[0].parents
                    best_score = score1
            else:
                expand_set.append(children[0])
        else:
            pruning_count += 1
            # print("pruning")

        # second situation,i.e. do not choose a certain node
        # print("children 1 parents :",children[1].parents)
        y_data = data[:, target_node]
        x_data = np.zeros((data_size, 1))
        for parent in children[1].parents:
            x_data = np.hstack((x_data, np.reshape(data[:, parent], (-1, 1))))
        x_data = x_data[:, 1:]

        xmax_data = data[:, 1:]
        upper_bound2 = cal_upper_bound2(xmax_data, y_data, metric, permutation_times, x_data[:,0:-1],**kwargs)
        if upper_bound2 >= best_score:
            if children[1].is_leaf_node():
                score2 = cal_score(x_data, y_data, metric, permutation_times, **kwargs)
                if score2 >= best_score:
                    history_upper_bound[best_score] = upper_bound2
                    history_parents[best_score] = best_parents
                    best_parents = children[1].parents
                    best_score = score2

            else:
                expand_set.append(children[1])
        else:
            pruning_count += 1
            # print("pruning")

    # print("candidate_parentes_size=", data.shape[1]-1)
    # print("prunning_count= ",pruning_count)
    return best_parents, best_score, history_parents, node_count, pruning_count