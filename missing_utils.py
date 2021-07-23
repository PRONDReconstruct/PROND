import numpy as np
import random
import math
from sklearn.cluster import KMeans
from itertools import combinations
import time
from scipy.optimize import fsolve
from matplotlib import pyplot as plt


def load_data(graph_path, result_path):
    # load data: groundtruth network, complete_record

    with open(result_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        diffusion_result = np.array([[int(state) for state in line] for line in lines])

    nodes_num = diffusion_result.shape[1]

    with open(graph_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        data = np.array([[int(node) for node in line] for line in lines])

        ground_truth_network = np.zeros((nodes_num, nodes_num))
        edges_num = data.shape[0]
        for i in range(edges_num):
            ground_truth_network[data[i, 0] - 1, data[i, 1] - 1] = 1

    return ground_truth_network, diffusion_result


def make_incomplete(diffusion_result, missing_rate, nodes_num, incomplete_choice, missing_index_address):
    # make record incomplete
    results_num = diffusion_result.shape[0]

    # incomplete_choice = 0 random missing；  = 1 load missing index from file
    if incomplete_choice == 0:
        missing_num = int(results_num * nodes_num * missing_rate)
        missing_index = random.sample(range(results_num * nodes_num), missing_num)
        np.savetxt(missing_index_address,missing_index,fmt='%d',delimiter=' ')
    elif incomplete_choice == 1:
        missing_index = np.loadtxt(missing_index_address).astype(int)

    tmp = diffusion_result.reshape((1, -1))

    print("missing 1/0")
    cnt_tmp = tmp[0, missing_index]
    one_cnt = np.sum(cnt_tmp == 1)
    zero_cnt = np.sum(cnt_tmp == 0)
    print("one_cnt, zero_cnt, sum = ", one_cnt, zero_cnt, one_cnt + zero_cnt)
    print("1/0 = ", one_cnt / zero_cnt)

    tmp[0, missing_index] = -1

    incomplete_result = tmp.reshape((results_num, nodes_num))

    # -1 denotes missing data
    return incomplete_result



def init_sample(incomplete_result, small_times, big_times, sample_choice):
    # initial sample for unobserved data
    results_list = []
    result_num = incomplete_result.shape[0]
    missing_num = np.sum(incomplete_result == -1, axis=0)
    tmp = incomplete_result.copy()
    tmp[tmp == -1] = 0

    # initial sample  choice---0:sample with prior prob  1:sample with 0.5
    if sample_choice == 0:
        prior_prob = np.sum(tmp, axis=0) / (result_num - missing_num)  # evaluate the prior probability being infected of
        # each node from observed data
    else:
        prior_prob = 0.5 * np.ones(incomplete_result.shape[1])

    # sample for unobserved data
    temp_results_list=[]
    for i in range(small_times * big_times):
        missing_mask = (incomplete_result == -1)
        sample = np.random.rand(*incomplete_result.shape)
        one_index = np.where(sample<prior_prob)
        zero_index = np.where(sample>= prior_prob)
        sample[one_index] = 1
        sample[zero_index] = 0
        masked_sample = sample * missing_mask
        observed_mask = ~missing_mask
        sample_result = observed_mask * incomplete_result + masked_sample
        temp_results_list.append(sample_result)

    for i in range(big_times):
        cur_result = temp_results_list[i*small_times].copy()
        for j in range(small_times-1):
            cur_result = np.vstack((cur_result, temp_results_list[i*small_times+j+1]))

        results_list.append(cur_result)

    return results_list, prior_prob



def IMI_prune(record_states):
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

            IMI[j, k] = M00 + M11 - abs(M10) - abs(M01)
            # IMI[j, k] = M00 + M11 +M10 + M01
            IMI[k, j] = IMI[j, k]

    # cluster
    IMI[np.where(IMI<0)] = 0
    tmp_IMI = IMI.reshape((-1, 1))
    tmp_IMI = tmp_IMI[np.where(tmp_IMI>0)].reshape((-1,1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_IMI)
    label_pred = estimator.labels_
    temp_0 = tmp_IMI[label_pred == 0]
    temp_1 = tmp_IMI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(IMI>tau)] = 1

    return prune_network


def numpy2dec(line):
    j = 0
    for m in range(line.size):
        j = j + pow(2, line.size - 1 - m) * line[m]

    return int(j)


def cal_score(child, parents, record_states):
    j_num = pow(2, parents.size)
    count_states = np.zeros((2, j_num))
    records_num = record_states.shape[0]

    for record_index in range(records_num):
        i_state = record_states[record_index, child]
        if j_num>1:
            j_state = record_states[record_index, parents]
        else:
            j_state = np.zeros(1)
        count_states[int(i_state), numpy2dec(j_state)] += 1

    nij = np.sum(count_states, axis=0)
    nij[np.where(nij==0)]+=1  # avoid divide zero error
    temp_count = count_states.copy()
    temp_count[np.where(temp_count==0)]+=1     # avoid log zero error

    log_nijk = count_states * np.log2(temp_count/nij)
    left = np.sum(log_nijk)
    right = 1/2*np.sum(np.log2(nij+1))
    score = left-right

    return score


def icde_construct(record_states):
    beta, nodes_num = record_states.shape

    # calculate mutual information and prune
    prune_network = IMI_prune(record_states)
    print("parents_num = ", np.sum(prune_network, axis=0))

    # calculate scores for all possible parents combinations, then utilize greedy strategy select parents
    constructed_network = np.zeros(prune_network.shape)
    for i in range(nodes_num):
        # calculate upper bound of the number of parents
        N1=np.sum(record_states[:,i]==0)
        N2=np.sum(record_states[:,i]==1)
        bound = math.log(2*N1*math.log(beta/N1,2)+2*N2*math.log(beta/N2,2)+math.log(beta+1,2),2)

        candidate_parents = np.where(prune_network[:, i] == 1)[0]
        candidate_size = candidate_parents.size

        if candidate_size <= bound:
            constructed_network[candidate_parents, i] = 1
        else:
            print("larger than bound")
            par_comb_sets = []
            par_comb_sets.append((np.array([]), cal_score(i, np.array([]), record_states)))
            for k in range(1, int(bound + 1)):
                # obtain all the combinations with size k
                k_combs = list(combinations(candidate_parents, k))
                for comb in k_combs:
                    score = cal_score(i, np.array(comb), record_states)
                    par_comb_sets.append((np.array(comb), score))

            sorted_sets = sorted(par_comb_sets, key=lambda comb: comb[1], reverse=True)

            for comb in sorted_sets:
                if np.sum(constructed_network[:, i] >= bound):
                    break
                temp_rel = constructed_network[:, i]
                temp_rel[comb[0].astype(int)] = 1
                if np.sum(temp_rel) > bound:
                    continue
                constructed_network[:, i] = temp_rel

    return constructed_network


def construct_network_icde(result_list, sample_times):
    # construct network using TENDS
    network_list = []
    for i in range(sample_times):
        constructed_network = icde_construct(result_list[i])
        network_list.append(constructed_network)

    return network_list

def construct_network_twind(result_list, sample_times):
    # construct network using TWIND
    network_list = []
    for i in range(sample_times):
        constructed_network = aaai_construct(result_list[i])
        network_list.append(constructed_network)

    return network_list


def aaai_construct(record_states):
    beta, nodes_num = record_states.shape

    # prune
    prune_network, MI, tau = MI_prune(record_states)
    print("candidate parents_num = ",np.sum(prune_network, axis=0))

    # cal upper bound
    bound = math.log((beta+1) * math.log(math.e * (beta+1)/2, 2),2)
    print("bound = ", bound)

    # calculate scores for all possible parents combinations, then utilize greedy strategy select parents
    constructed_network = np.zeros(prune_network.shape)
    for i in range(nodes_num):
        candidate_parents = np.where(prune_network[:,i]==1)[0]
        candidate_size = candidate_parents.size

        if candidate_size<=bound:
            constructed_network[candidate_parents, i] = 1
        else:
            par_comb_sets = []
            par_comb_sets.append((np.array([]), cal_score_twind(i, np.array([]), record_states)))
            for k in range(1,int(bound+1)):
                # obtain all the combinations with size k
                k_combs = list(combinations(candidate_parents, k))
                for comb in k_combs:
                    score = cal_score_twind(i, np.array(comb), record_states)
                    par_comb_sets.append((np.array(comb), score))

            sorted_sets = sorted(par_comb_sets, key = lambda comb:comb[1], reverse= True)

            for comb in sorted_sets:
                if np.sum(constructed_network[:,i]>=bound):
                    break
                temp_rel = constructed_network[:,i]
                temp_rel[comb[0].astype(int)] = 1
                if np.sum(temp_rel)>bound:
                    continue
                constructed_network[:,i] = temp_rel

    return constructed_network


def MI_prune(record_states):
    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    MI = np.zeros((nodes_num, nodes_num))

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

            # MI[j, k] = M00 + M11 - abs(M10) - abs(M01)
            MI[j, k] = M00 + M11 + M10 + M01
            MI[k, j] = MI[j, k]

    # cluster
    MI[np.where(MI<0)] = 0
    tmp_MI = MI.reshape((-1, 1))
    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI>tau)] = 1

    return prune_network, MI, tau


def cal_score_twind(child, parents, record_states):
    j_num = pow(2, parents.size)
    count_states = np.zeros((2, j_num))
    records_num = record_states.shape[0]

    for record_index in range(records_num):
        i_state = record_states[record_index, child]
        if j_num>1:
            j_state = record_states[record_index, parents]
        else:
            j_state = np.zeros(1)
        count_states[int(i_state), numpy2dec(j_state)] += 1

    factorial_1 = count_states[0, :]
    factorial_2 = count_states[1, :]
    factorial_sum = np.sum(count_states, axis=0)+1

    log_1 = np.zeros(factorial_1.shape)
    log_2 = np.zeros(factorial_2.shape)
    log_sum = np.zeros(factorial_sum.shape)
    for i in range(j_num):
        cur_log = 0
        for k in range(int(factorial_1[i])):
            cur_log+= math.log(k+1,2)
        log_1[i] = cur_log

        cur_log = 0
        for k in range(int(factorial_2[i])):
            cur_log+= math.log(k+1,2)
        log_2[i] = cur_log

        cur_log = 0
        for k in range(int(factorial_sum[i])):
            cur_log += math.log(k + 1, 2)
        log_sum[i] = cur_log


    temp_result = log_1+log_2-log_sum
    score = np.sum(temp_result)

    return score



def mat_max(a,b):
    temp = (a-b)>0
    return a*temp + b*(~temp)


def combine_network(network_list):
    # 将sample_times个network拼成一个
    nodes_num = network_list[0].shape[0]
    comb_network = np.zeros((nodes_num,nodes_num))
    for network in network_list:
        comb_network = comb_network + network

    comb_network[comb_network >= 1] =1

    return comb_network


def init_p(network_stru):
    p_matrix = np.random.rand(*network_stru.shape)*network_stru
    return p_matrix


def myfunc_s(x, *args):
    still_missing, network, cur_states, p_matrix = args
    equation_list = []
    for i in still_missing:
        parents = np.where(network[:, i] == 1)[0]
        temp_states = np.zeros(parents.size)
        for index in range(parents.size):
            if parents[index] not in still_missing:
                temp_states[index] = cur_states[parents[index]]
            elif parents[index] in still_missing:
                x_index = np.where(still_missing == parents[index])[0]
                temp_states[index] = x[x_index]

        s_i = 1-np.prod(1-p_matrix[parents, i]*temp_states)
        index = np.where(still_missing==i)[0]
        equation_list.append(s_i - x[index])

    res = np.squeeze(np.array(equation_list))

    return res


def cal_single_s(p_matrix, obs_result, network_stru, prior_prob, equation_flag = False):
    states = obs_result.copy().astype(np.float)

    mis_nodes = np.where(obs_result == -1)[0]  # nodes with missing final infection statuses
    no_par_nodes = np.where(np.sum(network_stru, axis=0) == 0)[0]  # nodes without parents
    inter_nodes = np.intersect1d(mis_nodes, no_par_nodes).astype(int)  # missing and no parents

    for i in inter_nodes:
        states[i] = prior_prob[i]

    stop_flag = False
    it_cnt = 0

    # case 1: no missing parents
    while not stop_flag:
        it_cnt += 1
        stop_flag = True
        s_missing = np.where(states == -1)[0]

        for i in s_missing:
            parents = np.where(network_stru[:, i] == 1)[0]
            s_j = states[parents]
            if not (s_j == -1).any():
                stop_flag = False
                states[i] = 1-np.prod(1-p_matrix[parents, i]*s_j)

    # print("case 1 finish")

    if not equation_flag:
        # case 2: missing parents
        delta = 0.0001
        still_missing = np.where(states == -1)[0]

        # initialize still_missing
        states[still_missing] = 0.5

        iter_count = 0
        stop_flag = False
        while not stop_flag:
            stop_flag = True
            iter_count+=1
            if iter_count>=100:
                break

            for i in still_missing:
                parents = np.where(network_stru[:, i] == 1)[0]
                s_j = states[parents]
                pre_si = states[i]
                states[i] = 1 - np.prod(1 - p_matrix[parents, i] * s_j)

                if abs(pre_si - states[i]) > delta:
                    stop_flag = False


    else:
        still_missing = np.where(states == -1)[0]
        if still_missing.size > 0:
            # initialize still_missing
            still_missing_initial = 0.5 * np.ones(still_missing.size)

            still_missing_value = fsolve(myfunc_s, still_missing_initial, (still_missing, network_stru, states, p_matrix))
            states[still_missing] = still_missing_value

    return states


def cal_s(p_matrix, incomplete_result, network_stru, prior_prob, equation_flag):
    # cal s matrix according to current p matrix
    s_begin = time.time()
    beta = incomplete_result.shape[0]
    s_matrix = np.zeros(incomplete_result.shape)
    for i in range(beta):
        # print("beta = ",i)
        s_matrix[i] = cal_single_s(p_matrix, incomplete_result[i], network_stru, prior_prob, equation_flag)

    s_end = time.time()
    print("cal s time cost: ", s_end-s_begin)

    return s_matrix


def myfunc_lambda(x, *args):
    p_matrix, cur_states, network_stru, obs_result = args
    missing_index = np.where(obs_result==-1)[0]
    equation_list = []

    for i in missing_index:
        # left part of equation
        inf_children = np.where(network_stru[i,:]*obs_result==1)[0]   # the infected child nodes of node i
        if inf_children.size >0:
            inf_p = p_matrix[:,inf_children].copy()
            inf_s = cur_states.reshape((-1,1))*network_stru[:,inf_children]
            A = 1-np.prod(1-inf_p*inf_s, axis=0)
            A[np.where(A==0)]=np.inf
            inf_p[i,:] = 0    # Fi\vj
            left = np.sum(np.prod(1-inf_p*inf_s, axis=0) * p_matrix[i,inf_children]/A)
        else:
            left = 0


        # mid part of equation
        temp = obs_result.copy()
        temp[np.where(temp==0)]=2
        uninf_children = np.where(network_stru[i,:]*temp==2)[0]  # the uninfected child nodes of node i
        if uninf_children.size > 0:
            uninf_p = p_matrix[:,uninf_children].copy()
            uninf_s = cur_states.reshape((-1,1))*network_stru[:,uninf_children]
            B = np.prod(1-uninf_p*uninf_s, axis=0)
            uninf_p[i,:] = 0
            mid = np.sum(np.prod(1 - uninf_p * uninf_s, axis=0) * p_matrix[i, uninf_children] / B)
        else:
            mid = 0

        # right part of equation
        missing_children = np.where(network_stru[i,:]*obs_result==-1)[0]  # child nodes of node i with final infection statuses missing
        if missing_children.size >0:
            right = 0
            for miss_child in missing_children:
                p = p_matrix[:,miss_child].copy()
                s = cur_states * network_stru[:,miss_child]
                p[i] = 0
                lambda_index = np.where(missing_index==miss_child)[0]
                right+=x[lambda_index]*np.prod(1-p*s)*p_matrix[i, miss_child]
        else:
            right = 0

        lambda_index = np.where(missing_index==i)[0]
        equation_list.append(left-mid+right-x[lambda_index])

    res = np.squeeze(np.array(equation_list))

    return res


def cal_single_lambda(p_matrix, cur_states, network_stru, obs_result, equation_flag_lambda):
    missing_index = np.where(obs_result==-1)[0]
    single_lambda = np.zeros(network_stru.shape[0])
    if missing_index.size>0:
        if equation_flag_lambda:
            missing_index_initial = np.ones(missing_index.size)
            ret_lambda = fsolve(myfunc_lambda, missing_index_initial, (p_matrix, cur_states, network_stru, obs_result))
            single_lambda[missing_index] = ret_lambda
        else:
            delta = 0.0001

            # initialize_missing_lambda
            cur_lambda = np.ones(missing_index.size)
            cur_deltas = 0

            stop_flag = False
            iter_cnt = 0
            while not stop_flag:
                stop_flag = True

                pre_lambdas = cur_lambda.copy()
                pre_deltas = cur_deltas

                for i in missing_index:
                    # left part of equation
                    inf_children = np.where(network_stru[i, :] * obs_result == 1)[0]  # the infected child nodes of node i
                    if inf_children.size > 0:
                        inf_p = p_matrix[:, inf_children].copy()
                        inf_s = cur_states.reshape((-1, 1)) * network_stru[:, inf_children]
                        A = 1 - np.prod(1 - inf_p * inf_s, axis=0)
                        A[np.where(A == 0)] = np.inf
                        inf_p[i, :] = 0  # Fi\vj
                        left = np.sum(np.prod(1 - inf_p * inf_s, axis=0) * p_matrix[i, inf_children] / A)
                    else:
                        left = 0

                    # mid part of equation
                    temp = obs_result.copy()
                    temp[np.where(temp == 0)] = 2
                    uninf_children = np.where(network_stru[i, :] * temp == 2)[0]  # the uninfected child nodes of node i

                    if uninf_children.size > 0:
                        uninf_p = p_matrix[:, uninf_children].copy()
                        uninf_s = cur_states.reshape((-1, 1)) * network_stru[:, uninf_children]
                        B = np.prod(1 - uninf_p * uninf_s, axis=0)

                        B[np.where(B==0)]=np.inf

                        uninf_p[i, :] = 0
                        mid = np.sum(np.prod(1 - uninf_p * uninf_s, axis=0) * p_matrix[i, uninf_children] / B)
                    else:
                        mid = 0

                    # right part of equation
                    missing_children = np.where(network_stru[i, :] * obs_result == -1)[0]  # child nodes of node i with final infection statuses missing
                    if missing_children.size > 0:
                        right = 0
                        for miss_child in missing_children:
                            p = p_matrix[:, miss_child].copy()
                            s = cur_states * network_stru[:, miss_child]
                            p[i] = 0
                            lambda_index = np.where(missing_index == miss_child)[0]
                            right += cur_lambda[lambda_index] * np.prod(1 - p * s) * p_matrix[i, miss_child]
                    else:
                        right = 0

                    lambda_index = np.where(missing_index == i)[0]
                    pre_lambda = cur_lambda[lambda_index]
                    cur_lambda[lambda_index] = left-mid+right

                    if abs(pre_lambda - cur_lambda[lambda_index]) > delta:
                        stop_flag = False

                cur_deltas = np.sum(abs(pre_lambdas-cur_lambda))
                if cur_deltas == pre_deltas:
                    break

                iter_cnt += 1
                if iter_cnt>=100:
                    break

            # print("iter_cnt=", iter_cnt)
            single_lambda[missing_index] = cur_lambda

    return single_lambda

def cal_lambda(p_matrix, s_matrix, network_stru, incomplete_result, equation_flag_lambda):
    # cal lambda with current p and s matrix
    lambda_begin = time.time()
    beta = s_matrix.shape[0]
    lambda_matrix = np.zeros(s_matrix.shape)
    for i in range(beta):
        # print("beta = ", i)
        lambda_matrix[i] = cal_single_lambda(p_matrix, s_matrix[i], network_stru, incomplete_result[i], equation_flag_lambda)

    lambda_end = time.time()
    print("cal lambda time cost:", lambda_end-lambda_begin)

    return lambda_matrix


def update_p(p_matrix, s_matrix, lambda_matrix, network_stru, incomplete_result, initial_epsilon, iter_cnt):
    # cal gradient of p, and update p

    # step 1: cal gradient
    beta, nodes_num = incomplete_result.shape
    p_gradient_matrix = np.zeros(p_matrix.shape)

    for i in range(beta):
        for j in range(nodes_num):
            parents = np.where(network_stru[:,j]==1)[0]
            gradient_j = np.zeros(nodes_num)
            if incomplete_result[i,j]==0:
                temp = 1-p_matrix[:,j]*s_matrix[i]
                temp[np.where(temp==0)]=np.inf
                gradient_j = -1*s_matrix[i]/temp*network_stru[:,j]
            elif incomplete_result[i,j]==1:
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1-np.prod(1-p*s)

                if A==0:
                    A=np.inf

                p = np.repeat(p[:,np.newaxis], parents.size, axis=1)
                temp = np.arange(parents.size)
                p[parents,temp] = 0     # Fi\vj
                temp_gradient = np.prod(1-p*s[:,np.newaxis], axis=0)*s[parents]/A
                gradient_j[parents] = temp_gradient.copy()
            elif incomplete_result[i,j]==-1:
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                p = np.repeat(p[:, np.newaxis], parents.size, axis=1)
                temp = np.arange(parents.size)
                p[parents, temp] = 0  # Fi\vj
                temp_gradient = lambda_matrix[i,j]*np.prod(1 - p * s[:, np.newaxis], axis=0) * s[parents]
                gradient_j[parents] = temp_gradient.copy()

            p_gradient_matrix[:,j]+=gradient_j


    # step 2: update p
    epsilon = initial_epsilon/np.sqrt(iter_cnt)


    p_matrix+=epsilon*p_gradient_matrix

    p_matrix[np.where(p_matrix<0)]=0
    p_matrix[np.where(p_matrix>1)]=1


    return p_matrix


def sample_data_s(s_matrix, incomplete_result, small_times, big_times):
    # sample complete data according to current s matrix

    temp_results_list = []
    results_list = []
    for i in range(small_times * big_times):
        missing_mask = (incomplete_result == -1)
        sample = np.random.rand(*incomplete_result.shape)
        one_index = np.where(sample < s_matrix)
        zero_index = np.where(sample >= s_matrix)
        sample[one_index] = 1
        sample[zero_index] = 0
        masked_sample = sample * missing_mask
        observed_mask = ~missing_mask
        sample_result = observed_mask * incomplete_result + masked_sample
        temp_results_list.append(sample_result)

    for i in range(big_times):
        cur_result = temp_results_list[i * small_times].copy()
        for j in range(small_times - 1):
            cur_result = np.vstack((cur_result, temp_results_list[i * small_times + j + 1]))

        results_list.append(cur_result)

    return results_list



def cal_F1(ground_truth_network, inferred_network):
    TP = np.sum(ground_truth_network + inferred_network == 2)
    FP = np.sum(ground_truth_network - inferred_network == -1)
    FN = np.sum(ground_truth_network - inferred_network == 1)
    epsilon = 1e-5
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f_score = 2 * precision * recall / (precision + recall + epsilon)

    return precision, recall, f_score


def cal_mae(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    temp = gt_p.copy()
    temp[temp==0]=1
    infer_p = groundtruth_network*infer_p

    mae = np.sum(abs(infer_p-gt_p)/temp)/edges_num

    return mae

def cal_mse(p, infer_p):
    mse = np.mean(np.square(p-infer_p))

    return mse






