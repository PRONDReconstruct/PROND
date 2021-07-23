import numpy as np
import random


def IC_model(beta, network, initial_rate, infect_prob):
    """
    :param beta: number of diffusion processes
    :param network: numpy array network, shape:nxn, n denotes the network size.
    :param initial_rate: an integer. initial infection ratio
    :param infect_prob:numpy array, shape:nxn, propagation probability of each edge
    """
    nodes_num = network.shape[0]
    diffusion_result = np.zeros((beta, nodes_num))

    for i in range(beta):
        # randomly choose initially infected nodes
        initial_nodes_num = nodes_num*initial_rate
        sel_list = [j for j in range(nodes_num)]
        initial_nodes = random.sample(sel_list, int(initial_nodes_num))

        diffusion_result[i,initial_nodes]=1
        last_node_infected = np.zeros(nodes_num)

        while(True):
            new_infect = np.where((diffusion_result[i]-last_node_infected)==1)[0]
            if new_infect.size==0:
                break
            else:
                last_node_infected=diffusion_result[i].copy()
                for j in new_infect:
                    j_neighbor = np.where(network[j]==1)[0]
                    for k in j_neighbor:
                        if diffusion_result[i,k]==0:
                            randP = random.random()
                            if randP<infect_prob[j,k]:
                                diffusion_result[i,k]=1

    print("IC model done!")
    return diffusion_result


