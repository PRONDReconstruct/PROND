from missing_utils import *
from hsic_construct_utils import *
import numpy as np

if __name__=='__main__':
    missing_rate=0.15   # unobserved data ratio
    nodes_num=200       # network size
    incomplete_choice = 1   # make data incomplete---0：random   1：load missing index from file
    small_times=1   # sample times for each graph
    big_times=6   # The sampling rounds for unobserved data S^mis
    sample_choice = 0    # initial sample  choice---0:sample with prior prob  1:sample with 0.5
    equation_flag_s = False   # False：iterative solve  True：fsolve method
    equation_flag_lambda = False  # False：iterative solve  True：fsolve method
    stop_threshold = 0.001   # stop condition for learning Probability Distribution
    initial_epsilon = 0.001   # learning rate
    degree =4   # degree  synthetic network
    perm_time = 1  # permutation time
    metric = 'linear'  # kernel function
    prune_choice = 'imi'  # decide which prune strategy to use. prune choice='hsic'/'mi'/'imi (infection mutual information)'

    graph_path='example_network.txt'
    result_path='example_diffusion_results.txt'
    missing_index_address='example_missing_index.txt'

    print("graph_path=%s, result_path=%s, missing_index_address=%s"%(graph_path, result_path, missing_index_address))
    print("missing_rate=%f, nodes_num=%d, incomplete_choice=%d, samll_times=%d, big_times=%d, "
          "sample_choice=%d, equation_flag_s=%r, equation_flag_lambda=%r, stop_threshold=%f, initial_epsilon=%f"
          ",per_time=%d,metric=%s,prune_choice=%s"%(missing_rate, nodes_num,incomplete_choice, small_times,big_times,
                                    sample_choice,equation_flag_s,equation_flag_lambda,stop_threshold,
                                    initial_epsilon,
                                    perm_time,metric,prune_choice))

    # load data
    ground_truth_network, diffusion_result = load_data(graph_path, result_path)
    print("load data suc.")
    p = ground_truth_network*0.3

    # make observation incomplete
    incomplete_result = make_incomplete(diffusion_result, missing_rate, nodes_num, incomplete_choice, missing_index_address)
    print("make data incomplete suc.")

    # initial sample
    results_list, prior_prob = init_sample(incomplete_result, small_times, big_times, sample_choice)

    print("init sample suc.")

    one_cnt_list = np.zeros(big_times)
    for i in range(big_times):
        result_i = results_list[i]
        one_cnt = np.sum(result_i[np.where(incomplete_result==-1)])
        one_cnt_list[i]=one_cnt

    print('initial sample one_cnt list=', one_cnt_list)
    print("average one_cnt =", np.mean(one_cnt_list))

    it_cnt = 1
    pre_network_stru = np.zeros(ground_truth_network.shape)

    total_construct_time = 0
    total_p_time = 0
    total_sample_time = 0

    begin_all = time.time()
    while True:
        it_begin = time.time()
        print("%d th iteration:"%it_cnt)

        # topology inference
        construct_begin = time.time()
        network_list = hsic_construct_list(results_list, big_times, metric, perm_time,prune_choice)
        print("construct network suc.")

        for i in range(big_times):
            precision, recall, f_score = cal_F1(ground_truth_network, network_list[i])
            print(precision)
            print(recall)
            print(f_score)
            print("--------------------")

        # combine network
        comb_network = combine_network(network_list)
        print("combine network suc.")
        construct_end = time.time()
        total_construct_time+=construct_end-construct_begin

        precision, recall, f_score = cal_F1(ground_truth_network, comb_network)
        print("precision = %.5f, recall = %.5f, f_score = %.5f" % (precision, recall, f_score))
        change_num = np.sum(abs(comb_network - pre_network_stru))
        print("number of changed edge：", change_num)
        if change_num==0:
            break
        pre_network_stru = comb_network.copy()


        if it_cnt==1:
            print("Pruning time cost:", time.time()-begin_all)

        p_begin = time.time()
        # initialize propagation probability
        p_matrix = init_p(comb_network)

        print("init p suc.")
        print("initial p sum =",np.sum(p_matrix))

        print("updating p with gradient...")
        grd_begin = time.time()
        grd_it_cnt = 1

        while True:
            inner_begin = time.time()
            s_matrix = cal_s(p_matrix, incomplete_result, comb_network, prior_prob, equation_flag_s)
            print("calculate s suc.")

            lambda_matrix = cal_lambda(p_matrix, s_matrix, comb_network, incomplete_result, equation_flag_lambda)
            print("calculate lambda suc.")

            pre_p_matrix = p_matrix.copy()

            p_matrix = update_p(p_matrix, s_matrix, lambda_matrix, comb_network, incomplete_result, initial_epsilon, grd_it_cnt)
            print("update p suc.")
            print("new p sum =", np.sum(p_matrix))

            mae = cal_mae(ground_truth_network, p, p_matrix)
            mse = cal_mse(p, p_matrix)
            print("MAE=%f, MSE=%f" % (mae, mse))

            inner_end = time.time()
            print("grad %d iter suc. time cost: %.2f"%(grd_it_cnt, inner_end-inner_begin))

            grd_it_cnt += 1
            max_delta_p = np.max(abs(p_matrix - pre_p_matrix))
            print("max_delta_p=", max_delta_p)
            print("number between 0,1：", np.sum((p_matrix > 0) * (p_matrix < 1)))
            print("-------------------------")
            if max_delta_p<stop_threshold or grd_it_cnt>=10:
                break


        grd_end = time.time()
        mae = cal_mae(ground_truth_network,p, p_matrix)
        print("MAE=",mae)
        mse = cal_mse(p, p_matrix)
        print('MSE=',mse)

        print("p updated. total iteration: %d,  time cost: %.5f second"%(grd_it_cnt, grd_end-grd_begin))
        p_end = time.time()
        total_p_time+=p_end-p_begin

        sample_begin = time.time()
        s_matrix = cal_s(p_matrix, incomplete_result, comb_network, prior_prob, equation_flag_s)  # 最新的s_matrix
        print("calculate s suc.")


        results_list = sample_data_s(s_matrix, incomplete_result, small_times, big_times)
        print("sample complete data suc.")
        sample_end = time.time()
        total_sample_time = sample_end-sample_begin

        one_cnt_list = np.zeros(big_times)
        for i in range(big_times):
            result_i = results_list[i]
            one_cnt = np.sum(result_i[np.where(incomplete_result == -1)])
            one_cnt_list[i] = one_cnt

        print('new sample one_cnt list=', one_cnt_list)
        print("average one_cnt =", np.mean(one_cnt_list))

        it_end = time.time()

        print("%dth iteration done，time cost: %.5f 秒" % (it_cnt, it_end - it_begin))
        if it_cnt>5:
            break

        it_cnt += 1


    end_all = time.time()
    print("algorithm done，total iteration:%d，total time cost: %.5f second"%(it_cnt, end_all-begin_all))
    print("construct network time cost:%.5f second，rate：%.5f "%(total_construct_time, total_construct_time/(end_all-begin_all)))
    print("estimate propagation probability time cost:%.5f second，rate：%.5f"%(total_p_time, total_p_time/(end_all-begin_all)))
    print("sample time cost:%.5f second，rate：%.5f"%(total_sample_time, total_sample_time/(end_all-begin_all)))