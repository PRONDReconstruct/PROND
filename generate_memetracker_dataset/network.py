import time
import re

def get_domain(full_url):
    gg_index = full_url.find('//',0,len(full_url))
    first_g_index = full_url.find('/',gg_index+2,len(full_url))
    domain = full_url[gg_index+2:first_g_index]
    return domain


def most_link(data_file,k):
    domain_link_cnt = {}
    for file in data_file:
        with open(file,'r') as f:
            for line in f:
                if line[0]=='P' or line[0]=='L':
                    url = line[2:]
                    domain = get_domain(url)
                    if domain in domain_link_cnt.keys():
                        domain_link_cnt[domain]+=1
                    else:
                        domain_link_cnt[domain]=1
                else:
                    continue
    domain_sorted = sorted(domain_link_cnt.items(),key=lambda x:x[1],reverse=True)
    topk_domain = domain_sorted[:k]
    return topk_domain


def most_document(data_file,k):
    domain_document_cnt = {}
    for file in data_file:
        with open(file, 'r') as f:
            for line in f:
                if line[0] == 'P':
                    url = line[2:]
                    domain = get_domain(url)
                    if domain in domain_document_cnt.keys():
                        domain_document_cnt[domain] += 1
                    else:
                        domain_document_cnt[domain] = 1
                else:
                    continue
    domain_sorted = sorted(domain_document_cnt.items(),key=lambda x:x[1],reverse=True)

    topk_domain = domain_sorted[:k]
    return topk_domain


def str2unixtm(tmstr):
    if tmstr[-1]=='\n':
        tmstr=tmstr[:-1]
    formatStr = "%Y-%m-%d %H:%M:%S"
    tmObject = time.strptime(tmstr, formatStr)
    tmStamp = time.mktime(tmObject)
    tmhour = tmStamp/3600
    return tmhour


def record_time(data_file, topk_domain):
    temp_topk_domain = [i[0] for i in topk_domain]
    domain_dic = {}
    time_dic={}
    for i in range(len(temp_topk_domain)):
        domain_dic[temp_topk_domain[i]] = i

    for file in data_file:
        with open(file, 'r') as f:
            record_flag=False
            for line in f:
                if line[0] == 'P':
                    url = line[2:]
                    domain = get_domain(url)
                    if domain in domain_dic.keys():
                        record_flag=True
                    else:
                        record_flag=False
                elif record_flag and line[0]=='T':
                    utime = str2unixtm(line[2:])
                    time_dic[url]=utime

    return time_dic


def generate_network(data_file, topk_domain, output_file, edge_threshold, time_dic):
    temp_topk_domain = [i[0] for i in topk_domain]
    domain_dic = {}
    edge_dic = {}
    for i in range(len(temp_topk_domain)):
        domain_dic[temp_topk_domain[i]]=i

    with open(output_file,'a') as f2:
        for i in range(len(topk_domain)):
            f2.write("%d\t%s\n"%(i,topk_domain[i][0]))
        f2.write("\n")

    top_link_cnt=0
    link_to_future_cnt=0
    for file in data_file:
        bbegin = time.time()
        with open(file,'r') as f:
            last_line = ''
            dest_link = []
            dest_url=[]
            for line in f:
                if line[0]=='P':
                    source = get_domain(line[2:])
                elif line[0]=='T':
                    stime = str2unixtm(line[2:])
                elif line[0]=='L':
                    dest_url.append(line[2:])
                    dest_link.append(get_domain(line[2:]))
                elif line[0]=='\n':
                    if len(dest_link)>0 and (source in domain_dic.keys()):
                        for i in range(len(dest_url)):
                            if dest_url[i] in time_dic.keys():
                                dtime=time_dic[dest_url[i]]
                                top_link_cnt += 1
                            else:
                                continue

                            if dtime>stime:
                                link_to_future_cnt+=1
                            if dtime<=stime and dest_link[i] in domain_dic.keys():
                                source_id = domain_dic[source]
                                dest_id = domain_dic[dest_link[i]]
                                cur_edge=(dest_id,source_id)
                                if cur_edge in edge_dic.keys():
                                    edge_dic[cur_edge]+=1
                                else:
                                    edge_dic[cur_edge]=1
                    dest_link.clear()
                    dest_url.clear()
                last_line = line

            if last_line[0]=='L':
                if len(dest_link) > 0 and (source in domain_dic.keys()):
                    for i in range(len(dest_url)):
                        if dest_url[i] in time_dic.keys():
                            dtime = time_dic[dest_url[i]]
                            top_link_cnt += 1
                        else:
                            continue

                        if dtime > stime:
                            link_to_future_cnt += 1
                        if dtime <= stime and dest_link[i] in domain_dic.keys():
                            source_id = domain_dic[source]
                            dest_id = domain_dic[dest_link[i]]
                            cur_edge = (dest_id, source_id)
                            if cur_edge in edge_dic.keys():
                                edge_dic[cur_edge] += 1
                            else:
                                edge_dic[cur_edge] = 1
                    dest_link.clear()
                    dest_url.clear()
        eend = time.time()
        print("file read finish, time cost ",eend-bbegin)
    print("link to future cnt=",link_to_future_cnt)
    print("top link cnt=",top_link_cnt)
    begin=time.time()
    with open(output_file,'a') as f2:
        for edge, ecnt in edge_dic.items():
            if ecnt>=edge_threshold:
                f2.write("%d\t%d\t%d\n"%(edge[0], edge[1], ecnt))
    end=time.time()
    print("write edge time cost:", end-begin)


def top_edge(network_file, k):
    edge_dic = {}
    edge_flag = False
    domain_dic={}
    with open(network_file,'r') as f:
        for line in f:
            if not edge_flag and line!='\n':
                domain_info = re.split('[\t\n]',line)
                domain_id = int(domain_info[0])
                domain = domain_info[1]
                domain_dic[domain]=domain_id
            if line=='\n':
                edge_flag=True
            if line!='\n' and edge_flag:
                edge_info = re.split('[\t\n]',line)
                node1=int(edge_info[0])
                node2=int(edge_info[1])
                cnt=int(edge_info[2])
                edge_dic[(node1,node2)]=cnt

    edge_sorted = sorted(edge_dic.items(), key=lambda x: x[1], reverse=True)

    cnt = 0
    topk_edge=[]
    for i in range(len(edge_sorted)):
        if edge_sorted[i][0][0]!=edge_sorted[i][0][1]:
            topk_edge.append(edge_sorted[i])
            cnt+=1
            if cnt==k:
                break

    with open('qc_top'+str(k)+'_network.txt','a') as f:
        for domain,domain_id in domain_dic.items():
            f.write("%d\t%s\n"%(domain_id,domain))
        f.write('\n')

        for i in range(len(topk_edge)):
            f.write("%d\t%d\t%d\n"%(topk_edge[i][0][0],topk_edge[i][0][1],topk_edge[i][1]))


if __name__=="__main__":
    data_file_list = ['./dataset/quotes_2008-08.txt',
                      './dataset/quotes_2008-09.txt',
                      './dataset/quotes_2008-10.txt',
                      './dataset/quotes_2008-11.txt',
                      './dataset/quotes_2008-12.txt',
                      './dataset/quotes_2009-01.txt',
                      './dataset/quotes_2009-02.txt',
                      './dataset/quotes_2009-03.txt',
                      './dataset/quotes_2009-04.txt']

    k=1000
    edge_threshold = 1
    output_file = './dataset/graph_document_top'+str(k)+'_'+str(edge_threshold)+'.txt'
    print("output_file=",output_file)

    begin=time.time()
    topk_domain=most_document(data_file_list,k)
    end_1=time.time()
    print("time cost1=",end_1-begin)
    print("topk_domain get! ")

    print("recording time")
    b=time.time()
    time_dic=record_time(data_file_list, topk_domain)
    e=time.time()
    print("record done. time cost",e-b)

    print("generating network...")
    generate_network(data_file_list,topk_domain,output_file,edge_threshold,time_dic)
    end_2=time.time()

    print("time cost2=",end_2-begin)