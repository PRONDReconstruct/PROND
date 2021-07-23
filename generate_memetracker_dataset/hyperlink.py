import time
from queue import Queue
import re
from matplotlib import pyplot as plt

def get_domain(full_url):
    gg_index = full_url.find('//',0,len(full_url))
    first_g_index = full_url.find('/',gg_index+2,len(full_url))
    domain = full_url[gg_index+2:first_g_index]
    return domain


def most_link(data_file,k,topk_domain_exist,domain_file):
    if not topk_domain_exist:
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
    else:
        topk_domain = []
        cnt=0
        with open(domain_file, 'r') as f:
            for line in f:
                if cnt>=k:
                    break
                domain_info = re.split('[\t\n]', line)
                domain = domain_info[0]
                domain_cnt = int(domain_info[1])
                topk_domain.append((domain, domain_cnt))
                cnt+=1
    return topk_domain


def most_document(data_file,k,topk_domain_exist,domain_file):
    if not topk_domain_exist:
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

        with open('./dataset/future_document_domain_cnt.txt','a') as f2:
            for i in range(len(domain_sorted)):
                f2.write(domain_sorted[i][0])
                f2.write('\t')
                f2.write(str(domain_sorted[i][1]))
                f2.write('\n')
        print("domain num=",len(domain_sorted))

        topk_domain = domain_sorted[:k]
    else:
        topk_domain = []
        cnt=0
        with open(domain_file,'r') as f:
            for line in f:
                if cnt>=k:
                    break
                domain_info = re.split('[\t\n]',line)
                domain=domain_info[0]
                domain_cnt=int(domain_info[1])
                topk_domain.append((domain,domain_cnt))
                cnt+=1
    return topk_domain



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



def str2unixtm(tmstr):
    if tmstr[-1]=='\n':
        tmstr=tmstr[:-1]
    formatStr = "%Y-%m-%d %H:%M:%S"
    tmObject = time.strptime(tmstr, formatStr)
    tmStamp = time.mktime(tmObject)
    tmhour = tmStamp/3600
    return tmhour


def reverse_edge_root_post(data_file,k,bound_child,domain_type,topk_domain_exist,domain_file):
    # domain_type=1 most document; domain_type=0 most link
    print("getting top %d domain..."%(k))
    begin = time.time()
    if domain_type==0:
        topk_domain = most_link(data_file,k,topk_domain_exist,domain_file)
    elif domain_type==1:
        topk_domain = most_document(data_file,k,topk_domain_exist,domain_file)
    end=time.time()
    print("finished, time cost",end-begin)

    time_dic=record_time(data_file, topk_domain)
    print("record time_dic done!")

    print("getting reverse edge...")
    reverse_edge = {}
    domain_dic = {}
    for i in range(len(topk_domain)):
        domain_dic[topk_domain[i][0]]=i
    root_post = []
    revbegin = time.time()
    file_cnt = 0
    for file in data_file:
        begin = time.time()
        with open(file,'r') as f:
            domain_flag = False
            parent_cnt=-1
            utime=-1
            for line in f:
                if line[0]=='P':
                    if parent_cnt==0:
                        root_post.append((url,utime))

                    url = line[2:]
                    domain = get_domain(url)
                    if domain in domain_dic.keys():
                        domain_flag=True
                        parent_cnt=0
                    else:
                        domain_flag=False
                        parent_cnt=-1
                elif domain_flag and line[0]=='T':
                    utime = str2unixtm(line[2:])
                elif domain_flag and line[0]=='L':
                    # parent_cnt+=1
                    p_url = line[2:]
                    p_domain = get_domain(p_url)
                    if p_domain in domain_dic.keys():
                        if p_url in time_dic.keys():
                            ptime=time_dic[p_url]
                        else:
                            continue
                        parent_cnt+=1
                        if ptime<=utime:
                            if p_url in reverse_edge.keys():
                                if len(reverse_edge[p_url])>=bound_child:    # limit on number of child nodes
                                    continue
                                reverse_edge[p_url].append((url,utime))
                            else:
                                reverse_edge[p_url]=[]
                                reverse_edge[p_url].append((url,utime))

            if parent_cnt==0:
                root_post.append((url, utime))
        file_cnt+=1

        end = time.time()
        print("reading %dth file done! time cost %f" % (file_cnt, end-begin))
    revend=time.time()
    print("reverse edge done! time cost",revend-revbegin)
    return reverse_edge, root_post, domain_dic


def post_cas2domain_cas(single_cascade):
    single_domain_cascade={}
    for url, utime in single_cascade.items():
        domain = get_domain(url)
        if domain in single_domain_cascade.keys():
            if utime<single_domain_cascade[domain]:
                single_domain_cascade[domain]=utime
        else:
            single_domain_cascade[domain]=utime
    return single_domain_cascade


def generate_cascades(data_file, cascades_output_file, diffusion_result_output_file, k, domain_type, bound_child, bound_level,
                      topk_domain_exist,domain_file,bound_traverse):
    reverse_edge, root_post, domain_dic = reverse_edge_root_post(data_file,k,bound_child, domain_type,topk_domain_exist,domain_file)

    cascades_cnt = 0
    print("generating cascades...")
    begin=time.time()
    print("len domain_dic=",len(domain_dic))
    with open(cascades_output_file,'a') as f1, open(diffusion_result_output_file,'a') as f2:
        for domain, domain_id in domain_dic.items():
            f1.write("%d\t%s\n"%(domain_id,domain))
            f2.write("%d\t%s\n" % (domain_id, domain))
        f1.write('\n')
        f2.write('\n')
        cas_len = []
        f1.flush()
        f2.flush()

        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("root post num=",len(root_post))
        rpost_traverse_cnt=0
        for rpost in root_post:
            rpost_traverse_cnt+=1
            myque = Queue()
            myque.put(rpost)
            tag = ('#',-1)          # mark the end of a traverse level
            myque.put(tag)
            level_cnt=0
            single_cascade = {}
            traverse_cnt=0
            while not myque.empty():
                if level_cnt>=bound_level:   # limit on the traverse level
                    break
                out_post = myque.get()
                url,utime=out_post
                if url=='#':
                    tag = ('#', -1)
                    myque.put(tag)
                    level_cnt+=1
                    continue
                single_cascade[url]=utime
                traverse_cnt+=1
                if traverse_cnt>=bound_traverse:
                    break
                if url not in reverse_edge.keys():
                    continue
                child_post = reverse_edge[url]
                for cpost in child_post:
                    url,_ = cpost
                    if url not in single_cascade.keys():
                        myque.put(cpost)
            domain_cas = post_cas2domain_cas(single_cascade)

            if len(domain_cas)>=2:   # non-trivial
                f1.write("%s"%(rpost[0]))
                f2.write("%s" % (rpost[0]))
                for domain,utime in domain_cas.items():
                    domain_id = domain_dic[domain]
                    f1.write(";%d,%f"%(domain_id,utime))
                    f2.write(";%d" % (domain_id))
                f1.write('\n')
                f2.write('\n')
                cas_len.append(len(domain_cas))
                cascades_cnt+=1

                # if cascades_cnt%1==0:
                end=time.time()
                print("%d cascades generated."%(cascades_cnt))
                print("len single_cascade:%d, len domain_cas:%d"%(len(single_cascade),len(domain_cas)))
                print("time cost",end-begin)


            if rpost_traverse_cnt%500==0:
                print("%d root post have been traversed"%(rpost_traverse_cnt))

        end = time.time()
        print("total %d cascades,time cost %f"%(cascades_cnt,end-begin))
        plt.hist(cas_len)
        plt.show()


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


    k=1000    # topk domain num
    domain_type=1     #0 most link ; 1 most document
    bound_child=500
    bound_level=100
    topk_domain_exist=False
    bound_traverse=1000000
    domain_file='./dataset/top1000_document_domain_cnt.txt'

    cascades_output_file='./dataset/cascades_top'+str(k)+'_'+str(domain_type)+'_'+str(bound_child)+'_'+str(bound_level)+'.txt'
    diffusion_result_output_file='./dataset/diffusion_result_top'+str(k)+'_'+str(domain_type)+'_'+str(bound_child)+'_'+str(bound_level)+'.txt'
    print("k=%d, domain_type=%d, bound_child=%d, bound_level=%d, bound_traverse=%d"%(k,domain_type,bound_child,bound_level,bound_traverse))
    print("data_file=",data_file_list)

    begin=time.time()
    generate_cascades(data_file_list, cascades_output_file, diffusion_result_output_file, k, domain_type,bound_child, bound_level
                      ,topk_domain_exist,domain_file,bound_traverse)
    end=time.time()
    print("finish! total time cost:",end-begin)