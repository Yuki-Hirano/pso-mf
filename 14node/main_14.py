#coding: UTF-8

from graphillion import GraphSet
import random
import numpy as np
import sys
import copy
import time

start_time = time.perf_counter()

np.random.seed(1)
args = sys.argv

#---------functions-------------
def fact(n):
    if n ==0:
        return 1
    else:
        return n*fact(n-1)

#culculate combination nCk
def comb(n,k):
    sum=0
    sum += int(fact(n)/(fact(n-k)*fact(k)))
    return sum

def Count_nk(graph_list):
    nk=np.zeros(E+1)
    for i in graph_list:
        nk[E-len(i)] += 1

    for j in range(0,E+1):
        nk[j] = comb(E,j) - nk[j]
    return nk

def setUniverse(input_graph):
    universe = []
    for i in range(0,N):
        for j in range(0,N):
            if (i < j and input_graph[i][j] == 1):
                universe.append((i,j))
    GraphSet.set_universe(universe)

def calc_f(nk,p,eps):
    right_sum = 0.0
    for k in range(0,E+1):
        right_sum += nk[k]*((p**k)*((1-p)**(E-k)))
    #print(eps - right_sum)
    y = E-1
    while True:
        left = 0.0
        for k in range(y+1,E+1):
            left += (comb(E,k)-nk[k])*((p**k)*((1-p)**(E-k)))

        if(left <= (eps - right_sum)):
            y -= 1
            if (y+1 < 0):
                return 0
        else:
            break

    return y+1

def calc_mnp(nk,f,p,eps):
    sum1=0.0
    sum2=0.0
    for k in range(0,E):
        sum1 += nk[k]*((p**k)*((1-p)**(E-k)))
    for k in range(f+1,E):
        sum2 += (comb(E,k)-nk[k])*((p**k)*((1-p)**(E-k)))
    m_np = comb(E,f)-nk[f]-int((eps - sum1 - sum2)/((p**f)*((1-p)**(E-f))))
    if m_np > 0:
        return int(m_np)
    else:
        print("All failure patterns are ignored.")
        sys.exit()


def shortest_path(path_graphset, weight_dic):
    metric_list = []
    metric_tuple = []
    for path in path_graphset:
        sum=0
        for hop in path:
            sum += weight_dic[hop]
        metric_list.append(sum)
        metric_tuple.append((path,sum))

    # print(metric_tuple)
    min_metric = min(metric_list)
    ecmp_path = []
    for i in metric_tuple:
        if i[1] == min_metric:
            ecmp_path.append(i[0])
    return ecmp_path

def calc_r(universe, graph, traffic_matrix, weight_list, cap_matrix):
    flow = np.zeros((N,N))
    failed_link = list(set(universe) - set(graph))
    # print("failed_link", failed_link)
    for s,d,traffic in tr:
        # print('({0},{1})'.format(s,d))
        all_paths = GraphSet.paths(s,d)
        if failed_link == []:
            paths = all_paths
        else:
            paths = all_paths.excluding(failed_link)
        # print("paths", len(paths))
        ospf_path_list = shortest_path(paths, weight_list)
        # print(ospf_path_list)

        ecmp_div = len(ospf_path_list)
        #print(ecmp_div)
        for ospf_path in ospf_path_list:
            for hop in ospf_path:
                flow[hop[0]][hop[1]] += traffic/ecmp_div
                # print('{0},{1},{2}'.format(hop[0],hop[1],traffic/ecmp_div))

    # print(flow)
    congestion = flow/G_cap
    #print('------------')
    # print(congestion)
    cgn_s = int(np.argmax(congestion) / N)
    cgn_d = int(np.argmax(congestion) % N)
    r = np.amax(congestion)
    # print(r)
    # print((cgn_s,cgn_d))
    return cgn_s, cgn_d, r ,congestion


def failed_G_cap(G_cap,node_list):
    G_fail = np.zeros((N,N))
    for i in node_list:
        G_fail[i[0]][i[1]] = G_cap[i[0]][i[1]]
    return G_fail

def congestion_eval(G_cap,universe,candidate,traffic,weights):
    r=[]
    for i in candidate:
        # print(i)
        G_fail = failed_G_cap(G_cap,i)
        # print(G_fail)
        cng_s,cng_d,r_val,congestion = calc_r(universe, i, tr, weights,G_fail)
        # print(congestion)
        r.append((cng_s,cng_d,round(r_val,10))) # tuple of the most congested link (s,d) and r
        # print(congestion)
    return r

#initialize parameters
N = 14
E = 21
p = float(args[1])
eps = float(args[2])
U_c = 100
U_d = 1
delta = 1e-10



# --------------Setting candidates of failure pattern ---------------------------
#input graph
G_connect =[
[0,1,1,1,0,0,0,0,0,0,0,0,0,0],
[1,0,1,0,0,0,0,1,0,0,0,0,0,0],
[1,1,0,0,0,0,1,0,0,0,0,0,0,0],
[1,0,0,0,1,0,0,0,1,0,0,0,0,0],
[0,0,0,1,0,1,1,0,0,0,0,0,0,0],
[0,0,0,0,1,0,0,1,0,0,0,0,0,0],
[0,0,1,0,1,0,0,0,0,1,0,0,1,0],
[0,1,0,0,0,1,0,0,0,0,1,0,0,0],
[0,0,0,1,0,0,0,0,0,0,0,1,0,1],
[0,0,0,0,0,0,1,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,1,0,1,0,1,0,1],
[0,0,0,0,0,0,0,0,1,0,1,0,1,0],
[0,0,0,0,0,0,1,0,0,0,0,1,0,1],
[0,0,0,0,0,0,0,0,1,0,1,0,1,0],
        ]
G_cap = np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        if(G_connect[i][j] == 1):
            G_cap[i][j] = np.random.randint(10*U_c, 100*U_c)
        else:
            G_cap[i][j] = delta

# print('Network capacity')
# print(G_cap)
# print('\n')


# Set setUniverse
universe = []
for i in range(0,N):
    for j in range(0,N):
        if (i < j and G_connect[i][j] == 1):
            universe.append((i,j))
GraphSet.set_universe(universe)

gc = GraphSet.connected_components(range(N)) #Graphset with connectivity
#print(gc.len())
nk = Count_nk(gc) #the number of non-connected failure pattern
# print('The number of non-connected failure pattern')
# print(nk)
# print('\n')

num_fp = []
for i in range(0,E+1):
    num_fp.append(int(comb(E,i)-nk[i])) #the number of connected failure pattern

# print('The number of connected failure pattern')
# print(num_fp)
# print('\n')


# --------------Setting link metrics ---------------------------
target_graph = gc.larger(E-1).choice()
weights={}
for i in target_graph: #initialize all link weight as 1
    weights[i] = np.random.randint(1,5)


#------------Setting traffic demand------------------
tr=[]
for i in range(50):
    s = np.random.randint(0,N-1)
    d = np.random.randint(0,N-1)
    while (s == d):
        d = np.random.randint(0,N-1)
    tr.append((s,d,np.random.randint(0, 100*U_d)))
# tr = [(0,1,3),(1,4,5)]
# print('Traffic demands')
# print(tr)
# print('\n')

#---------Setting failure patterns---------------
f = calc_f(nk,p,eps)
m_np = calc_mnp(nk,f,p,eps)
# print("f = ", f, " m = ", m_np)
# file = open('output.txt','a')
# file.write("p ={0}, eps = {1}, f = {2}, m = {3}　\n".format(p, eps, f, m_np))
# file.close()


I_max= 20
C_max = 50

#-------------Optimization-mf------------------
tl_mf = []
R_min_mf = np.inf
w_opt_mf = copy.deepcopy(weights)

tl_so = []
R_min_so = np.inf
w_opt_so = copy.deepcopy(weights)

#-----Set F_mf in PSO-M  and non-failure case in SO-----
cand_mf = GraphSet()
cand_so = GraphSet()



#Setting for F_mf
for i in range(0,f):
    cand_mf.update(gc.len(E-i))
cand_select = gc.len(E-f)
for cnt in range(0,m_np):
    rand_graph = next(cand_select.rand_iter())
    cand_mf.add(rand_graph)
    cand_select.remove(rand_graph)


#Setting non-failure set
cand_so = GraphSet()
cand_so.update(gc.len(E))
# for i in cand_so:
#     print(i)

# print("total {0} patterns to consider".format(cand_mf.len()))
# print("total {0} patterns to consider".format(cand_so.len()))
# for i in cand_mf:
#       print(i)

for loop in range(0,I_max):

    #----------Step 1--------------


    target_graph = gc.larger(E-1).choice()
    weights={}
    for i in target_graph: #initialize all link weight
        weights[i] = np.random.randint(1,5)
        #weights[i] = 1
    # print('initial link weight for ',i,'th iteraion')
    # print(weights)
    # print('\n')
    w_tmp_mf = copy.deepcopy(weights)
    w_tmp_so = copy.deepcopy(weights)

    r_pre_mf = []
    r_pre_so = []

    #-------------Optimization-mf------------------
    C_cnt = 0
    # for i in cand_mf:
    #        print(i)
    while True:
        # print('{0} th iteration'.format(loop))
        # r = []
        # print(cand_mf.len())
        r_mf = congestion_eval(G_cap,universe,cand_mf,tr,w_tmp_mf)
        # r_so = congestion_eval(G_cap,universe,cand_so,tr,w_tmp_so)
        # print(r_mf)
        R_val_mf = 0
        # R_val_so = 0

        for s,d,r_val_mf in r_mf:
            if r_val_mf > R_val_mf:
                R_val_mf = r_val_mf
                R_link_mf = (s,d) #最大輻輳値とそのリンクを保持

        # for s,d,r_val_so in r_so:
        #     if r_val_so > R_val_so:
        #         R_val_so = r_val_so
        #         R_link_so = (s,d) #最大輻輳値とそのリンクを保持


        # print('congestion ratio')
        # print(r_mf)
        #
        # print('worst congestion')
        # print(R_val_mf)
        #
        # print('weight')
        # print(w_tmp_mf)
        # print('\n')
        # if w_tmp_mf in tl_mf:
        #     print('flag')

        if not(w_tmp_mf in tl_mf):

            if r_mf == r_pre_mf:
                tl_mf.append(w_tmp_mf)
                break

            if R_val_mf < R_min_mf:
                R_min_mf = R_val_mf
                # print('flag')
                w_opt_mf = copy.deepcopy(w_tmp_mf)
                # eval_cand = cand_mf
                # cng_opt = copy.deepcopy(cng_list)
                C_cnt = 0
            else:
                C_cnt += 1

            #termination condition
        # print(C_cnt)
        if C_cnt > C_max:
            break

        r_pre_mf = copy.deepcopy(r_mf)

        w_tmp_mf[R_link_mf] += 1 #最大輻輳リンクの重みをインクリメント


#-------------Optimization-so------------------
    # print('----------so-----------')
    C_cnt = 0
    # for i in cand_so:
    #        print(i)
    while True:
        # print('{0} th iteration'.format(loop))
        # r = []

        r_so = congestion_eval(G_cap,universe,cand_so,tr,w_tmp_so)
        # r_so = congestion_eval(G_cap,universe,cand_so,tr,w_tmp_so)
        # print(r_so)
        R_val_so = 0
        # R_val_so = 0

        for s,d,r_val_so in r_so:
            if r_val_so > R_val_so:
                R_val_so = r_val_so
                R_link_so = (s,d) #最大輻輳値とそのリンクを保持

        # for s,d,r_val_so in r_so:
        #     if r_val_so > R_val_so:
        #         R_val_so = r_val_so
        #         R_link_so = (s,d) #最大輻輳値とそのリンクを保持


        # print('congestion ratio')
        # print(r)
        #
        # print('worst congestion')
        # print(R_val_so)
        # print(R_min_so)
        #
        # print('weight')
        # print(w_tmp_so)
        # print('\n')

        if not(w_tmp_so in tl_so):
            # if weights_tmp in tl:
            # print('flag')

            if r_so == r_pre_so:
                tl_so.append(w_tmp_so)
                break

            if R_val_so < R_min_so:
                R_min_so = R_val_so
                # print('flag')
                w_opt_so = copy.deepcopy(w_tmp_so)
                # eval_cand = cand_so
                # cng_opt = copy.deepcopy(cng_list)
                C_cnt = 0
            else:
                C_cnt += 1

            #termination condition
        # print(C_cnt)
        if C_cnt > C_max:
            break

        r_pre_so = copy.deepcopy(r_so)

        w_tmp_so[R_link_so] += 1 #最大輻輳リンクの重みをインクリメント



# print('-----optimized------')
# print(tr)
#
# print(R_min_mf)
# print(w_opt_mf)
#
# print(R_min_so)
# print(w_opt_so)



# -----alpha----------
# print("total {0} patterns to consider".format(eval_cand.len()))
# for i in eval_cand:
#      print(i)

r_mf = congestion_eval(G_cap,universe,cand_mf,tr,w_opt_mf)
r_so = congestion_eval(G_cap,universe,cand_mf,tr,w_opt_so)

alpha_mf = 0
for s,d,r_val in r_mf:
    if r_val > alpha_mf:
        alpha_mf = r_val
        R_link_mf = (s,d) #最大輻輳値とそのリンクを保持

alpha_so = 0
for s,d,r_val in r_so:
    if r_val > alpha_so:
        alpha_so = r_val
        R_link_so = (s,d) #最大輻輳値とそのリンクを保持

# print('alpha')
# print(r)
# print('\n')
alpha = (alpha_so-alpha_mf)/alpha_so
# print(alpha_mf,alpha_so,alpha)

# ----------beta---------

# failure_list = GraphSet()
#
# for i in range(0,1):
#     failure_list.update(gc.len(E-i))
# print(failure_list.len())

# print("total {0} patterns to consider".format(cand_so.len()))

r_mf = congestion_eval(G_cap,universe,cand_so,tr,w_opt_mf)
r_so = congestion_eval(G_cap,universe,cand_so,tr,w_opt_so)

beta_mf = 0
for s,d,r_val in r_mf:
    if r_val > beta_mf:
        beta_mf = r_val
        R_link_mf = (s,d) #最大輻輳値とそのリンクを保持

beta_so = 0
for s,d,r_val in r_so:
    if r_val > beta_so:
        beta_so = r_val
        R_link_so = (s,d) #最大輻輳値とそのリンクを保持

# print('beta')
beta = (beta_mf-beta_so)/beta_so
# print(beta_mf,beta_so,beta)# print(r)

ecution_time = time.perf_counter() - start_time
# print(ecution_time)


#--------output--------------
data = [eps,f,m_np,cand_mf.len(),alpha_mf,alpha_so,alpha,beta_mf,beta_so,beta,ecution_time]
# print(data)
data_str = ','.join(map(str,data))
file_path = './result14_{0}_i{1}_c{2}_tr{3}_fix_2.txt'.format(str(p).split('.')[1],I_max,C_max,len(tr))
with open(file_path,'a',encoding="utf-8") as f:
    f.write(data_str)
    f.write('\n')
    f.write(str(w_opt_mf))
    f.write('\n')
    f.write(str(w_opt_so))
    f.write('\n')
