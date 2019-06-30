#coding: UTF-8

from graphillion import GraphSet
import random
import numpy as np
import sys
import copy

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

def calc_f(nk):
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
        else:
            break

    return y+1

def calc_mnp(nk,f):
    sum1=0.0
    sum2=0.0
    for k in range(0,E):
        sum1 += nk[k]*((p**k)*((1-p)**(E-k)))
    for k in range(f+1,E):
        sum2 += (comb(E,k)-nk[k])*((p**k)*((1-p)**(E-k)))
    m_np = comb(E,f)-int((eps - sum1 - sum2)/((p**f)*((1-p)**(E-f))))

    if m_np > 0:
        return m_np
    else:
        print("error: m_np is negative.")
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
    return cgn_s, cgn_d, r, congestion


def failed_G_cap(G_cap,node_list):
    G_fail = np.zeros((N,N))
    for i in node_list:
        G_fail[i[0]][i[1]] = G_cap[i[0]][i[1]]
    return G_fail


#initialize parameters
N = 6
E = 11
p = float(args[1])
eps = float(args[2])
U_c = 100
U_d = 1
delta = 1e-10



# --------------Setting candidates of failure pattern ---------------------------
#input graph
G_connect =[
        [0,1,1,1,1,1],
        [1,0,1,0,0,1],
        [1,1,0,1,1,0],
        [1,0,1,0,1,1],
        [1,0,1,1,0,0],
        [1,1,0,1,0,0]]
G_cap = np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        if(G_connect[i][j] == 1):
            G_cap[i][j] = np.random.randint(10*U_c, 100*U_c)
        else:
            G_cap[i][j] = delta

print('Network capacity')
print(G_cap)
print('\n')


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
print('The number of non-connected failure pattern')
print(nk)
print('\n')

num_fp = []
for i in range(0,E+1):
    num_fp.append(int(comb(E,i)-nk[i])) #the number of connected failure pattern

print('The number of connected failure pattern')
print(num_fp)
print('\n')


# --------------Setting link metrics ---------------------------
input = []

##dijkstra algorithm (attention!: input data graph can include failures )
target_graph = gc.larger(E-1).choice()
weights={}
for i in target_graph: #initialize all link weight as 1
    weights[i] = np.random.randint(1,5)
    #weights[i] = 1


# print("target: ", target_graph)
# print('\n')
# for src, dst in target_graph:
#     input.append((src, dst, np.random.randint(0,1000)))

print('initial link weight')
print(weights)
print('\n')

# g = dijkstra.Graph()
# for src, dst, weight in input:
#     g.add_edge(src, dst, weight)
#     g.add_edge(dst, src, weight)


#------------Setting traffic demand------------------
tr=[]
for i in range(100):
    s = np.random.randint(0,N-1)
    d = np.random.randint(0,N-1)
    while (s == d):
        d = np.random.randint(0,N-1)
    tr.append((s,d,np.random.randint(0, 100*U_d)))
# tr = [(0,1,3),(1,4,5)]
print('Traffic demands')
print(tr)
print('\n')

#----------Setting failure patterns--------

cand_list = GraphSet()

#1本以下の故障本数を持つものはcand_listへ
for i in range(0,1):
    cand_list.update(gc.len(E-i))
print(cand_list.len())


print("total 1 patterns to consider".format(cand_list.len()))
# for i in cand_list:
#      print(i)

#-------------Optimization------------------
I_max=5
tl = []
R_min = np.inf
MAX_loop = 100
weights_opt = copy.deepcopy(weights)
for i in range(0,I_max):
    C_cnt = 0
    C_max = 15

    target_graph = gc.larger(E-1).choice()
    weights={}
    for i in target_graph: #initialize all link weight as 1
        weights[i] = np.random.randint(1,5)
        #weights[i] = 1
    # print('initial link weight for ',i,'th iteraion')
    # print(weights)
    # print('\n')
    weights_tmp = copy.deepcopy(weights)

    r_pre = []

    for loop in range(0,MAX_loop):
        print('{0} th iteration'.format(loop))
        r = []
        # cng_list = []
        for i in cand_list:
            # print(i)
            G_fail = failed_G_cap(G_cap,i)
            # print(G_fail)
            cng_s,cng_d,r_val,congestion = calc_r(universe, i, tr, weights_tmp,G_fail)
            # print(congestion)
            r.append((cng_s,cng_d,round(r_val,10))) # tuple of the most congested link (s,d) and r
            # print(congestion)
            # cng_list.append((cng_s,cng_d,congestion))

        R_val = 0
        for s,d,r_val in r:
            if r_val > R_val:
                R_val = r_val
                R_link = (s,d) #最大輻輳値とそのリンクを保持
        print('congestion ratio')
        print(r)

        print('worst congestion')
        print(R_val)

        print('weight')
        print(weights_tmp)
        # print('\n')

        if not(weights_tmp in tl):
            # if weights_tmp in tl:
            #      print('flag')

            if r == r_pre:
                tl.append(weights_tmp)
                break

            if R_val < R_min:
                R_min = R_val
                # print('flag')
                weights_opt = copy.deepcopy(weights_tmp)
                # cng_opt = copy.deepcopy(cng_list)
                C_cnt = 0
            else:
                C_cnt += 1

            #termination condition
        print(C_cnt)
        if C_cnt > C_max or loop == MAX_loop-1:
            break

        r_pre = copy.deepcopy(r)

        weights_tmp[R_link] += 1 #最大輻輳リンクの重みをインクリメント
        print('\n')


print(R_min)
print(weights_opt)

print('-----optimized------')

# -----alpha----------
f = calc_f(nk)
m_np = calc_mnp(nk,f)
failure_list = GraphSet()


for i in range(0,f):
    failure_list.update(gc.len(E-i))
print(failure_list.len())

cand_select = gc.len(E-f)
#print(cand_select.len())
for cnt in range(0,m_np):
    rand_graph = next(cand_select.rand_iter())
    failure_list.add(rand_graph)
    cand_select.remove(rand_graph)

print("total {0} patterns to consider".format(failure_list.len()))


r = []
# cng_list = []
for i in failure_list:
    # print(i)
    G_fail = failed_G_cap(G_cap,i)
    # print(G_fail)
    cng_s,cng_d,r_val,congestion = calc_r(universe, i, tr, weights_opt,G_fail)
    r.append((cng_s,cng_d,round(r_val,10))) # tuple of the most congested link (s,d) and r
    # print(congestion)
    # cng_list.append((cng_s,cng_d,congestion))

R_val = 0
for s,d,r_val in r:
    if r_val > R_val:
        R_val = r_val
        R_link = (s,d) #最大輻輳値とそのリンクを保持
print('alpha')
print(R_val,R_link)
# print(r)
# print('\n')

# ----------beta---------

failure_list = GraphSet()

for i in range(0,1):
    failure_list.update(gc.len(E-i))
print(failure_list.len())

print("total {0} patterns to consider".format(failure_list.len()))


r = []
# cng_list = []
for i in failure_list:
    # print(i)
    G_fail = failed_G_cap(G_cap,i)
    # print(G_fail)
    cng_s,cng_d,r_val,congestion = calc_r(universe, i, tr, weights_opt,G_fail)
    r.append((cng_s,cng_d,round(r_val,10))) # tuple of the most congested link (s,d) and r
    print(congestion)
    # cng_list.append((cng_s,cng_d,congestion))

R_val = 0
for s,d,r_val in r:
    if r_val > R_val:
        R_val = r_val
        R_link = (s,d) #最大輻輳値とそのリンクを保持
print('beta')
print(R_val,R_link)
# print(r)
