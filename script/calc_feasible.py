import numpy as np
import sys

E = 23

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

def calc_f(p,eps):
    y = E-1
    while True:
        left = 0.0
        for k in range(y+1,E+1):
            left += comb(E,k)*((p**k)*((1-p)**(E-k)))

        if(left <= eps):
            y -= 1
        else:
            break

    return y+1

def calc_f2(p,eps,nk):
    right_sum = 0.0
    for k in range(0,E+1):
        right_sum += nk[k]*((p**k)*((1-p)**(E-k)))
    # print(eps - right_sum)
    y = E-1
    while True:
        left = 0.0
        # print(y)
        for k in range(y+1,E+1):
            left += (comb(E,k)-nk[k])*((p**k)*((1-p)**(E-k)))
        # print(y,left)
        if(left <= (eps - right_sum)):
            y -= 1
            if (y+1 < 0):
                return 0
        else:
            break

    return y+1

def calc_mnp2(p,eps,nk,f):
    sum1=0.0
    sum2=0.0
    for k in range(0,E+1):
        # print(k)
        sum1 += nk[k]*((p**k)*((1-p)**(E-k)))
    # print(eps-sum1)
    for k in range(f+1,E+1):
        sum2 += (comb(E,k)-nk[k])*((p**k)*((1-p)**(E-k)))
    # print(sum2)
    # print(eps-sum1-sum2)
    # print(int((eps - sum1 - sum2)/(p**f*(1-p)**(E-f))))
    m_np = comb(E,f)-nk[f]-int((eps - sum1 - sum2)/((p**f)*((1-p)**(E-f))))

    if m_np > 0:
        return m_np
    else:
        print("All failure patterns are ignored.")


def calc_mnp(p,eps,f):
    sum=0.0

    for k in range(f+1,E):
        sum += comb(E,k) * ((p**k)*((1-p)**(E-k)))
    m_np = comb(E,f)-int((eps - sum)/((p**f)*((1-p)**(E-f))))

    if m_np >= 0:
        return m_np
    else:
        print("error: m_np is negative.")
        sys.exit()

args = sys.argv
#p = float(args[1])
#eps = float(args[2])

# f=calc_f(eps,p)
# print(calc_f(eps,p))
# print(calc_mnp(eps,p,f))

#E=8, 6node
# n_k = [ 0,  0,  2, 20, 70, 56, 28,  8,  1]

#E=11, 6node
# n_k = [  0,   0,   1,  11,  53, 151, 282, 330, 165,  55,  11,   1]

#E=15, 6node
# n_k = [0, 0, 0, 0, 0, 6, 60,270,735,1345,1707,1365,455,105,15,1]

# E=14, abilene
# n_k = [0, 0, 11, 142, 750, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1]

# E=23 NJlata
n_k=[0, 0, 3, 66, 688, 4517, 20940, 72816, 196874, 423158, 732883, 1030384, 1178774, 1094049, 8.17190, 490314, 245157, 100947, 33649, 8855,1771, 253, 23, 1]

#E=21 NSFNET
# n_k = [0, 0, 2, 51, 596, 4247,20539, 70386, 171993, 293930, 352716, 352716,293930, 203490, 116280, 54264, 20349, 5985, 1330, 210, 21, 1]


a = np.arange(0,0.001,0.0001)
b = np.arange(0.001, 0.01,0.001)
c = np.arange(0.01,0.1,0.01)
d = np.arange(0.1,1,0.1)
e = [0.0075]
p_list = np.concatenate([a,b,c,d,e])
# for p in p_list:



for p in p_list:
# for p in [0.150000]:
     sum=0
     for k in range(len(n_k)):
         # print(k)
         sum += n_k[k] * p**k * (1-p)**(E-k)
         if(sum >= 1.0):
              sum=1.0
     print('{0},{1}'.format(p.round(5),sum))

print('\n')

p=0.01

for eps in [0.4,0.3,0.2,0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015,0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003,0.0025, 0.002, 0.0018,0.001,]:
   Gamma=calc_f2(p,eps,n_k)
   m=calc_mnp2(p,eps,n_k,Gamma)
   print('{0},{1},{2}'.format(eps,Gamma,m))


# p=0.02
# eps =0.009
# Gamma=calc_f2(p,eps,n_k)
# print(Gamma)
# m=calc_mnp2(p,eps,n_k,Gamma)
# print('{0},{1},{2}'.format(eps,Gamma,m))
