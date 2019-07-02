import numpy
import sys

E = 15

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
p = float(args[1])
# eps = float(args[2])

# f=calc_f(eps,p)
# print(calc_f(eps,p))
# print(calc_mnp(eps,p,f))

#E=8, 6node
# n_k = [ 0,  0,  2, 20, 70, 56, 28,  8,  1]

#E=11, 6node
# n_k = [  0,   0,   1,  11,  53, 151, 282, 330, 165,  55,  11,   1]

#E=15, 6node
n_k = [0, 0, 0, 0, 0, 6, 60,270,735,1345,1707,1365,455,105,15,1]

for eps in [0.2,0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]:
    Gamma=calc_f(p,eps)
    m=calc_mnp(p,eps,Gamma)
    print('{0}|{1}'.format(Gamma,m))

sum=0

for k in range(len(n_k)):
    sum += n_k[k] * p**k * p**(E-k)

print('feasible lower bound of eps')
print(sum)
