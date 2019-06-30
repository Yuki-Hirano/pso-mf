import numpy
import sys

E = 6

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

def calc_f(eps,p):
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

def calc_mnp(eps,p,f):
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

n_k = [ 0,  0,  2, 20, 70, 56, 28,  8,  1]
sum=0

for k in range(len(n_k)):
    sum += n_k[k] * p**k * p**(E-k)

print('feasible lower bound of eps')
print(sum)
