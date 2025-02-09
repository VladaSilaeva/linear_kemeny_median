import numpy as np

"""n = 5
r = [[3, 5, 2, 4, 1], [4, 5, 2, 3, 1], [1, 3, 5, 2, 4], [1, 2, 4, 3, 5], [5, 3, 2, 1, 4], [1, 4, 5, 2, 3]]
c = [0.5, 1, 2, 0.5, 0.5, 1.5]"""
"""n = 5
r = [[1,4,2,5,3],[1,3,2,5,4],[4,2,5,3,1],[5,3,4,2,1],[4,1,2,3,5],[3,2,5,4,1]]
c = [1.5,0.5,1,2,0.5,0.5]"""
n=20
r=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
   [16,15,20,19,18,17,14,13,12,11,10,9,8,7,6,5,4,3,2,1],
   [19,14,20,13,12,11,16,17,10,9,8,7,6,5,4,3,2,1,15,18],
   [18,13,12,11,16,10,9,8,7,6,20,17,5,4,3,2,1,15,14,19],
   [12,20,17,19,11,10,9,8,7,6,5,4,3,2,1,15,16,14,13,18],
   [20,11,16,10,18,17,9,8,7,6,5,4,3,2,1,15,14,13,12,19],
   [19,18,17,10,9,8,7,6,5,4,3,2,1,15,14,16,13,12,11,20],
   [18,19,9,20,17,8,7,6,5,4,3,2,1,16,15,14,13,12,11,10],
   [8,7,6,19,5,4,3,2,20,1,15,14,16,13,12,18,17,11,10,9]]
c=[1.0]*len(r)
"""n=10 # кол-во альтернатив (те кто соревнуются)
r=[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]
c=[1, 1] # важность (вес?) эксперта"""
"""n=5
r=[[1,2,3,4,5],[2,4,1,5,3],[4,2,1,5,3],[2,1,4,3,5],[3,1,2,4,5],[5,4,1,3,2]]
c=[2,1.5,0.5,1,0.5,0.5]"""
R = []
m=len(r) # кол-во экспертов
# 0. из r_k: a_i1<a_i3<a_i2 в R_k: [[1,1,1],[0,1,0],[0,1,1]]
# или сразу ввод R_k
for k in range(m):
    a = np.array([[-1] * n for _ in range(n)])
    for i in r[k]:
        a[i - 1][i - 1] = 1
        for j in range(n):
            if a[i - 1][j] == -1:
                a[i - 1][j] = 1
        for j in range(n):
            if a[j][i - 1] == -1:
                a[j][i - 1] = 0
    R.append(a)

for k in range(m):
    print("<", r[k], ">")
    print(*R[k], sep='\n')
P = np.zeros((n, n))
for k in range(m):
    P += c[k] * R[k]
print(f'P\n{P}\n')
P=P.T
P1 = np.zeros((n, n))
P1[:] = m
P1 -= P
print(P1)
print()
Px = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Px[i][j] = min(P[i][j], P1[i][j])
print(Px)

beta = np.zeros(n)
alpha = np.zeros(n)
gamma = np.zeros(n)
eta = np.zeros(n)
D=dict()
for i in range(n):
    beta[i] = P1[i][i] + (np.sum(P[i]) - P[i][i])
    gamma[i] = np.sum(P1[i])
    D[tuple(list(range(i))+list(range(i+1,n))+[i, -1])]=gamma[i]
    D[(i,-1)]=beta[i]
    alpha[i] = np.sum(Px[i])
    eta[i] = gamma[i] - alpha[i]
print(f'\nbeta:{beta}')
print(f'\ngamma:{gamma}')
print(f'\nalpha:{alpha}')
print(f'\neta:{eta}')
#print(D)


def get_D(I):
    rho=tuple(sorted(I[:-1])+[I[-1], -1])
    #print(I, rho)
    d = 0
    if rho in D.keys():
        #print("====")
        return D[rho]
    for i in range(n):
        if i in I:
            d += P1[I[-1]][i]
        else:
            d += P[I[-1]][i]
    D[rho]=d
    return d


def get_beta(I):
    if len(I) == 1:
        return beta[I[0]]
    return get_D(I) + get_beta(I[:-1])


def get_nu(I):
    nu = get_beta(I)
    eta_min = len(r) * n
    for i in range(n):
        if i not in I:
            eta_min = min(eta_min, eta[i])
            nu += alpha[i]
    nu += eta_min
    return nu


A = list(range(n))
#print(A)

def get_addition(I):
    I_add = []
    for i in A:
        if i not in I:
            I_add.append(i)
    return I_add


def solve(I, best=[]):
    #print(f'I{I}, best={best}')
    #print(f'best:{best}')

    I_add = get_addition(I)
    #print(I, A,I_add)
    nu_i = {}
    for i in I_add:
        nu_i[i] = get_nu(I + [i])
    nu_i_sort = sorted(nu_i.items(), key=lambda item: item[::-1])
    #print(f'nu:{nu_i_sort}')
    if len(best) and best[0] < nu_i_sort[0][1]:

        #print(f'-best={best}')
        return best
    if len(nu_i_sort) == 2:
        if len(best):
            if best[0] == nu_i_sort[0][1]:
                best[1].append(I + [nu_i_sort[0][0]])
            else:
                best[0] = nu_i_sort[0][1]
                best[1] = [I + [nu_i_sort[0][0]]]
        else:
            best=[-1,[]]
            best[0] = nu_i_sort[0][1]
            best[1] = [I + [nu_i_sort[0][0]]]
        if nu_i_sort[1][1] == nu_i_sort[0][1]:
            best[1].append(I + [nu_i_sort[1][0]])
        #print(f'1 best={best}')
        return best
    for nu in nu_i_sort:
        best=solve(I+[nu[0]], best)
    #print(f'2 best={best}')
    return best

I = [19,20,16,18,17,11,10,9,8,7,6,5,4,3,2,1,15,14,13,12]
I=[i-1 for i in I]
D_min=1074
ans = solve([ ])
print(f'solve:{ans[0]}  {[[j+1 for j in i] for i in ans[1]]}')

#print(f'solve:{solve(I[:-2])}')
#print(f'solve 1074:{solve([],[1074,[]])}')

#print("D", D)

