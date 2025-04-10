import numpy as np
import time

"""n = 5
r = [[3, 5, 2, 4, 1], [4, 5, 2, 3, 1], [1, 3, 5, 2, 4], [1, 2, 4, 3, 5], [5, 3, 2, 1, 4], [1, 4, 5, 2, 3]]
c = [0.5, 1, 2, 0.5, 0.5, 1.5]"""
"""n = 5
r = [[1,4,2,5,3],[1,3,2,5,4],[4,2,5,3,1],[5,3,4,2,1],[4,1,2,3,5],[3,2,5,4,1]]
c = [1.5,0.5,1,2,0.5,0.5]"""
"""n=20
r=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
   [16,15,20,19,18,17,14,13,12,11,10,9,8,7,6,5,4,3,2,1],
   [19,14,20,13,12,11,16,17,10,9,8,7,6,5,4,3,2,1,15,18],
   [18,13,12,11,16,10,9,8,7,6,20,17,5,4,3,2,1,15,14,19],
   [12,20,17,19,11,10,9,8,7,6,5,4,3,2,1,15,16,14,13,18],
   [20,11,16,10,18,17,9,8,7,6,5,4,3,2,1,15,14,13,12,19],
   [19,18,17,10,9,8,7,6,5,4,3,2,1,15,14,16,13,12,11,20],
   [18,19,9,20,17,8,7,6,5,4,3,2,1,16,15,14,13,12,11,10],
   [8,7,6,19,5,4,3,2,20,1,15,14,16,13,12,18,17,11,10,9]]
c=[1.0]*len(r)"""
"""n=10 # кол-во альтернатив (те кто соревнуются)
r=[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]
c=[1, 1] # важность (вес?) эксперта"""
"""n=5
r=[[1,2,3,4,5],[2,4,1,5,3],[4,2,1,5,3],[2,1,4,3,5],[3,1,2,4,5],[5,4,1,3,2]]
c=[2,1.5,0.5,1,0.5,0.5]"""

def read_file(filename='test.txt'):
    r=[]
    f = open(filename, 'r')
    c_line = f.readline()
    for line in f.readlines():
        if len(line.strip()):
            r.append([int(i) for i in line.split()])
        else:
            break
    f.close()
    n = len(r[0])
    m = len(r)  # кол-во экспертов
    if len(c_line.strip()):
        c = [float(i) for i in c_line.split()]
        if len(c) != m or sum(c) != m:
            print('wrong c')
            c=[1.0]*m
    else:
        c = [1.0] * m
    return r,c,n,m

def get_R(r,n,m):
    R=[]
    for k in range(m):
        a = [[-1] * n for _ in range(n)]
        for i in r[k]:
            a[i - 1][i - 1] = 1
            for j in range(n):
                if a[i - 1][j] == -1:
                    a[i - 1][j] = 1
            for j in range(n):
                if a[j][i - 1] == -1:
                    a[j][i - 1] = 0
        R.append(a)
    return R
def get_Pm(R,c,n,m, v=False):
    if v:
        print(f'n={n}, m={m}')
        print('R=')
        # print(R)
        for k in range(m):
            print("r = <", r[k], ">")
            print(*R[k], sep='\n')
            print('-' * n * 2)
    P = np.zeros((n, n))
    for k in range(m):
        P += c[k] * np.array(R[k])
    if v: print(f'P\n{P}\n')
    return P, m


def get_D(I,v=False):
    rho=tuple(sorted(I[:-1])+[I[-1], -1])
    if v: print(f'\tget_D({I}={rho})=')
    d = 0
    if rho in D.keys():
        ifv:print(f'\t\talready count={D[rho]}')
        return D[rho]
    for i in range(n):
        if i in I:
            d += P1[I[-1]][i]
        else:
            d += P[I[-1]][i]
    if v:print(f'\t\tnew={d}')
    D[rho]=d
    return d


def get_beta(I,v=False):
    if v: print(f'get_beta({I})=')
    if len(I) == 1:
        if v: print(f'\t{beta[I[0]]}')
        return beta[I[0]]
    return get_D(I,v) + get_beta(I[:-1],v)


def get_nu(I,v=False):
    if v: print(f'get_nu({I})=beta+sum(alpha)+eta_min=',end='')
    nu = get_beta(I,v)
    if v: print(f'{nu}+(',end='')
    eta_min = m * n
    for i in range(n):
        if i not in I:
            eta_min = min(eta_min, eta[i])
            nu += alpha[i]
            if v: print(f'{alpha[i]}+',end='')
    nu += eta_min
    if v: print(f')+{eta_min}={nu}')
    return nu



"""хранить последовательность как i[0]=номер '1' в поледовательности или -1 если не используется?
или просто доп массив 1/0 - беру/не беру..."""
def get_addition(I, n):
    A = list(range(n))
    I_add = []
    for i in A:
        if i not in I:
            I_add.append(i)
    return I_add


def solve(I, best=[], v=False, forced_down=False):
    if v:print(f'I{I}, best={best}')
    #print(f'best:{best}')
    if forced_down:
        if v: print(I[:-1])
        nu = get_nu(I[:-1],v)
        if v: print(nu)
        best=[nu,[I[:-1]]]
        I=[]
        forced_down=False
    I_add = get_addition(I, n)
    #print(I, A,I_add)
    nu_i = {}
    for i in I_add:
        nu_i[i] = get_nu(I + [i])
    nu_i_sort = sorted(nu_i.items(), key=lambda item: item[::-1])
    if v:print(f'\tnu:{nu_i_sort}')
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
        if len(best) and best[0]<nu[1]:
            return best
        best=solve(I+[nu[0]], best, v, forced_down)
    #print(f'2 best={best}')
    return best

def presolve(R,c,n,m, v=False):
    P, m = get_Pm(R,c,n,m)
    """P = [[0, 0, 1, 0, 0],
         [0, 0, 2, 3, 0],
         [0, 0, 0, 2, 0],
         [0, 0, 0, 0, 4],
         [2, 1, 0, 0, 0]]
    m=1"""
    n = len(P)
    P1 = np.zeros((n, n))
    P1[:] = m
    P1 -= P
    if v:
        print(P1)
        print()
    Px = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Px[i][j] = min(P[i][j], P1[i][j])
    if v: print(Px)

    beta = np.zeros(n)
    alpha = np.zeros(n)
    gamma = np.zeros(n)
    eta = np.zeros(n)
    D = dict()
    for i in range(n):
        beta[i] = P1[i][i] + (np.sum(P[i]) - P[i][i])
        gamma[i] = np.sum(P1[i])
        D[tuple(list(range(i)) + list(range(i + 1, n)) + [i, -1])] = gamma[i]
        D[(i, -1)] = beta[i]
        alpha[i] = np.sum(Px[i])
        eta[i] = gamma[i] - alpha[i]
    if v:
        print(f'\nbeta:{beta}')
        print(f'\ngamma:{gamma}')
        print(f'\nalpha:{alpha}')
        print(f'\neta:{eta}')
        # print(D)
    return P, P1, Px, alpha, beta, gamma, eta, D

def get_first(P,n):
    R_=[[0]*n for _ in range(n)]
    for i in range(n):
        R_[i][i]=1
        for j in range(i+1,n):
            if P[i][j]>P[j][i]:
                R_[i][j]=1
            elif P[i][j]==P[j][i]:
                R_[i][j]=0.5
    k=dict()
    for i in range(n):
        k[i]=sum(R_[i])
    a=sorted(k.items(),key=lambda x:x[1])
    #print(a)
    return [a[i][0] for i in range(n)]

def main(rR,c,n_,m_,lin=True):
    global P, P1, Px, alpha, beta, gamma, eta, D,n,m
    n,m=n_,m_
    R=[]
    if lin:
        R=get_R(rR,n,m)
    else:
        R=rR
    P, P1, Px, alpha, beta, gamma, eta, D = presolve(R, c, n, m)
    print(P)
    I = get_first(P, n)
    start = time.time()
    ans = solve(I, forced_down=True)
    ans = solve([])
    finish = time.time()
    ans_str=f'Решение:{ans[0]}\n'
    for a in ans[1]:
        ans_str+=f"\t{get_addition(a, n)[0] + 1}"
        for i in a[-1::-1]:
            ans_str+=f"<{i + 1}"
        ans_str+='\n'
    ans_str+=f'Время {finish - start}s.\n'
    return ans_str
P,P1,Px,alpha,beta,gamma,eta,D=[],[],[],[],[],[],[],dict()
n,m=-1,-1
if __name__=="__main__":
    r,c,n,m=read_file('test20.txt')
    print(main(r,c,n,m))
    """R=get_R(r,n,m)
    P, P1, Px, alpha, beta, gamma, eta, D=presolve(R,c,n,m)
    print(P)
    I=get_first(P,n)
    #print(I)
    start = time.time()
    ans=solve(I,forced_down=True)
    #I=[18, 19, 15, 17, 16, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14, 13, 12, 11]
    #ans=solve(I[::-1], forced_down=True)
    ans=solve([])
    finish = time.time()
    print(f'solve:{ans[0]}')
    for a in ans[1]:
        print(f"\t{get_addition(a,n)[0]+1}",end='')
        for i in a[-1::-1]:
            print(f"<{i+1}", end='')
        print()

    print(f'time {finish - start}s.')
    print(len(ans[1]))"""
