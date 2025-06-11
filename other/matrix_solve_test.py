import numpy as np
import time
import graphviz
count=0
class MatrixSolve:
    def __init__(self, n, m, P, v=0):
        self.v = v
        self.n = n
        self.m = m
        self.P = np.array(P)
        self.P1 = np.zeros((n, n))
        self.P1[:] = m
        self.P1 -= self.P
        if self.v:
            print('P1')
            print(self.P1)
            print()
        self.Px = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.Px[i][j] = min(self.P[i][j], self.P1[i][j])
        if self.v: print(self.Px)

        self.beta = np.zeros(self.n)
        self.alpha = np.zeros(self.n)
        self.gamma = np.zeros(self.n)
        self.eta = np.zeros(self.n)
        self.D = dict()
        self.first_I=[]
        for i in range(self.n):
            self.beta[i] = self.P1[i][i] + (np.sum(self.P[i]) - self.P[i][i])
            self.gamma[i] = np.sum(self.P1[i])
            self.D[tuple(list(range(i)) + list(range(i + 1, self.n)) + [i, -1])] = self.gamma[i]
            self.D[(i, -1)] = self.beta[i]
            self.alpha[i] = np.sum(self.Px[i])
            self.eta[i] = self.gamma[i] - self.alpha[i]
        if self.v:
            print(f'\nbeta:{self.beta}')
            print(f'\ngamma:{self.gamma}')
            print(f'\nalpha:{self.alpha}')
            print(f'\neta:{self.eta}')


    def get_D(self, I):
        if self.v > 3: print(f'\tget_D({I}=', end='')
        rho = tuple(sorted(I[:-1]) + [I[-1], -1])
        if self.v > 3: print(f'{rho})=')
        d = 0
        if rho in self.D.keys():
            if self.v > 3: print(f'\t\talready count={self.D[rho]}')
            return self.D[rho]
        for i in range(self.n):
            if i in I:
                d += self.P1[I[-1]][i]
            else:
                d += self.P[I[-1]][i]
        if self.v > 3: print(f'\t\tnew={d}')
        self.D[rho] = d
        return d

    def get_beta(self, I):
        if self.v > 2: print(f'get_beta({I})=')
        if len(I) == 1:
            if self.v > 2: print(f'\t{self.beta[I[0]]}')
            return self.beta[I[0]]
        return self.get_D(I) + self.get_beta(I[:-1])

    def get_nu(self, I):
        if self.v > 1: print(f'get_nu({I})=beta+sum(alpha)+eta_min=', end='')
        nu = self.get_beta(I)
        if self.v > 1: print(f'{nu}+(', end='')
        eta_min = self.m * self.n
        for i in range(self.n):
            if i not in I:
                eta_min = min(eta_min, self.eta[i])
                nu += self.alpha[i]
                if self.v > 1: print(f'{self.alpha[i]}+', end='')
        nu += eta_min
        if self.v > 1: print(f')+{eta_min}={nu}')
        return nu

    def get_addition(self, I, without=None):
        A = list(range(self.n))
        I_add = []
        for i in A:
            if i not in I:
                if without is not None and i not in without:
                    I_add.append(i)
        return I_add


    def solve(self, I=None, best=None, first=-1, last=-1, forced_down=False):
        if I in None: I=[[],list(range(self.n))]
        if best is None: best=[]
        global count
        count+=1
        if count%100==0:
            if self.v:print(f'count={count}')
        if self.v: print(f'I{I}, best={best}')
        if forced_down and first==-1 and last==-1:
            self.get_first()
            nu = self.get_nu([self.first_I,[]])
            if self.v: print(f'presolve record nu={nu}: {I[:-1]}')
            #print(f'presolve record nu={nu}: {I[:-1]}')
            best = [nu, []]
            I = []
            forced_down = False
        if first != -1:
            if not len(I):
                I = [first]
        I_add = self.get_addition(I, [last])
        if self.v: print(f'get_add{I}={I_add}')
        nu_i = {}
        for i in I_add:
            nu_i[i] = self.get_nu(I + [i])
        nu_i_sort = sorted(nu_i.items(), key=lambda item: item[::-1])
        if self.v: print(f'\tnu:{nu_i_sort}')
        # print(f'add:{I_add}')
        if len(best) and best[0] < nu_i_sort[0][1]:
            # print(f'-best={best}')
            return best

        if len(nu_i_sort) == 2 - int(last != -1):
            # print(f"-------------{nu_i_sort}")
            I_ = I + [nu_i_sort[0][0]]
            I_.append(self.get_addition(I_)[0])
            if len(best):
                if best[0] == nu_i_sort[0][1]:
                    best[1].append(I_)
                else:
                    best[0] = nu_i_sort[0][1]
                    best[1] = [I_]
            else:
                best = [-1, []]
                best[0] = nu_i_sort[0][1]
                best[1] = [I_]
            if len(nu_i_sort) == 2:  # last == -1
                I_ = I + [nu_i_sort[1][0]]
                I_.append(self.get_addition(I_)[0])
                if nu_i_sort[1][1] == nu_i_sort[0][1]:
                    best[1].append(I_)
            # print(f'1 best={best}')
            return best
        for nu in nu_i_sort:  # если не 2
            if len(best) and best[0] < nu[1]:
                return best
            best = self.solve(I + [nu[0]], best, first, last)
        # print(f'2 best={best}')
        return best

    def get_first(self):
        if len(self.first_I): return self.first_I
        R_ = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            R_[i][i] = 1
            for j in range(i + 1, self.n):
                if self.P[i][j] > self.P[j][i]:
                    R_[i][j] = 1.0
                elif self.P[i][j] == self.P[j][i]:
                    R_[i][j] = 0.5
        k = []
        for i in range(self.n):
            k.append([i, sum(R_[i])])
        k.sort(key=lambda x: x[1])
        if self.v: print(f'get first={k}')
        self.first_I=[[0]*(self.n-1),[0]]
        for i in range(n-1):
            self.first_I[0][k[i][0]]=i+1
        self.first_I[1][0]=self.n
        return self.first_I


def read_file(filename='test.txt'):
    r = []
    f = open(filename, 'r')
    nm_line = f.readline()
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
            c = [1.0] * m
    else:
        c = [1.0] * m
    return r, c, n, m


def get_R(r, n, m):
    R = []
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


def get_P(R, c, n, m, v=False):
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
    return P


def main(rR, c, n, m, forced_down=False, first=-1, last=-1, lin=True, v=0):
    global count
    count=0
    start = time.time()
    R = []
    if lin:
        R = get_R(rR, n, m)
    else:
        R = rR
    P = get_P(R, c, n, m)

    if v:
        print('P')
        print(P)

    solver = MatrixSolve(n, m, P,v)
    ans = solver.solve(forced_down=forced_down, first=first, last=last)
    print(f'f_d={forced_down}, f={first}, l={last}, ans={ans}')
    finish = time.time()
    ans_str = f'Значение функционала:{ans[0]}\nНайденные линейные порядки:\n'
    for a in ans[1]:
        ans_str += f"\t"
        for i in a[-1::-1]:
            ans_str += f"{i + 1}<"
        ans_str = ans_str[:-1]+'\n'
    ans_str += f'Время {finish - start}s.\nколичество count={count}\n'
    count=0
    """I=[8, 9, 4, 7, 0, 5, 6, 1, 3]
    print(f'{solver.get_nu(I)} {I}')
    I = [8, 7, 9, 0, 1, 4, 5, 3, 2, 6][:-1] # 7<3<4<6 <5<2<1<10<8 <9
    print(f'{solver.get_nu(I)} {I}')
    I = [8, 4, 7, 9, 0, 1, 5, 3, 2, 6][:-1] # 7<3<4<6 <2<1<10<8<5 <9
    print(f'{solver.get_nu(I)} {I}')"""
    I = ans[1][0][:-1]
    if v: print(f'{solver.get_nu(I)} {I}')
    # forced_down=True
    """if forced_down:
        start = time.time()
        ans = solver.solve([], [], forced_down=True)
        finish = time.time()
        ans_str += f'Решение с ускорением:{ans[0]}\n'
        ans_str += f'nu({solver.first_I})={solver.get_nu(solver.first_I[:-1])}\n'
        for a in ans[1]:
            ans_str += f"\t"
            for i in a[-1::-1]:
                ans_str += f"{i + 1}<"
            ans_str = ans_str[:-1]+'\n'
        ans_str += f'Время {finish - start}s.\n'"""
    return ans_str, ans


def scc(C,v=0):
    n=len(C)
    C_bool=np.array(np.matrix(C),dtype=bool)
    if v:
        print('C')
        print(C_bool)
    A=[np.array(np.eye(n),dtype=bool)]
    for i in range(n):
        A_i=np.matmul(A[-1],C_bool)
        if v:
            print(f'A{i}')
            print(np.array(A_i,dtype=int))
        A.append(A_i)
    T_D=np.array(sum(A),dtype=bool)
    if v:
        print('T_D')
        print(np.array(T_D,dtype=int))
    S_D=T_D&T_D.T
    if v:
        print('S_D')
        print(np.array(S_D,dtype=int))
    D=[]
    k=-1
    is_in_D=[0]*n
    while sum(is_in_D)!=n:
        for i in range(n):
            if is_in_D[i]:
                continue
            D.append([])
            for j in range(n):
                if is_in_D[j]:
                    continue
                if S_D[i][j]:
                    D[-1].append(j)
                    is_in_D[j]=1
    if v:
        print(D)
    return D

def get_C(P,n=None):
    if n is None: n =len(P)
    C = [[0] * n for i in range(n)]
    for i in range(len(C)):
        for j in range(len(C[0])):
            ci = P[i][j] - P[j][i]
            if ci > 0:
                C[i][j] = 1
    return C
def special_case(r, c, n, m, v=0):
    global count
    count=0
    start = time.time()
    ans_str=''
    P = get_P(get_R(r, n, m), c, n, m, v)
    C = get_C(P,n)
    if v:
        print('C')
        print(*C,sep='\n',end='\n\n')
    D = scc(C,v=v)
    k = len(D)
    if v: print(f'k={k}-----------------------')
    """if k == 1:#??
        return"""
    CD = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            CD[i][j] = C[D[i][0]][D[j][0]]
    if v:
        print('CD')
        print(*CD, sep='\n',end='\n\n')
        print('D')
        print(D)
    sD=[(sum(CD[i]),D[i])for i in range(k)]
    sD.sort()
    if v:
        print(sD)
    ans_str+=f'D={[[d+1 for d in sd[1]] for sd in sD]}\n'

    ans_rename = []
    if v!=-1:
        g = graphviz.Digraph(name='main', engine='fdp')
        g.attr(compound='true')

        for i in range(n):
            g.node(f'a{i}', label=f'{i + 1}')
        for k1 in range(k):
            with g.subgraph(name=f'cluster{k1}') as c:
                for i in sD[k1][1]:
                    c.node(f'a{i}')
                    for j in sD[k1][1]:
                        if C[i][j]:
                            c.edge(f'a{i}', f'a{j}')
                c.attr(label=f'D{k1 + 1} ({len(sD[k1][1])})')
        for k1 in range(k):
            for k2 in range(k):
                if C[sD[k1][1][0]][sD[k2][1][0]]:
                    # g.edge(f'a{D[k1][0]}',f'a{D[k2][0]}',ltail=f'cluster{k1}', lhead=f'cluster{k2}',color='red')
                    g.edge(f'cluster{k1}', f'cluster{k2}', color='blue')

        if v: print(g.source)
        g.format = 'png'
        g.render(directory='doctest-output', view=bool(v))
    best=[]
    for s,Di in sD:
        if v: print(Di)
        ni = len(Di)
        p = [[P[i][j] for j in Di] for i in Di]
        solver = MatrixSolve(ni, m, p, v=0)
        if v:
            print('ni Pi')
            print(ni)
            print(p)
        if ni == 1:
            a = [0, [[0]]]
        else:
            a = solver.solve([], [], forced_down=True)
        a_r = []
        best.append(a[0])
        for a_ in a[1]:
            ai_r = []
            if v: print(a_)
            for i in a_:
                ai_r.append(Di[i])
            a_r.append(ai_r)
        #ans_str+=f'\t{a[0]}\n\t{a_r}\n'
        ans_rename.append(a_r)
    if v:
        print('ren',end='\n\t')
        print(*ans_rename,sep='\n\t')
    #ans_str+=f'ans={ans_rename}\n'
    res=[[]]
    for part in ans_rename:
        res1 = []
        for r in res:
            for p in part:
                res1.append(r + p)
        res = res1
    finish = time.time()
    ans_str += f'Значения функционала:{best}\nРешения:\n'
    for a in res:
        ans_str += f"\t"
        for i in a[-1::-1]:
            ans_str += f"{i + 1}<"
        ans_str = ans_str[:-1] + '\n'
    ans_str += f'Время {finish - start}s.\n'
    return ans_str,[best,res]


if __name__ == "__main__":
    r, c, n, m = read_file('tests/test_n20_m9_from_article.txt')
    print(main(r, c, n, m, forced_down=False, v=5)[0])