import numpy as np
import time
import graphviz

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
        self.first_I = []

    def get_D(self, I):
        if self.v == 3: print(f'\tget_D({I}=', end='')
        rho = tuple(sorted(I[:-1]) + [I[-1], -1])
        if self.v == 3: print(f'{rho})=')
        d = 0
        if rho in self.D.keys():
            if self.v == 3: print(f'\t\talready count={self.D[rho]}')
            return self.D[rho]
        for i in range(self.n):
            if i in I:
                d += self.P1[I[-1]][i]
            else:
                d += self.P[I[-1]][i]
        if self.v == 3: print(f'\t\tnew={d}')
        self.D[rho] = d
        return d

    def get_beta(self, I):
        if self.v == 2: print(f'get_beta({I})=')
        if len(I) == 1:
            if self.v == 2: print(f'\t{self.beta[I[0]]}')
            return self.beta[I[0]]
        return self.get_D(I) + self.get_beta(I[:-1])

    def get_nu(self, I):
        if self.v == 1: print(f'get_nu({I})=beta+sum(alpha)+eta_min=', end='')
        nu = self.get_beta(I)
        if self.v == 1: print(f'{nu}+(', end='')
        eta_min = self.m * self.n
        for i in range(self.n):
            if i not in I:
                eta_min = min(eta_min, self.eta[i])
                nu += self.alpha[i]
                if self.v == 1: print(f'{self.alpha[i]}+', end='')
        nu += eta_min
        if self.v == 1: print(f')+{eta_min}={nu}')
        return nu

    def get_addition(self, I, without=[]):
        A = list(range(self.n))
        I_add = []
        for i in A:
            if i not in I:
                if i not in without:
                    I_add.append(i)
        return I_add

    def solve(self, I, best=[], first=-1, last=-1, forced_down=False):
        if self.v: print(f'I{I}, best={best}')
        # print(f'best:{best}')
        if forced_down:
            I = self.get_first()
            if self.v: print(I[:-1])
            nu = self.get_nu(I[:-1])
            if self.v: print(nu)
            best = [nu, [I]]
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
        R_ = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            R_[i][i] = 1
            for j in range(i + 1, self.n):
                if self.P[i][j] > self.P[j][i]:
                    R_[i][j] = 1
                elif self.P[i][j] == self.P[j][i]:
                    R_[i][j] = 0.5
        k = dict()
        for i in range(self.n):
            k[i] = sum(R_[i])
        a = sorted(k.items(), key=lambda x: x[1])
        if self.v: print(f'get first={a}')
        self.first_I = [a[i][0] for i in range(self.n)]
        return [a[i][0] for i in range(self.n)]


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


def main(rR, c, n, m, forced_down=False, first=-1, last=-1, lin=True, v=False):
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

    solver = MatrixSolve(n, m, P)
    # I = get_first(P, n)
    # print(I, get_nu(I[:-1]))
    # ans = solve([])
    ans = solver.solve([], [], forced_down=forced_down, first=first, last=last)
    finish = time.time()
    ans_str = f'Значение функционала:{ans[0]}\nРешения:\n'
    for a in ans[1]:
        ans_str += f"\t"
        for i in a[-1::-1]:
            ans_str += f"{i + 1}<"
        ans_str = ans_str[:-1]+'\n'
    ans_str += f'Время {finish - start}s.\n'
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


def scc(C, n=None):
    def matrix_multiply(matrix_a, matrix_b):
        # Проверка размеров матриц
        if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
            raise ValueError("Матрицы должны быть одинакового размера.")

        n = len(matrix_a)
        result_matrix = [[0 for _ in range(n)] for _ in range(n)]

        # Перемножение матриц
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result_matrix[i][j] += int(bool(matrix_a[i][k] * matrix_b[k][j]))
                result_matrix[i][j]=int(bool(result_matrix[i][j]))
        return result_matrix
    def matrix_E(n):
        E = [[0] * n for _ in range(n)]
        for i in range(n):
            E[i][i] = 1
        return E

    def matrix_sum(M):
        return sum([np.array(m) for m in M])

    def matrix_trans(M):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

    def matrix_and(a, b):
        return np.array(a) & np.array(b)
    def process_matrix(matrix):
        def mark_and_get_nonzeros(row_idx):
            """Ищет индексы ненулевых элементов в указанной строке,
               одновременно помечая строку и столбцы нулевыми значениями."""
            indices = []
            for col_idx, value in enumerate(matrix[row_idx]):
                if value != 0:
                    indices.append(col_idx)
                    # Обнуляем весь столбец
                    for r in range(len(matrix)):
                        matrix[r][col_idx] = 0
            # Обнуляем всю строку
            matrix[row_idx][:] = [0] * len(matrix[row_idx])
            return indices

        result = []
        rows_to_check = list(range(len(matrix)))
        while rows_to_check:
            current_row = rows_to_check.pop(0)
            non_zero_cols = mark_and_get_nonzeros(current_row)
            if non_zero_cols:
                result.append(non_zero_cols)
                # Убираем из списка проверяемых строк те, которые уже стали пустыми
                rows_to_check[:] = [row for row in rows_to_check if sum(matrix[row]) > 0]
        return result
    if n is  None: n=len(C)
    A = [matrix_E(n)]
    A.append([row.copy() for row in C])
    for i in range(n):
        A.append(matrix_multiply(A[i], C))
    """for i in range(n + 1):
        if v:
            print(i)
            print(*A[i], sep='\n')"""
    T_D = matrix_sum(A)
    S_D = matrix_and(T_D, matrix_trans(T_D))
    return process_matrix(S_D)

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
    start = time.time()
    ans_str=''
    P = get_P(get_R(r, n, m), c, n, m, v)
    C = get_C(P,n)
    if v:
        print('C')
        print(C)
    D = scc(C)
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
        print(*CD, sep='\n')
        print('D')
        print(D)
    sD=[(sum(CD[i]),D[i])for i in range(k)]
    sD.sort()
    if v:
        print(sD)
    ans_str+=f'D={[sd[1] for sd in sD]}\n'

    ans_rename = []
    g = graphviz.Digraph(name='main', engine='fdp')
    g.attr(compound='true')

    for i in range(n):
        g.node(f'a{i}', label=f'{i + 1}')
    for k1 in range(k):
        with g.subgraph(name=f'cluster{k1}') as c:
            for i in D[k1]:
                c.node(f'a{i}')
                for j in D[k1]:
                    if C[i][j]:
                        c.edge(f'a{i}', f'a{j}')
            c.attr(label=f'D{k1 + 1}')
    for k1 in range(k):
        for k2 in range(k):
            if C[D[k1][0]][D[k2][0]]:
                # g.edge(f'a{D[k1][0]}',f'a{D[k2][0]}',ltail=f'cluster{k1}', lhead=f'cluster{k2}',color='red')
                g.edge(f'cluster{k1}', f'cluster{k2}', color='blue')

    print(g.source)
    g.format = 'png'
    g.render(directory='doctest-output', view=v)
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
    r, c, n, m = read_file('test20.txt')
    if m % 2: # and c==[1.0]*m
        print(special_case(r, c, n, m, 1))
    print(main(r, c, n, m))
    """
    ans=[]
    for i in range(n):
        print(f'last={i+1}')
        ans_str, a=main(r,c,n,m,last=i)
        print(ans_str)
        ans.append(a)
    print('\n'.join([f'r_last({i+1})={ans[i]}' for i in range(n)]))
    ans = []
    for i in range(n):
        print(f'first={i + 1}')
        ans_str, a = main(r, c, n, m, first=i)
        print(ans_str)
        ans.append(a)
    print('\n'.join([f'r_first({i + 1})={ans[i]}' for i in range(n)]))
    ans=dict()
    for i in range(n):
        for j in range(n):
            if i == j: continue
            print(f'first={i+1}, last={j+1}')
            ans_str, a=main(r,c,n,m,first=i,last=j)
            print(ans_str)
            ans[(i,j)]=a
    print('\n'.join([f'r({k[0]+1},{k[1]+1})={ans[k]}' for k in ans.keys()]))"""
