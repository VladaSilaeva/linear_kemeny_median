import numpy as np
import time
import graphviz
from itertools import permutations, product
from math import factorial
from random import shuffle
from treelib import Tree


count = 0
counting=True

class MatrixSolve:
    def __init__(self, n, m, P, v=0,use_tree=True):
        self.v = v  # verbal - вывод разных уровней работы программы (от 0 до 5)
        self.n = n  # количество альтернатив a_i
        self.m = m  # количество экспертов

        self.P = np.array(P)
        self.P1 = np.zeros((n, n))  # P^(1)
        self.Px = np.zeros((self.n, self.n))  # P^*

        self.beta = np.zeros(self.n)
        self.alpha = np.zeros(self.n)
        self.gamma = np.zeros(self.n)
        self.eta = np.zeros(self.n)

        self.D = dict()  # словарь хранит уже посчитанные значения D_{i1,i2,...,ik}

        self.first_I = None  # хранит значение предварительно посчитанного рекорда

        self.tree = None  # дерево решения
        self.tree_countD=0
        self.use_tree = use_tree

        self.P1[:] = m
        self.P1 -= self.P
        if self.v:
            print('P1')
            print(self.P1)
            print()

        for i in range(self.n):
            for j in range(self.n):
                self.Px[i][j] = min(self.P[i][j], self.P1[i][j])
        if self.v: print(self.Px)

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
        """Функция считает новые значения D_{i1,i2,...,ik} или берет их из словаря self.D
        Принимает:
            I =[i1, i2, ..., ik] - список индексов текущей последовательности <a_i1, a_i2, ..., a_ik>
        Возвращает:
            float - значение D_{i1,i2,...,ik}"""
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
        """Функция рекурсивно считает значения beta_{i1,...,ik}=D_{i1,...,ik}+beta_{i1,...,ik-1}
        Принимает
            I =[i1, i2, ..., ik] - список индексов текущей последовательности <a_i1, a_i2, ..., a_ik>
        Возращает
            float - значения beta_{i1,...,ik}"""
        if self.v > 2: print(f'get_beta({I})=')
        if len(I) == 1:
            if self.v > 2: print(f'\t{self.beta[I[0]]}')
            return self.beta[I[0]]
        return self.get_D(I) + self.get_beta(I[:-1])

    def get_nu(self, I):
        """Функция считает значения нижней оценки nu_{i1,i2,...,ik}
        Принимает:
            I =[i1, i2, ..., ik] - список индексов текущей последовательности <a_i1, a_i2, ..., a_ik>
        Возращает
            float - значения beta_{i1,...,ik}"""
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


    def solve(self, I=None, I_add=None, best=None, first=-1, last=-1, forced_down=False):
        """Функция рекурсивного обхода дерева решений для метода ветвей и границ.
        Принимает:
            first=i_f - индекс закрепленного на 1-м месте или -1, если 1-й не закреплен
            last=i_l - индекс закрепленного на n-м месте или -1, если последний не закреплен
            forced_down - считать ли предварительный рекорд
        Передает сама себе:
            I =[i1, i2, ..., ik] - список индексов текущей последовательности <a_i1, a_i2, ..., a_ik>
            I_add=set({1,2,...,n}\{i1, i2, ..., ik}) - дополнение к I. Также если закреплен последний, то last не в I_add
            best =[D, [[i1,i2,...,in-1], [i1',i2',...,in-1'], ...]], где
                D - текущее значение рекорда,
                rho=<a_i1, a_i2, ..., a_ik>, rho'=<a_i1', a_i2', ..., a_ik'> -найденные ранжировки с D(rho)=D(rho')

        Возвращает:
            [float, [[int]*n,...]] - best - значение рекорда (float) и множество всех медиан
                (медиана - [i1,...,in], где ik - int от 1 до n)"""

        global count  # для подсчета количества вызовов solve
        count+=1
        if count % 100 == 0:
            if self.v or counting:print(count)
        if I is None: # начало решения
            if self.use_tree: self.tree = Tree()
            if self.v:print("solution start")
            I=[]
            I_add=set(range(self.n))
            if forced_down: # найти предварительный рекорд
                self.get_first()
                if self.v: print(f"get first nu={self.first_I[0]}")
                best=[self.first_I[0], []]
            if first != -1: # если первый закреплен
                if self.use_tree:self.tree.create_node(f"({first+1})", f"{first}")
                if self.v:print(f"first={first}")
                I.append(first)
                I_add.remove(first)
            else:
                if self.use_tree:self.tree.create_node("(-)", "")
            if last != -1:
                if self.v:print(f"last={last}")
                I_add.remove(last)
        if self.v:print(f"I={I}, I_add={I_add}, best={best}")
        nu_beta_i = [(self.get_nu(I+[i]),self.get_beta(I+[i]),i) for i in I_add]
        nu_beta_i.sort()
        a=tuple([i for nu,beta, i in nu_beta_i])
        nu_beta_i.sort(key=lambda x:(x[0], x[2]))
        b = tuple([i for nu, beta, i in nu_beta_i])
        print(a==b)
        if self.v:print(f"nu={nu_beta_i}")
        if len(I) == self.n-2: # если уровень дерева n-2 (ниже спускаться не нужно)
            if self.v:print(f"level n-2")
            if last!=-1: end=last
            else: end=nu_beta_i[1][2]
            if best is None: # первый рекорд
                best = [nu_beta_i[0][0], [I + [nu_beta_i[0][2],end]]]
                if self.use_tree: self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[0][0]}─[{nu_beta_i[0][2]+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[0][2],end])),
                                                        parent=' '.join(map(str, I + [nu_beta_i[0][2]])))
                if self.v:print(f"first best={best}")
            elif best[0] == nu_beta_i[0][0]: # добавляем послед-ть0 к существующему рекорду
                best[1].append(I + [nu_beta_i[0][2],end])
                if self.use_tree: self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[0][0]}─[{nu_beta_i[0][2]+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[0][2],end])),
                                                        parent=' '.join(map(str, I + [nu_beta_i[0][2]])))
                if self.v:print(f"add to best {best[1][-1]}")
            elif best[0] > nu_beta_i[0][0]:  # послед-ть0 - новый рекорд
                if self.v:print(f"{best[0]}>{nu_beta_i[0][0]} => ", end='')
                best = [nu_beta_i[0][0], [I + [nu_beta_i[0][2],end]]]
                self.tree_countD+=1
                if self.use_tree: self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[0][0]}─[{nu_beta_i[0][2]+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[0][2],end])),
                                                        parent=' '.join(map(str, I + [nu_beta_i[0][2]])))
                if self.v:print(f"new best={best}")
            else: # рекорд лучше ()
                return best
            if best[0] == nu_beta_i[1][0]: # случай если nu1==nu2
                best[1].append(I + [nu_beta_i[1][2]])
                if self.use_tree: self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[1][0]}─[{nu_beta_i[1][2]+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[1][2],nu_beta_i[0][2]])),
                                                        parent=' '.join(map(str, I + [nu_beta_i[1][2]])))
                if self.v:print(f"add to best {I + [nu_beta_i[1][2],nu_beta_i[0][2]]}")
            return best
        k=0
        while k<len(nu_beta_i):
            nu,beta,i=nu_beta_i[k]
            if self.v:print(f"nu={nu}, beta={beta}, I+i={I}+{i}")
            if best is not None and nu > best[0]:
                break
            if self.use_tree:
                self.tree.create_node(f"nu={nu}─({i + 1})", ' '.join(map(str, I + [i])),
                                  parent=' '.join(map(str, I)))
            if self.v:print(f"to {I + [i]}")
            best = self.solve(I + [i], I_add - {i}, best, first, last)
            k+=1
        if self.use_tree:
            while k<len(nu_beta_i):
                nu, beta, i = nu_beta_i[k]
                self.tree.create_node(f"nu={nu}─({i + 1})x", ' '.join(map(str, I + [i])),
                                      parent=' '.join(map(str, I)))
                k+=1
        return best

    def get_first(self):
        """Поиск предварительного рекорда"""
        if self.first_I is not None: return self.first_I
        R_ = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            R_[i][i] = 1.0
            for j in range(i + 1, self.n):
                if self.P[i][j] > self.P[j][i]:
                    R_[i][j] = 1.0
                elif self.P[i][j] == self.P[j][i]:
                    R_[i][j] = 0.5
        kappa = dict()
        for i in range(self.n):
            k = sum(R_[i])
            if k not in kappa.keys():
                kappa[k] = []
            kappa[k].append(i)
        kappa_sort = sorted(kappa.items())
        if self.v: print(f'get first={kappa_sort}')
        K=1
        for ks in kappa_sort:
            K*=factorial(len(ks[1]))
        if self.v:print(f"num of lambda={K}={'*'.join([f'{len(ks[1])}!' for ks in kappa_sort])}")
        if K>10000:
            nus=[]
            for i in range(10000):
                I=[]
                for k,p in kappa_sort:
                    shuffle(p)
                    I+=p
                nus.append(self.get_nu(I[:-1]))
            self.first_I = [min(nus),[]]
        else:
            prods=product(*[list(permutations(i[1])) for i in kappa_sort])
            nus=[]
            st=time.time()
            i=0
            for prod in prods:
                nus.append(self.get_nu(sum([list(p) for p in prod], start=[])[:-1]))
            nu = min(nus)
            ft=time.time()
            print(ft-st)
            self.first_I=[nu,[]]
            if self.v: print(self.first_I)
            print(self.first_I)
        return self.first_I


def read_file(filename='test.txt'):
    """Чтение из файла формата:
    3 4                 # n и m через пробел или пустая строка (можно вычислить n,m по следующему вводу)
    1.0 0.5 1.5 1.0     # m весов (дробные числа) через пробел или пустая строка если они равны c_1=...=c_m=1.0
    1 2 3               # мнение 1-го эксперта (перестановка индексов i=1,2,...,n записанная через пробел)
    2 3 1               # мнение 2-го эксперта
    1 3 2
    3 2 1               # мнение m-го эксперта

    Возвращает:
        r, c, n, m
        r=[r1,...,rm], где rt=[i1, i2,...,in]-линейный порядок ik=1,n
        c=[c1,...,cm], где ct-float, sum(c)=m
        n,m - int
    """
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

    solver = MatrixSolve(n, m, P)
    ans = solver.solve(forced_down=forced_down, first=first, last=last)
    finish = time.time()
    ans_str = f'Значение функционала:{ans[0]}\nНайденные линейные порядки:\n'
    for a in ans[1]:
        ans_str += f"\t"
        for i in a[-1::-1]:
            ans_str += f"{i + 1}<"
        ans_str = ans_str[:-1]+'\n'
    ans_str += f'Время {finish - start}s.\nколичество count={count}\n'
    count=0
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

def print_solve(rR, c, n, m, forced_down=False, first=-1, last=-1, lin=True, v=False):
    global count
    count = 0
    start = time.time()
    if lin:
        R = get_R(rR, n, m)
    else:
        R = rR
    P = get_P(R, c, n, m)
    solver = MatrixSolve(n, m, P, v=v)
    ans = solver.solve2( forced_down=forced_down, first=first, last=last)
    finish = time.time()

    ans_str = f'Значение функционала:{ans[0]}\nНайденные линейные порядки:\n'
    for a in ans[1]:
        ans_str += f"\t"
        for i in a[-1::-1]:
            ans_str += f"{i + 1}<"
        ans_str = ans_str[:-1] + '\n'
    ans_str += f'Время {finish - start}s.\nколичество count={count}\n'
    count = 0
    print(ans_str)


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
    global count
    count=0
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
    ans_str+=f'D={[[d+1 for d in sd[1]] for sd in sD]}\n'

    ans_rename = []
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
    r, c, n, m = read_file('test_n20_m9_from_article.txt')
    print_solve(r, c, n, m, v=0)
    ans_str, *_=main(r,c,n,m)
    print(ans_str)
    ans_str, *_ = main(r, c, n, m, first=0)
    print(f'f=1')
    print(ans_str)
    #print_solve(r, c, n, m, v=0,first=0)
    ans_str, *_ = main(r, c, n, m, last=9)
    print(f'l=10')
    print(ans_str)
    #print_solve(r, c, n, m, v=0,last=9)