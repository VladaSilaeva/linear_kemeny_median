import time
import sys
from itertools import permutations, product
from math import factorial
from random import shuffle

import graphviz
import numpy as np
from treelib import Tree # для красивого вывода дерева решений

count = 0
counting=False
source="graph-output/main" # graph-output - имя будущей папки, для хранения графа main.gv и его изображения main.gv.png
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
        self.D_min=0 # минимальное значение функции D на множестве ВСЕХ бинарных отношений

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
        self.D_min=sum(self.alpha)
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

    def solve(self, I=None, I_add=None, best=None, first=-1, last=-1, is_beta_sort=True):
        """Функция рекурсивного обхода дерева решений для метода ветвей и границ.
        Принимает:
            first=i_f - индекс закрепленного на 1-м месте или -1, если 1-й не закреплен
            last=i_l - индекс закрепленного на n-м месте или -1, если последний не закреплен
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
            if counting:print(count)
        if I is None: # начало решения
            nu_str=""
            if self.use_tree:
                self.tree = Tree()
                self.tree_countD=0
            if self.v:print("solution start")
            I=[]
            I_add=set(range(self.n))
            if first != -1: # если первый закреплен
                if self.use_tree:self.tree.create_node(f"({first+1})", f"{first}")
                if self.v:print(f"first={first}")
                I.append(first)
                I_add.remove(first)
            else:
                if self.use_tree:self.tree.create_node(nu_str+"(-)", "")
            if last != -1:
                if self.v:print(f"last={last}")
                I_add.remove(last)
        if self.v:print(f"I={I}, I_add={I_add}, best={best}")
        nu_beta_i = [(self.get_nu(I+[i]),self.get_beta(I+[i]),i) for i in I_add]
        if is_beta_sort:
            nu_beta_i.sort(key=lambda x:(x[0], x[2]))
        else:
            nu_beta_i.sort()
        if self.v:print(f"nu={nu_beta_i}")
        if len(I) == self.n-2: # если уровень дерева n-2 (ниже спускаться не нужно)
            if self.v:print(f"level n-2")
            if last!=-1: end=last
            else: end=nu_beta_i[1][2]
            if best is None: # первый рекорд
                best = [nu_beta_i[0][0], [I + [nu_beta_i[0][2],end]]]
                if self.use_tree:
                    self.tree.create_node(f"nu={nu_beta_i[0][0]}─[{nu_beta_i[0][2]+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[0][2]])),
                                                        parent=' '.join(map(str, I)))
                    self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[0][0]}─[{end+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[0][2],end])),
                                                        parent=' '.join(map(str, I + [nu_beta_i[0][2]])))
                if self.v:print(f"first best={best}")
            elif best[0] == nu_beta_i[0][0]: # добавляем послед-ть0 к существующему рекорду
                best[1].append(I + [nu_beta_i[0][2],end])
                if self.use_tree:
                    self.tree.create_node(f"nu={nu_beta_i[0][0]}─[{nu_beta_i[0][2] + 1}]",
                                          ' '.join(map(str, I + [nu_beta_i[0][2]])),
                                          parent=' '.join(map(str, I)))
                    self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[0][0]}─[{end+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[0][2],end])),
                                                        parent=' '.join(map(str, I + [nu_beta_i[0][2]])))
                if self.v:print(f"add to best {best[1][-1]}")
            elif best[0] > nu_beta_i[0][0]:  # послед-ть0 - новый рекорд
                if self.v:print(f"{best[0]}>{nu_beta_i[0][0]} => ", end='')
                best = [nu_beta_i[0][0], [I + [nu_beta_i[0][2],end]]]
                self.tree_countD+=1
                if self.use_tree:
                    self.tree.create_node(f"nu={nu_beta_i[0][0]}─[{nu_beta_i[0][2] + 1}]",
                                          ' '.join(map(str, I + [nu_beta_i[0][2]])),
                                          parent=' '.join(map(str, I )))
                    self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[0][0]}─[{end+1}]",
                                                        ' '.join(map(str, I + [nu_beta_i[0][2],end])),
                                                        parent=' '.join(map(str, I + [nu_beta_i[0][2]])))
                if self.v:print(f"new best={best}")
            else: # рекорд лучше ()
                return best
            if last==-1 and best[0] == nu_beta_i[1][0]: # случай если nu1==nu2
                best[1].append(I + [nu_beta_i[1][2],nu_beta_i[0][2]])
                if self.use_tree:
                    self.tree.create_node(f"nu={nu_beta_i[1][0]}─[{nu_beta_i[1][2] + 1}]",
                                          ' '.join(map(str, I + [nu_beta_i[1][2]])),
                                          parent=' '.join(map(str, I)))
                    self.tree.create_node(f"D{self.tree_countD}={nu_beta_i[1][0]}─[{nu_beta_i[0][2]+1}]",
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
            i=0
            for prod in prods:
                nus.append(self.get_nu(sum([list(p) for p in prod], start=[])[:-1]))
            nu = min(nus)
            self.first_I=[nu,[]]
            if self.v: print(self.first_I)
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
    """Получение списка матриц отношений R
        r - список линейных порядков
        n - количество альтернатив
        m - количество экспертов
        Возвращает
            R=[R1,...,Rm], где Rt=||r_ij||_nxn, где r_ij in {0,1}"""
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
    """Получение матрицы P по списку матриц отношений R и весов важности экспертов c
        R - список матриц отношений
        c - список весов важности
        n - количество альтернатив
        m - количество экспертов
        v - verbal - выводить ли на консоль?
        Возвращает:
            P=np.array(n,n)=||p_ij||, где p_ij - float"""
    if v:
        print(f'n={n}, m={m}')
        print('R=')
        for k in range(m):
            print("r = <", r[k], ">")
            print(*R[k], sep='\n')
            print('-' * n * 2)
    P = np.zeros((n, n))
    for k in range(m):
        P += c[k] * np.array(R[k])
    if v: print(f'P\n{P}\n')
    return P


def main(rR, c, n, m, solver=None, forced_down=False, first=-1, last=-1, lin=True, v=0, use_tree=True, is_beta_sort=False):
    """Оболочка для запуска решения solve
    Обеспечивает красивый вывод решения, засекает время выполнения
    Принимает:
        rR: r, если lin=True, R иначе
        c - веса экспертов
        n,m - количество альтернатив и экспертов соответственно
        solver - может использовать старый решатель
        forced_down - применять ли ускорение метода
        first, last - индексы альтернатив, закрепленных на концах, или -1 если не закреплен
        lin - линейный ли ввод
        v - verbal будет ли вывод на консоль
    Возвращает:
        ans_str - строки ответа
        ans=solve(...) - best от заданных параметров
        solver - решатель с заданными параметрами
        str(solver.tree) - строка дерево решений"""
    global count
    count = 0


    if solver is None:
        R = []
        if lin:
            R = get_R(rR, n, m)
        else:
            R = rR
        P = get_P(R, c, n, m)
        if v: print('P', P)
        solver = MatrixSolve(n, m, P, v=v, use_tree=use_tree)
    best=None
    prep=time.time()
    if v:print(f'----------before prep {prep}')
    if forced_down:  # найти предварительный рекорд
        solver.get_first()
        if v: print(f"get first nu={solver.first_I[0]}")
        nu_str = f"f_nu={solver.first_I[0]}--"
        best = [solver.first_I[0], []]
    start = time.time()
    if v:print(f'----------start {start}: {start-prep}')
    ans = solver.solve(best=best, first=first, last=last, is_beta_sort=is_beta_sort)
    finish = time.time()
    if v or counting:print(f'----------finish {finish}: {finish-prep}={start-prep}+{finish-start}')
    # создание вывода для оконного приложения
    ans_str=""
    if first==-1 and last==-1:
        ans_str += f"Минимальное значение функции      D = {solver.D_min}\n"
    if forced_down:
        ans_str += f'Значение предварительного рекорда \u010e = {solver.first_I[0]}\n'
    nameD="D_"
    if first!=-1:
        nameD+="i"
    if last!=-1:
        nameD+="j"
    if len(nameD)==2:
        nameD+="LO"
    nameD=" "*(6-len(nameD))+nameD
    ans_str += f'Минимальное значение функции {nameD} = {ans[0]}\nНайденные линейные порядки:\n'
    for a in ans[1]:
        ans_str += f"\t"
        for i in a[-1::-1]:
            ans_str += f"{i + 1}<"
        ans_str = ans_str[:-1] + '\n'
    ans_str += f'Время {finish - start:.3f}s.\nколичество пройденных вершин={count}\n'

    if v:
        print(f'f_d={forced_down}, f={first}, l={last}, ans={ans}')
        I = ans[1][0][:-1]
        print(f'{solver.get_nu(I)} {I}')
    tree_str="no tree"
    if use_tree and count>1000:
        s=input("show tree?('y' or other)")
        if s=="y":
            tree_str = str(solver.tree)
        else:
            tree_str=f"no tree count={count}"
    else:
        tree_str = str(solver.tree)
    c=count
    count = 0

    return ans_str, ans, solver, tree_str, [finish-start, c]


def scc(C, v=0):
    """Разбиение мажорантного графа с матрицей связности C на компоненты связности
    Возвращает [D1, D2,...,Dk], где Di=[i1,...] - компонента связности, i_l=1,n"""
    n = len(C)
    C_bool = np.array(np.matrix(C), dtype=bool)
    if v:   print('C', C_bool)
    A = [np.array(np.eye(n), dtype=bool)]
    for i in range(n):
        A_i = np.matmul(A[-1], C_bool)
        if v:   print(f'A{i}', np.array(A_i, dtype=int))
        A.append(A_i)
    T_D = np.array(sum(A), dtype=bool)
    if v:   print('T_D', np.array(T_D, dtype=int))
    S_D = T_D & T_D.T
    if v:   print('S_D', np.array(S_D, dtype=int))
    D = []
    k = -1
    is_in_D = [0] * n
    while sum(is_in_D) != n:
        for i in range(n):
            if is_in_D[i]: continue
            D.append([])
            for j in range(n):
                if is_in_D[j]: continue
                if S_D[i][j]:
                    D[-1].append(j)
                    is_in_D[j] = 1
    if v:         print(D)
    return D


def get_C(P, n=None):
    """Построение матрицы связности C для мажорантного графа из P (размерность nxn)
    Возвращает:
        [[cij for j in range(n)] for i in range(n)], где cij in {0,1}"""
    if n is None: n = len(P)
    C = [[0] * n for i in range(n)]
    for i in range(len(C)):
        for j in range(len(C[0])):
            ci = P[i][j] - P[j][i]
            if ci > 0:
                C[i][j] = 1
    return C


def special_case(r, c, n, m, v=0):
    """Решение в особом случае с помощью разделения на компоненты связности
    Также строит картинку конденсации графа
    Возвращает:
        ans_str - строку ответа
        [best, res], где best - список значений рекорда для подзадач, res - множество решений
    """
    global count
    count = 0

    start = time.time()


    P = get_P(get_R(r, n, m), c, n, m, v)
    solver = MatrixSolve(n, m, P, v=0, use_tree=False)
    ans_str = f"Минимальное значение функции    D = {solver.D_min}\n"
    C = get_C(P, n)
    if v:
        print('C')
        print(*C, sep='\n', end='\n\n')
    D = scc(C, v=v)
    k = len(D)
    if v: print(f'k={k}-----------------------')
    C_Di = [[0] * k for _ in range(k)]  # граф конденсации, компоненты Di стали вершинами
    for i in range(k):
        for j in range(k):
            C_Di[i][j] = C[D[i][0]][D[j][0]]
    if v:
        print('C_Di')
        print(*C_Di, sep='\n', end='\n\n')
        print('D')
        print(D)
    sorted_Di = [(sum(C_Di[i]), D[i]) for i in range(k)]  # подсчет количества исходящих ребер в графе конденсации
    sorted_Di.sort()  # выстраиваем компоненты в линейный порядок
    if v: print(sorted_Di)

    if v != -1:  # построение конденсации графа (картинка хранится в sourse+".gv.png")
        graph = graphviz.Digraph(name=source.split('/')[1], engine='fdp')
        graph.attr(compound='true')
        for i in range(n):
            graph.node(f'a{i}', label=f'{i + 1}')
        for k1 in range(k):
            with graph.subgraph(name=f'cluster{k1}') as cluster:
                for i in sorted_Di[k1][1]:
                    cluster.node(f'a{i}')
                    for j in sorted_Di[k1][1]:
                        if C[i][j]:
                            cluster.edge(f'a{i}', f'a{j}')
                cluster.attr(label=f'D{k1 + 1} ({len(sorted_Di[k1][1])})')
        for k1 in range(k):
            for k2 in range(k):
                if C[sorted_Di[k1][1][0]][sorted_Di[k2][1][0]]:
                    graph.edge(f'cluster{k1}', f'cluster{k2}', color='blue')
        if v: print(graph.source)
        graph.format = 'png'
        graph.render(directory=source.split('/')[0], view=bool(v))

    ans_rename = []
    best = []
    counts=[]
    for s, Di in sorted_Di:
        if v: print(Di)
        ni = len(Di)
        Pi = [[P[i][j] for j in Di] for i in Di]
        count=0
        solver_i = MatrixSolve(ni, m, Pi, v=0)
        if v:
            print('ni Pi')
            print(ni)
            print(Pi)
        if ni == 1:
            a = [0, [[0]]]
        else:
            solver_i.get_first()
            best = [solver_i.first_I[0], []]
            a = solver_i.solve(best=best)
        a_r = []
        counts.append(count)
        best.append(a[0])
        for a_ in a[1]:  # возвращение из индексов подзадач ([1],[1,2],[1]) к индексам основной задачи ([3],[4,1],[2])
            ai_r = []
            if v: print(a_)
            for i in a_:
                ai_r.append(Di[i])
            a_r.append(ai_r)
        ans_rename.append(a_r)
    if v: print('ren', *ans_rename, sep='\n\t')

    # генерация всех вариантов перестановок: [{[1,2], [2,1]}, {[3,4], [4,3]}] -> 1234,1243,2134,2143
    res = get_res_from_parts(ans_rename)
    finish = time.time()
    best=solver.get_nu(res[0][:-1])
    ans_str += f'Минимальное значение функции D_LO = {best}\n'
    ans_str += f'Разбиение на компоненты: {[[d + 1 for d in sd[1]] for sd in sorted_Di]}\n'
    ans_str += f'Решения:\n'
    for a in res:
        ans_str += f"\t"
        for i in a[-1::-1]:
            ans_str += f"{i + 1}<"
        ans_str = ans_str[:-1] + '\n'
    ans_str += f'Время {finish - start:.3f}s.\n'
    count=0
    return ans_str, [best, res], solver, [finish-start,counts]


def get_res_from_parts(parts):
    res = [[]]
    for part in parts:
        res1 = []
        for r in res:
            for p in part:
                res1.append(list(r) + list(p))
        res = res1
    return res

if __name__ == "__main__":
    counting=True
    test_name='tests/test_n4_ni3_D2_m3_2025_05_06_03_13_51921570.txt'
    if len(sys.argv) > 1:
        test_name=sys.argv[1]
    r, c, n, m = read_file(test_name)
    ans_s, ans, solver, tree, count_ = main(r, c, n, m, v=0,is_beta_sort=0,use_tree=0)
    print(ans_s)
    #print(tree)
    #print(sum(solver.alpha))
    input("next?1")
    ans_s, ans, solver, tree, count_ = main(r, c, n, m, forced_down=True, v=0)
    print(ans_s)
    #print(tree)
    input("next?2")
    if m % 2:  # and c==[1.0]*m
        print(special_case(r, c, n, m, 5)[0])
        input("next?")
    ans = []
    print("-----------------------------------------")
    for i in range(n):
        print(f'last={i + 1}')
        ans_str, a, solver, tree, count_ = main(r, c, n, m, solver=solver, last=i)
        print(ans_str)
        print(tree)
        ans.append(a)
        input("next?")
    print('\n'.join([f'r_last({i + 1})={ans[i]}' for i in range(n)]))
    ans = []
    for i in range(n):
        print(f'first={i + 1}')
        ans_str, a, solver, tree, count_ = main(r, c, n, m, solver=solver, first=i)
        print(ans_str)
        print(tree)
        ans.append(a)
    print('\n'.join([f'r_first({i + 1})={ans[i]}' for i in range(n)]))
    ans = dict()
    rr = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            print(f'first={i + 1}, last={j + 1}')
            ans_str, a, solver, tree, count_ = main(r, c, n, m, solver, first=i, last=j)
            print(ans_str)
            print(tree)
            rr[i][j] = a[0]
            ans[(i, j)] = a
    print('\n'.join([f'r({k[0] + 1},{k[1] + 1})={ans[k]}' for k in ans.keys()]))
    print(*rr, sep='\n')
    for i in range(n):
        for j in range(i + 1, n):
            rr[i][j], rr[j][i] = rr[j][i] / rr[i][j], rr[i][j] / rr[j][i]
