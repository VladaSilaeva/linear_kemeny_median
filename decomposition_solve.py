import datetime
import json
import os
import time
from itertools import product
import numpy as np

from matrix_solve import get_P, get_C, scc, MatrixSolve, read_file, get_R
from cycle_remove import get_min_loop_split


def get_lin_order_matrix(lin_order, ind_start=0):
    n = len(lin_order)
    C = [[-1] * n for _ in range(n)]
    for i in lin_order:
        i -= ind_start
        C[i][i] = 1
        for j in range(n):
            if C[i][j] == -1:
                C[i][j] = 1
        for j in range(n):
            if C[j][i] == -1:
                C[j][i] = 0
    return C


def get_lin_extensions(C):
    """
    Возвращает список всех линейных порядков (топологических сортировок),
    совместимых с частичным порядком, заданным матрицей C.

    Параметры:
    C -- квадратная матрица (list of lists) из 0 и 1, где C[i][j] = 1 означает,
         что i должен предшествовать j.

    Возвращает:
    Список списков, каждый внутренний список -- один из возможных порядков
    (индексы элементов от 0 до n-1).
    """
    n = len(C)
    C_bool = [[bool(C[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        C_bool[i][i] = False
    # вычисляем in_degree для каждой вершины
    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if C_bool[i][j]:
                in_degree[j] += 1

    used = [False] * n  # отмечает уже размещённые вершины
    current = []  # текущий строящийся порядок
    result = []  # все найденные порядки

    def backtrack():
        if len(current) == n:
            result.append(current.copy())
            return

        # собираем кандидатов (не использованы и in_degree == 0)
        candidates = [v for v in range(n) if not used[v] and in_degree[v] == 0]
        if not candidates:  # тупик (например, цикл в графе)
            return

        for v in candidates:
            # добавляем вершину v в порядок
            used[v] = True
            current.append(v)

            # уменьшаем in_degree для всех последователей v
            for u in range(n):
                if C_bool[v][u]:
                    in_degree[u] -= 1

            backtrack()

            # откатываем изменения
            for u in range(n):
                if C_bool[v][u]:
                    in_degree[u] += 1
            current.pop()
            used[v] = False

    backtrack()
    return result


def dec_solve(n, m, P, matrix_subsolve=True, verbose=0, graph=None):
    """
    строит мажоритарный графа C
    разбивает на компоненты функцией scc,
    запускает для каждой КСС функцию:
        поиска медиан
        -(или разбиения циклов),
    объединяет решения в единый граф (всевозможными способами),
    расширяет частичного порядка до линейного,
    возвращает всевозможные линейные порядки
    """

    C = get_C(P, n)

    D = scc(C)

    l = len(D)
    D_len = [(d, len(d)) for d in D]
    D_len.sort(key=lambda el: -el[-1])
    if graph is not None:
        graph[0] = [n for d, n in D_len]
        graph[1] = [None for _ in D_len]

    C_sub_res = []

    for i, D_l in enumerate(D_len):
        Di, ni = D_l
        if verbose:
            print("Di=", Di)
        Ci_res = []
        loc_lin_orders = []
        if ni == 1:
            loc_lin_orders = [[0]]
            if graph is not None and not matrix_subsolve:
                graph[1][i] = [0, 0]
        elif matrix_subsolve:
            Pi = np.array([[P[i][j] for j in Di] for i in Di])
            if verbose > 0:
                print(f"Pi={Pi}\n")
            solver_i = MatrixSolve(ni, m, Pi, v=0)
            solver_i.get_first()
            best = [solver_i.first_I[0], []]
            _, loc_lin_orders = solver_i.solve(best=best)
        else:
            Ci = np.array([[C[i][j] for j in Di] for i in Di])
            if verbose > 0:
                print(f"Ci={Ci}\n")
            # разбиение контуров внутри Ci (несколько вариантов в Ci_res)
            cost, loc_lin_orders = get_min_loop_split(Ci, verbose=verbose, graph=graph, i=i)

        if verbose > 0:  print("lin_order")
        for lin_order in loc_lin_orders:
            if verbose > 0:  print(lin_order)
            Ci_res.append(np.array(get_lin_order_matrix(lin_order)))
        if verbose > 0:
            print("Ci_res:")
            for j in range(len(Ci_res)):
                print(f"{np.array(Ci_res[j])}\n{'or' * (j != len(Ci_res) - 1)}")
            print()
        C_sub_res.append(Ci_res)
    res = []
    for Csr_prod in product(*C_sub_res):
        newC = get_C(P, n)
        for D_l, Ci in zip(D_len, Csr_prod):
            Di, l = D_l
            for i in range(l):
                for j in range(l):
                    newC[Di[i]][Di[j]] = Ci[i][j]
        cur_res = get_lin_extensions(newC)
        for lin_order in cur_res:
            if lin_order not in res:
                res.append(lin_order)
    return res


from measure_time import measure_time_with_result


def measure_dec_solve(n, m, P, matrix_subsolve=True, verbose=0, graph=None):
    start = time.time()
    res = dec_solve(n, m, P, matrix_subsolve, verbose, graph)
    end = time.time()
    return res, end - start


if __name__ == "__main__":
    # test_name = "tests/test_n4_ni3_D2_m3_2025_05_06_03_13_51921570.txt"
    test_name = "tests/test_n15_m5.txt"
    # test_name="tests/test_n20_m9_from_article.txt"
    # test_name = "tests/test_n35_ni17_D3_m9_2025_06_05_23_24_49458764.txt"
    """r, c, n, m = read_file(test_name)
    P = get_P(get_R(r, n, m), c, n, m)
    solver = MatrixSolve(n, m, P)
    print(f"test={test_name}")
    print(f"P={P}\n")

    print("=== Matrix Solve ===")
    graph = [[],[]]
    start = time.time()
    res = dec_solve(n, m, P, 1, verbose=1,graph=graph)
    end = time.time()
    print(f"scc {graph[0]}")
    print(f'\ntime={end - start:.4f}')
    print(f'num of res: {len(res)}')
    for i, res_i in enumerate(res):
        print(f"{i+1}) D={solver.get_beta(res_i)}, {res_i}")


    print("=== Graph Solve ===")
    graph = [[], []]
    start = time.time()
    res = dec_solve(n, m, P, 0, verbose=1,graph=graph)
    end = time.time()
    print(f"scc {graph[0]}")
    print(f"(num of edges, num of cycles)={graph[1]}")
    print(f'\ntime={end - start:.4f}')
    print(f'num of res: {len(res)}')
    for i, res_i in enumerate(res):
        print(f"{i + 1}) D={solver.get_beta(res_i)}, {res_i}")
    """
    m = 8
    timeout = 60.0 * 5
    for n in range(5, 40, 5):
        K = n * (n - 1) // 2
        for k in range(K // 10, K, K // 10):
            test_name = f'tests/test_n{n}_m{m}_k{k}-R.json'
            print(f"test={test_name}")
            with open(test_name) as json_file:
                R = json.load(json_file)
            c = [1] * m
            P = get_P(R, c, n, m)
            solver = MatrixSolve(n, m, P)
            print("P=")
            for row in P:
                print(*row, sep='\t')

            for matrix_subsolve in (True, False):
                print()
                if matrix_subsolve:
                    print("=== Matrix Solve ===")
                else:
                    print("=== Graph Solve ===")
                graph = [[], []]
                elapsed, result = measure_time_with_result(dec_solve, timeout,
                                                           args=(n, m, P, matrix_subsolve, -1, graph))
                if elapsed is None:
                    if result is not None: print(result)
                    print(f'\ntime > {timeout:.4f} (timeout)')
                    continue
                start = time.time()
                res = dec_solve(n, m, P, matrix_subsolve, verbose=0, graph=graph)
                end = time.time()
                print(f"\nscc {graph[0]}")
                if not matrix_subsolve: print(f"(num of edges, num of cycles)={graph[1]}")
                print(f'\ntime={end - start:.4f}')
                print(f'num of res: {len(res)}')
                for i, res_i in enumerate(res):
                    print(f"\t{i + 1}) D={solver.get_beta(res_i)}, {res_i}")
            print('=' * 100)
