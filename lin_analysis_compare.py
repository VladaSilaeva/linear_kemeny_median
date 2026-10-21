import time
import datetime
import json
import os
from tqdm import tqdm
from prettytable import PrettyTable
import numpy as np

from measure_time import measure_time_with_result
from matrix_solve import get_P, get_C, scc, MatrixSolve
from research_practice_rand_test import get_test_dist_k
from rand_test import get_test_perm_k, get_test
from decomposition_solve import dec_solve, get_lin_order_matrix
from read_lin_analysis import find_dir

get_test_func = {"M1": get_test, "M2": get_test_perm_k, "M3": get_test_dist_k}

def get_test(N, M, k=-1, model="M3"):
    str_test, n, m, c, r = get_test_func[model](N, M, k)

    R=[]
    if model == "M3":
        R=r
    else:
        for r_i in r:
            R.append(get_lin_order_matrix(r_i, ind_start=1))

    P = get_P(R, c, n, m)

    return n,m,P

def rand_case(n,m,P, matrix_subsolve=True):
    """Прогон одного теста:
    генерирует теста функцией get_test_k
    получает всевозможные линейные порядки функцией dec_solve
    замеряет время работы dec_solve
    """
    graph = [[],[]]
    start = time.perf_counter()
    res = dec_solve(n, m, P, matrix_subsolve, verbose=0, graph=graph)
    end = time.perf_counter()

    exec_time = end - start
    str_res = f'{len(res)}: {res}'

    return str_res, res, exec_time, graph


if __name__ == '__main__':
    K = 10
    N = 35
    M = -1
    k_ = -1

    model = "M3"
    name_dir0=f"lin_analysis_{model}_compare"

    timeout = 60.0 * 5

    if not os.path.exists(name_dir0):
        os.mkdir(name_dir0)

    tests_all = []  # тесты, которые стоит проверить

    for k_ in range(80, N*(N-1)//2, 10):
        for M in (5,6,7,8):
            r = -1
            date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
            if model == "M1":
                name_dir = name_dir0+f'/lin_analysis_k{K}_n{N}_m{M}_{date_str}'
            elif model == "M2" or r < 0:
                name_dir = name_dir0 + f'/lin_analysis_k{K}_k{k_}_n{N}_m{M}_{date_str}'
            else:
                name_dir = name_dir0+f'/lin_analysis_k{K}_r{r}_n{N}_m{M}_{date_str}'
            os.mkdir(name_dir)
            file_name = f'{name_dir}/'

            exec_times = []  # время исполнения теста
            n_max = []  # размер макс компоненты для теста
            tests = []
            print('='*100)
            r = k_*2/(N*(N-1))*100
            print(f'n={N}, m={M}, k={k_} (r={r:.0f}%)')
            print('=' * 100)
            for i in range(K):
                n, m, P = get_test(N, M, k_, model)
                exec_times.append([])
                n_max.append(None)
                for matrix_subsolve in (True, False):
                    print(f"{'matrix' if matrix_subsolve else 'graph '}",end=" ")
                    elapsed, result = measure_time_with_result(rand_case, timeout, args=(n,m,P,matrix_subsolve))
                    if elapsed is None:
                        print(f"Таймаут, результат не получен")
                        exec_times[-1].append(None)
                        print(result)
                    else:
                        str_res, res, exec_time, graph = result
                        print(f"Время: {exec_time:.3f} сек, \n\tрезультат: {str_res}")
                        print(f"\t\tSCC={graph[0]}"+f"\n\t\t(num of edges, num of cycles)={graph[1]}"*(not matrix_subsolve))
                        exec_times[-1].append(exec_time)
                        n_max[-1] = graph[0][0]
                t1, t2 = exec_times[-1]
                EPS=0.1
                if t1 is None or t2 is None or abs(t2-t1)>EPS:
                    print(f"n={n},m={m},k={k_},t=[{t1},{t2}],n_max={n_max[-1]}")
                    print(P)
                    P=(np.array(P)).tolist()
                    tests.append((n,m,k_,P,t1,t2, n_max[-1]))
                print('-'*100)



            with open(file_name + "exec_times.json", "w") as write_file:
                json.dump(exec_times, write_file)
            with open(file_name + "n_max.json", "w") as write_file:
                json.dump(n_max, write_file)
            with open(file_name + "/tests.json", "w") as write_file:
                json.dump(tests, write_file)
            tests_all += tests
    with open(name_dir0 + "/tests.json", "w") as write_file:
        json.dump(tests_all, write_file)