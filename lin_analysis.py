import datetime
import json
import os
import time

from tqdm import tqdm
from prettytable import PrettyTable

from matrix_solve import get_P, get_C, scc, MatrixSolve
from research_practice_rand_test import get_test_dist_k
from rand_test import get_test_perm_k, get_test
from decomposition_solve import dec_solve, get_lin_order_matrix

get_test_func = {"M1": get_test, "M2": get_test_perm_k, "M3": get_test_dist_k}


def rand_case(N, M, k=-1, model="M3", name_dir="name_dir",matrix_subsolve=True):
    """Прогон одного теста:
    генерирует теста функцией get_test_k
    получает всевозможные линейные порядки функцией dec_solve
    замеряет время работы dec_solve
    """
    str_test, n, m, c, r = get_test_func[model](N, M, k)

    R=[]
    if model == "M3":
        R=r
    else:
        for r_i in r:
            R.append(get_lin_order_matrix(r_i, ind_start=1))

    P = get_P(R, c, n, m)
    graph = [[],[]]
    start = time.perf_counter()
    res = dec_solve(n, m, P, matrix_subsolve, graph=graph)
    end = time.perf_counter()
    d_l,_ = graph

    exec_time = end - start
    str_res = f'{len(res)}: {res}'

    return str_res, [res, exec_time, d_l]

from read_lin_analysis import find_dir

if __name__ == "__main__":
    K = 10
    N = 35
    M = 5
    k_ = 1
    model = "M3"
    matrix_subsolve = False
    name_dir0=f"lin_analysis_{model}"
    if not matrix_subsolve:
        name_dir0 += "_graph"
    if not os.path.exists(name_dir0):
        os.mkdir(name_dir0)
    for k_ in range(10, 100, 10):
        for M in (8,7,):
            file = find_dir(K, N, M, model=model,matrix_subsolve=matrix_subsolve, k=k_, v=0)
            if file is not None:
                continue
                print()
            date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
            if model == "M1":
                name_dir = name_dir0+f'/lin_analysis_k{K}_n{N}_m{M}_{date_str}'
            else:
                name_dir = name_dir0+f'/lin_analysis_k{K}_{k_}_n{N}_m{M}_{date_str}'
            os.mkdir(name_dir)
            file_name = f'{name_dir}/'

            exec_times = []  # время исполнения теста
            n_max = []  # размер макс компоненты для теста
            index0 = [0] * N  # количество случаев, когда лидер оказывается на i-том месте (sum >=K)
            len_ans = dict()  # количество тестов, когда получилось x ответов (sum == K)
            print(f'n={N}, m={M}, k={k_}')
            for i in tqdm(range(K)):
                str_res, ans = rand_case(N, M, k=k_, model=model, matrix_subsolve=matrix_subsolve)
                res, exec_time, d_l = ans
                for res_i in res:
                    index0[res_i.index(0)] += 1
                l = len(res)
                if l not in len_ans.keys():
                    len_ans[l] = 0
                len_ans[l] += 1
                exec_times.append(exec_time)
                n_max.append(d_l[0])
            print()
            with open(file_name + "index0.json", "w") as write_file:
                json.dump(index0, write_file)
            with open(file_name + "exec_times.json", "w") as write_file:
                json.dump(exec_times, write_file)
            with open(file_name + "n_max.json", "w") as write_file:
                json.dump(n_max, write_file)
            len_ans_sort = sorted(len_ans.items())
            with open(file_name + "len_ans.json", "w") as write_file:
                json.dump([[k, i] for k, i in len_ans_sort], write_file)
