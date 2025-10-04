"""Для вычисления времени выполнения с ускорением и без"""
from matrix_solve import main, read_file, count, counting
import time
import json
import os

import datetime
from rand_test import get_test_k
from prettytable import PrettyTable

counting = False


def get_ans(r, c, n, m, forced_down=False, is_beta_sort=False):

    ans_s, ans, solver, tree, time_count = main(r, c, n, m, v=0, use_tree=False, forced_down=forced_down,
                                            is_beta_sort=is_beta_sort)
    print(f"\tend={datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S%f')}")
    time_, count_ = time_count
    print(f"\ttime={time_}")
    print(f"\tcount={count_}")
    D = -1
    if forced_down:
        D = solver.first_I[0]
    return round(time_,3), count_, D, ans


m = 9
is_new = False
date = "fast_test_1"
name_dir = "tests/test_presa"
if is_new:
    date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S%f')
    name_dir = f"tests/fast_test{date}"
    os.mkdir(name_dir)
    l = 0.3
    for n in range(2, 41):
        k = max(round(n * l / 2), 1)
        f = open(f'tests/fast_test_k/test_n{n}_m{m}_fast_test_.txt', 'w', encoding='UTF-8')
        f.write(get_test_k(n, m, k, v=1)[0])
        f.write(f'\nk={k} пар переставлено\n')
        f.write(f'\n(ожидалось {l * 100:.3f}% перестановок')
        f.write(f'\n получилось {k / n * 200:.3f}% - до 2k элементов из n переставлены)')
        f.close()
T = []
t = []

try:
    for n in range(2,8):  # 2-26
        print(n)
        r, c, n, m = read_file(f'{name_dir}/test_n{n}_m{m}_{date}.txt')
        T.append([[r, c, n, m]])
        t=get_ans(r, c, n, m, forced_down=False, is_beta_sort=False)
        T[-1].append(t)
        t = []
        t = get_ans(r, c, n, m, forced_down=True, is_beta_sort=False)
        T[-1].append(t)
        t = []
        print(T[-1])
finally:
    with open(f"tests/fast_result{date}.json", "w") as write_file:
        if len(t):
            T[-1].append(t)
        json.dump(T, write_file)

        tab=PrettyTable(['n','m','D','~D','T', 'Tf', 'Tf/T%', 'count delta'])
        t=[['r','c','n','m'],['time','count','-1','[D,[...]]'], ['time_fast','count_fast','~D','[D,[...]]']]
    for t in T:
        print(t)
        if len(t)!=3:break
        data, t1,t2=t
        r,c,n,m=data
        time_,count_,D,_=t1
        time_fast,count_fast,D,ans=t2
        D1=ans[0]
        proc=0
        if time_!=0:
            proc=round((time_fast)/time_*100,3)
        tab.add_row([n,m,D,D1,time_,time_fast,proc,count_-count_fast])
        if count_-count_fast>0:
            with open(f'{name_dir}/test_n{n}_m{m}_{date}.txt','r') as f:
                lines=f.readlines()
            with open(f'{name_dir}/_test_n{n}_m{m}_{date}.txt','w') as f:
                f.write(''.join(lines))
                f.write(f'\n\ncount_delta={count_-count_fast}')
    print(tab)
    with open(f"tests/fast_result{date}.txt", "w") as write_file:
        write_file.write(tab.get_string())