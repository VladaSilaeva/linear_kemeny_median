"""Для вычисления времени выполнения с ускорением и без"""
from matrix_solve import main, read_file, count,counting
import time
import json
import os

import datetime
from rand_test import get_test
from prettytable import PrettyTable


counting=False
def get_ans(r,c,n,m,forced_down=False,is_beta_sort=False):
    print(f"n={n}, m={m}, f_d={forced_down}, is_b_s={is_beta_sort}:\tstart={datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S%f')}")
    start = time.time()
    ans_s, ans, solver, tree, count_ = main(r, c, n, m, v=0, use_tree=False,forced_down=forced_down,is_beta_sort=is_beta_sort)
    finish = time.time()
    t = finish - start
    print(f"\tend={datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S%f')}")
    print(f"\ttime={t}")
    print(f"\tcount={count_}")
    D=-1
    if forced_down:
        D=solver.first_I[0]
    return t,count_,ans,D

m = 9
is_new=False
date=""
name_dir="tests/fast_test"
if is_new:
    date=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S%f')
    name_dir=f"tests/fast_test{date}"
    os.mkdir(name_dir)
    for n in range(2, 26):
        f = open(f'{name_dir}/test_n{n}_m{m}_{date}.txt', 'w')
        f.write(get_test(n, m)[0])
        f.close()
T=[]
try:
    for n in range(2, 8):#2-26
        r, c, n, m = read_file(f'{name_dir}/test_n{n}_m{m}_fast_test{date}.txt')
        t=[]
        T.append([[r,c,n,m]])
        t = get_ans(r,c,n,m,forced_down=True,is_beta_sort=True)
        T[-1].append(t)
        t=[]
        t=get_ans(r,c,n,m,forced_down=False,is_beta_sort=True)
        T[-1].append(t)
        t = []
        t=get_ans(r,c,n,m,forced_down=True,is_beta_sort=False)
        T[-1].append(t)
        t = []
        t=get_ans(r,c,n,m,forced_down=False,is_beta_sort=False)
        T[-1].append(t)
        t = []
        print(T[-1])
finally:
    with open(f"tests/fast_result{date}.json", "w") as write_file:
        if len(t):
            T[-1].append(t)
        json.dump(T, write_file)


