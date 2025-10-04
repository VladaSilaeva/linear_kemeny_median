import datetime
import json
import os

#import PySimpleGUI as sg
from prettytable import PrettyTable

from matrix_solve import get_R, get_P, get_C, scc
from rand_test import get_test_k, get_test



def rand_case(N, M, k=-1,Ni=-1,Ni_=0):
    """Прогон одного теста:
    разбивает на компоненты,
    возвращает размеры компонент,
    сохраняет тест если размер максимальной компоненты ni<Ni или ni>Ni_
    """
    if k!=-1:str_test, n, m, c, r = get_test_k(N, M,k)
    else: str_test, n, m, c, r = get_test(N, M)
    P = get_P(get_R(r, n, m), c, n, m)
    C = get_C(P, n)
    D = scc(C)
    l = len(D)
    ni = max([len(Di) for Di in D])
    D_len = [(d, len(d)) for d in D]
    D_len.sort(key=lambda el: -el[-1])
    str_res = f'{l}:{" ".join([str(n) for d, n in D_len])}'
    if ni < Ni:
        date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
        fi = open(f'{name_dir}/n{N}_m{M}_ni{ni}_D{l}_{date_str}.txt', 'w')
        fi.write(str_test)
        fi.write(f'\nlen(D)={l}, ni={ni}\nD={[d for d, n in D_len]}')
        fi.close()
    if Ni_ and ni>Ni_:
        date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
        fi = open(f'{name_dir}/n{N}_m{M}_ni{ni}_D{l}_{date_str}.txt', 'w')
        fi.write(str_test)
        fi.write(f'\nlen(D)={l}, ni={ni}\nD={[d for d, n in D_len]}')
        fi.close()
    return str_res, tuple([n for d, n in D_len])

if __name__ == "__main__":
    if not os.path.exists('analysis'):
        os.mkdir("analysis")
    K = 100000
    N = 35
    M = 5
    k_ = 1
    for k_ in range(18,36):
        for M in (3,5,7,9):
            date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
            name_dir = f'analysis/analysis_k{K}_{k_}_n{N}_m{M}_{date_str}'
            os.mkdir(name_dir)
            file_name = f'{name_dir}/data'
            d = dict()
            for i in range(K):
                #sg.one_line_progress_meter(f'Analysis progress N{N} M{M} k{k_}', i + 1, K) # красивое окошко с прогрессом
                if i % 1000 == 0: # вывод прогресса на консоль
                    date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
                    print(f'm{M} k{k_}\ti={i}\t{date_str}')
                str_res, d_l = rand_case(N, M, k_)
                if d_l not in d.keys():
                    d[d_l] = 0
                d[d_l] += 1
            d_k = sorted(d.items(), reverse=True)

            table = [['count', 'lens of Di']]
            tab = PrettyTable(table[0])
            tab.align[table[0][0]] = "r"
            tab.align[table[0][1]] = "l"

            f = open(file_name + '.txt', 'w')
            for d, k in d_k:
                table.append([k, d])
            tab.add_rows(table[1:])
            f.write(tab.get_string())
            f.close()
            with open(file_name + ".json", "w") as write_file:
                json.dump([[d, k] for d, k in d_k], write_file)
