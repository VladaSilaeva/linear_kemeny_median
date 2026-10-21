import random
import numpy as np
import json

def randomDistance_k(n, k):
    """Меняем k элементов матрице"""
    # Создаем исходную матрицу R с помощью NumPy
    R = np.triu(np.ones((n, n), dtype=int))
    # Генерируем все возможные пары (i, j) выше главной диагонали
    rows, cols = np.triu_indices(n, k=1)
    pairs = list(zip(rows, cols))
    # Проверяем, что k не превышает количество доступных пар
    if k > len(pairs):
        raise ValueError(f"k={k} превышает количество элементов выше диагонали ({len(pairs)})")
    # Случайно выбираем k пар
    selected_indices = random.sample(range(len(pairs)), k)
    # Меняем выбранные элементы
    for idx in selected_indices:
        i, j = pairs[idx]
        R[i, j] = 1 - R[i, j]
        R[j, i] = 1 - R[j, i]
    return R


def get_test_dist_k(n, m, k, v=False):
    """Генерация теста с инверсией k элементов в матрице"""
    if v: print(f'n={n}')
    str_res = f'{n} {m}\n'
    c = [1] * m
    str_res += '\n'
    if v: print(f'k={k}')
    if v: print('r=[')
    R = []
    for i in range(m):
        if i==0:
            R_i = randomDistance_k(n,0)
        else:
            R_i = randomDistance_k(n, k)
        R.append(R_i)
        str_R_i = '\n '.join(map(str, R_i))
        str_res += str_R_i + '\n\n'
        if v: print(f'[{str_R_i}],')
    if v: print('  ]')
    return str_res, n, m, c, R


if __name__ == "__main__":
    m = 8
    for n in range(5, 40, 5):
        K = n * (n-1) // 2
        for k in range (K//10, K , K//10):
            file_name=f'tests/test_n{n}_m{m}_k{k}-'
            with open(file_name + "R.json", "w") as write_file:
                str_res, n, m, c, R =get_test_dist_k(n, m, k, v=False)
                R=(np.array(R)).tolist()
                json.dump(R, write_file)