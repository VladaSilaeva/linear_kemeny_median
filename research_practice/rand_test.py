import random
import numpy as np

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


def get_test_k(n, m, k, v=False):
    """Генерация теста с перестановкой k пар альтернатив"""
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
    m=9
    l=0.3
    for n in range(2, 42):
        k = max(round(n * l / 2), 1)
        f = open(f'tests/test_n{n}_m{m}.txt', 'w',encoding='UTF-8')
        f.write(get_test_k(n, m, k, v=1)[0])
        f.write(f'\nk={k} растояния от\n')
        f.write(f'\n(ожидалось {l*100:.3f}% перестановок')
        f.write(f'\n получилось {k/n*200:.3f}% - до 2k элементов из n переставлены)')
        f.close()