import random


def randomPermutation_k(n,k):
    """Перестановка k пар (без ограничения на индексы i,j)"""
    a = list(range(1, n + 1))
    for _ in range(k):
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        a[i], a[j] = a[j], a[i]
    return a


def get_test_k(n, m, k, v=False):
    """Генерация теста с перестановкой k пар альтернатив"""
    if v: print(f'n={n}')
    str_res = f'{n} {m}\n'
    c = [1] * m
    str_res += '\n'
    if v: print(f'k={k}')
    if v: print('r=[')
    r = []
    for i in range(m):
        perm = randomPermutation_k(n,k)
        r.append(perm)
        str_perm = ' '.join(map(str, perm))
        str_res += str_perm + '\n'
        if v: print(f'\t<{str_perm}>,')
    if v: print('  ]')
    return str_res, n, m, c, r


def get_test(n, m, is_c_eq=True, l=1, v=False):
    """Генерация теста со случайной перестановкой (shuffle)"""
    if v: print(f'n={n}')
    str_res = f'{n} {m}\n'
    c = [1] * m
    if is_c_eq:
        str_res += '\n'
    else:
        c = [random.random() for _ in range(m)]
        s = sum(c)
        c = [round(i / s * m, l) for i in c]
        c[-1] = round(m - sum(c[:-1]), l)
        if v: print(f'm={sum(c)}')
        c_ = ' '.join([str(i) for i in c])
        str_res += c_ + '\n'
        if v: print(f'c=[{c_}]')
    if v: print('r=[')
    r = []
    for i in range(m):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        r.append(perm)
        str_perm = ' '.join(map(str, perm))
        str_res += str_perm + '\n'
        if v: print(f'\t<{str_perm}>,')
    if v: print('  ]')
    return str_res, n, m, c, r


if __name__ == "__main__":
    m=9
    l=0.3
    for n in range(2, 41):
        k = max(round(n * l / 2), 1)
        f = open(f'tests/fast_test_k/test_n{n}_m{m}_fast_test_.txt', 'w',encoding='UTF-8')
        f.write(get_test_k(n, m,k,v=1)[0])
        f.write(f'\nk={k} пар переставлено\n')
        f.write(f'\n(ожидалось {l*100:.3f}% перестановок')
        f.write(f'\n получилось {k/n*200:.3f}% - до 2k элементов из n переставлены)')
        f.close()
