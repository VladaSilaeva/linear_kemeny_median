import random


def randomPermutation(n):  # есть shaffle(list)
    a = list(range(1, n + 1))
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        a[i], a[j] = a[j], a[i]
    return a


def get_test(n, m, is_c_eq=True, l=1, v=False):
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
        perm = randomPermutation(n)
        r.append(perm)
        str_perm = ' '.join(map(str, perm))
        str_res += str_perm + '\n'
        if v: print(f'\t<{str_perm}>,')
    if v: print('  ]')
    return str_res, n, m, c, r


def get_test2(n, m, is_c_eq=True, l=1, v=False):
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
    for n in range(2,26):
        f = open(f'tests/test_n{n}_m{m}_fast_test.txt', 'w')
        f.write(get_test(n, m)[0])
        f.close()
