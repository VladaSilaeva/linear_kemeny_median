import random
import datetime

def get_special_case(N, M, Ni):
    """Генерирует и возвращает 'особый случай' теста:
    N - количество альтернатив
    M - количество экспертов
    Ni - верхняя грань для максимального размера компонент"""
    lst = list(range(1, N + 1))
    D = []
    while len(lst):
        print(f'lst={lst}')
        ni = random.randrange(1, min(Ni, len(lst) + 1))
        print(f'ni={ni}')
        # D.append(random.choices(lst, k=ni))
        D.append(list(random.sample(set(lst), ni)))
        print(f'D+={D[-1]}')
        for el in D[-1]:
            print(f'\t{el}')
            lst.remove(el)
    print(D)
    str_res = f'{N} {M}\n\n'
    for i in range(M):
        str_res += ' '.join([' '.join(map(str, d)) for d in D]) + '\n'
        for j in range(len(D)):
            random.shuffle(D[j])

    ni = max([len(d) for d in D]) # ni<=Ni
    return str_res, ni, D
def generate_special_case(N, M, Ni, name=None):
    """Записывает 'особый случай' теста в файл name:
    N - количество альтернатив
    M - количество экспертов
    Ni - верхняя грань для максимального размера компонент"""
    date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
    print(date_str)
    str_res,ni,D=get_special_case(N,M,Ni)
    if name is None: name=f'tests/test_n{N}_ni{ni}_D{len(D)}_m{M}_{date_str}.txt'
    f = open(name, 'w')
    f.write(str_res)
    f.write(f'\n{D}')
    f.close()
    return str_res

if __name__=="__main__":
    generate_special_case(35,9,21)