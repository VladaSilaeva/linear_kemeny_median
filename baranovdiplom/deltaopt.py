import sys
import numpy as np
import json

######################################################################
class PreferenceOfExperts:  # КЛАСС ИНДИВИДУЛЬНЫХ ПРЕДПОЧТЕНИЙ
    a = []
    b = []
    sumRo = int()

    def __init__(self, a, m):  # КОНСТРУКТОР
        self.a = a
        self.m = m

    def printlist(self):
        print("Эксперт: ", self.m)
        print("Индивидуальные предпочтения: ", self.a)

    def makeMatrix(self):  # СОЗДАНИЕ МАТРИЦЫ СМЕЖНОСТИ
        Matrix = np.array([])
        Matrix = np.eye(len(self.a))  # СОЗДАНИЕ ЕДИНИЧНОЙ МАТРИЦЫ
        for i in range(0, len(self.a)):  # СРАВНЕНИЕ ЭЛЕМЕНТОВ ЛЕВЕЕ
            it = i
            while it != 0:
                it = it - 1
                Matrix[self.a[it] - 1, self.a[i] - 1] = 1
        return Matrix


class adr:
    a = int()
    b = int()

    def __init__(self, a, b):
        self.a = a
        self.b = b


def makeG(p, n):
    Matrix = np.array([])
    # единичная матрица
    Matrix = np.eye(n)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            if p[i, j] <= p[j, i]:
                Matrix[i, j] = 0
            else:
                Matrix[i, j] = 1
    return Matrix


def swap(g1, i, j):
    tmp1 = int(g1[i, j])
    tmp2 = int(g1[j, i])
    g1[i, j] = tmp2
    g1[j, i] = tmp1
    return g1


def solve(g, p1, n):
    if solve2(g, p1, n) == 1:
        print("Найдены все 2-оптимальные решения\n\n")
        return
    else:
        print("2-оптимальных решения не существует\n\n")
    if solve4(g, p1, n) == 1:
        print("Найдены все 4-оптимальные решения\n\n")
        return
    else:
        print("4-оптимальных решения не существует\n\n")
    if solve6(g, p1, n) == 1:
        print("Найдены все 6-оптимальные решения\n\n")
        return
    else:
        print("6-оптимальных решения не существует\n\n")


def check(g1, n):
    v = list()
    for i in range(0, n):
        v.append(0)
    for i in range(0, n):
        for j in range(0, n):
            if g1[i, j] == 1:
                v[i] = v[i] + 1
    v = sorted(v)
    for i in range(0, n - 1):
        if v[i] != i + 1:
            return 0
    print("Решение: \n", g1, "\nОтношение: ", binotn(g1))
    return 1

def binotn(g1):
    m = list()
    for i in range(0, n):
        ct = 0
        for j in range(0, n):
            if g1[i,j] == 1:
                ct = ct + 1
        m.append(ct)
    dt = {k: v for k, v in enumerate(m, start=0)}
    sort_dt = sorted(dt, key=dt.get, reverse=True)
    for i in range(0, len(sort_dt)):
        sort_dt[i] = sort_dt[i] + 1
    return sort_dt

def solve2(g, p1, n):
    count = 0
    for i in range(0, n):
        for j in range(0, n):
            if p1[i, j] == 1:
                g1 = np.copy(g)
                g1 = swap(g1, i, j)
                if check(g1, n) == 1:
                    count = count + 1
    if count != 0:
        return 1
    else:
        return 0


def solve4(g, p1, n):
    flag = 0
    lst = list()
    count = 0
    for i in range(0, n):
        for j in range(0, n):
            if p1[i, j] == 1:
                ad = adr(i, j)
                lst.append(ad)
                count = count + 1
    for i in range(0, count - 1):
        for j in range(i+1, count):
            g1 = np.copy(g)
            g1 = swap(g1, lst[i].a, lst[i].b)
            g1 = swap(g1, lst[j].a, lst[j].b)
            if check(g1, n) == 1:
                flag = flag + 1
    if flag != 0:
        return 1
    else:
        return 0

def solve6(g, p1, n):
    flag = 0
    lst = list()
    count = 0
    for i in range(0, n):
        for j in range(0, n):
            if p1[i, j] == 1:
                ad = adr(i, j)
                lst.append(ad)
                count = count + 1
            elif p1[i, j] == 3:
                g1 = np.copy(g)
                swap(g1, i, j)
                if check(g1, n) == 1:
                    flag = flag + 1
    for i in range(0, count - 2):
        for j in range(i+1, count):
            for z in range(j+1, count):
                g1 = np.copy(g)
                g1 = swap(g1, lst[i].a, lst[i].b)
                g1 = swap(g1, lst[j].a, lst[j].b)
                g1 = swap(g1, lst[z].a, lst[z].b)
                if check(g1, n) == 1:
                    flag = flag + 1
    if flag != 0:
        return 1
    else:
        return 0


# начало программы, ввод исходных данных
n = int()
m = int()
data = []
preferences = []  # МАССИВ ИНДИВИДУАЛЬНЫХ ПРЕДПОЧТЕНИЙ ЭКСПЕРТОВ
if len(sys.argv) > 1:
    data = dict()
    with open("./" + sys.argv[1], "r") as read_file:
        data = json.load(read_file)
    n = data["n"]
    m = data["m"]
    preferencesjson = list(data["preferences"])
    for index, el in enumerate(preferencesjson):
        preferences.append(PreferenceOfExperts(el, index + 1))
else:
    n = int(input("Введите число альтернатив: "))
    m = int(input("Введите число экспертов: "))
    while m % 2 == 0:
        print("Ошибка, m должно быть нечётным числом")
        m = int(input("Введите число экспертов: "))
    print("Введите через пробел индивидуальные предпочтения экспертов, например: 1 2 3 4")
    #preferences = []
    itm = 0
    chck = []
    for i in range(0, n):
        chck.append(int(i + 1))
    while itm != m:
        try:
            print("Введите индивидуальные предпочтения для эксперта ", itm + 1)
            el = list(map(int, input().split()))
            checkel = list(el)
            checkel.sort()
            if checkel == chck:
                preferences.append(PreferenceOfExperts(el, itm + 1))
                itm = itm + 1
            else:
                print("Введённое количество или названия альтернатив не соответствуют формату")
        except ValueError:
            print("Ошибка, вводите числа")

p = np.zeros((n, n))
for el in preferences:
    el.printlist()
    p += el.makeMatrix()
    print(el.makeMatrix())

print("\nP:\n", p)
g = makeG(p, n)
print("\nG:\n", g)
pt = np.transpose(p)
c = p - pt
for i in range(0, n):
    for j in range(0, n):
        if c[i, j] < 0:
            c[i, j] = 0
print("\nC:\n", c, "\n-----------")
solve(g, c, n)
sys.exit()