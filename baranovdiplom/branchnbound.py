import sys
import numpy as np
import time
import json
######################################################################
#класс деревьев
class tree:
    #глобальные переменные класса
    #переменная для записи оценки рекорда
    record = int()
    #массив последовательностей альтернатив рекордов
    records = []
    #переменная для записи матрицы Р
    p = np.array([])
    #переменная для записи матрицы Р1
    p1 = np.array([])
    #переменная для записи массива значений альфа
    alpha = []
    #переменная для записи массива значений бета
    beta = []
    #переменная для записи массива значений гамма
    gamma = []
    #переменная для записи массива значений ню
    nu = []
    #переменная для записи массива значений мю
    mu = []
    #переменная для записи количества альтернатив
    nalt = int()
    #переменная для записи количества сделанных итеаций
    it = int()

    #инициализация дерева
    def __init__(self):
        print("ДЕРЕВО СОЗДАНО")

    #функция запуска решения нахождения агрегированного отношения: alternatives - варианты альтернатив 1 уровня
    def start(self, alternatives):
        #пустой массив для заполнения выбранной в ходе решения последовательности альтернатив
        branch = []
        self.it = 0
        #запуск функции динамического генерирования дерева
        self.findbest(self, alternatives, branch, 0, 0)

    #динамическое генерирование дерева решений с применением метода ветвей и границ: variants - массив для выбора возможных альтернатив, branch - массив выбранных альтернатив, beta_k - значение бета текущей альтернативы, level - текущий уровень дерева
    def findbest(self, variants, branch, beta_k, level):
        print("findbest(): variants = ", variants, " branch = ", branch, " beta_k = ", beta_k, " level = ", level)
        #проверка на наличие возможных альтернатив для дальнейшего выбора
        if len(variants) == 0:
            print(branch, " КОНЕЦ ПОСЛЕДОВАТЕЛЬНОСТИ АЛЬТЕРНАТИВ!")
            #проверка на рекорд
            if self.record > beta_k or self.record == 0:
                self.record = int(beta_k)
                self.records.clear()
                self.records.append(list(branch))
                print("АБСОЛЮТНЫЙ РЕКОРД: ", self.record, " ПОСЛЕДОВАТЕЛЬНОСТЬ: ", self.records)
            elif self.record == beta_k:
                self.records.append(list(branch))
                print("НОВАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ, РЕКОРД: ", self.record, " ПОСЛЕДОВАТЕЛЬНОСТИ: ", self.records)
            return
        #создание буферных переменных для сохранения результатов на текущем уровне
        save_variants = list(variants)
        sorted_variants = list()
        sorted_branch = list(branch)
        save_beta_k = int(beta_k)
        save_beta = list()
        sorted_beta = list()
        save_d = list()
        sorted_d = list()
        save_mu = list()
        sorted_mu = list()
        #оценка нижних листьев, цикл перебирает по очереди все элементы массива возможных вариантов альтернатив. index - индекс элемента в массиве save_variants, el - сам элемент
        for index, el in enumerate(save_variants):
            print("# ПОДСЧЁТ ОЦЕНКИ ДЛЯ НИЖЕСТОЯЩЕЙ ВЕРШИНЫ # level = ", level, " element = ", el)
            if level == 0:
                #все оценки для 1 уровня дерева всегда посчитаны заранее
                save_mu.append(int(self.mu[index]))
                save_d.append(0)
                save_beta.append(int(self.beta[index]))
            else:
                #в общем случае:
                #check_branch - массив выбранных альтернатив + альтернатива, которая оценивается в этом цикле
                check_branch = list(branch)
                check_branch.append(int(el))
                #подсчёт значения D
                d = self.countd(self, check_branch)
                #сохранение значения D в массиве
                save_d.append(d)
                #сохранение значения бета в массиве
                save_beta.append(save_beta_k + d)
                #подсчёт значения мю
                save_mu_k = self.countmu(self, check_branch, save_beta_k + d)
                #сохранение значения мю в массиве
                save_mu.append(save_mu_k)
        #словарь dt индексирует все элементы save_mu от 0 до n
        dt = {k: v for k, v in enumerate(save_mu,start=0)}
        #sort_mu содержит массив индексов словаря dt, который отсортировали по значениям мю
        sort_mu = sorted(dt, key=dt.get)
        #сортировка вариантов альтернатив
        for el in sort_mu:
            sorted_variants.append(save_variants[el])
            sorted_beta.append(save_beta[el])
            sorted_d.append(save_d[el])
            sorted_mu.append(save_mu[el])
        #выбор следующего шага, index - индекс элемента в массиве sorted_variants, el - сам элемент
        for index, el in enumerate(sorted_variants):
            print("## ПРОВЕРКА ОЦЕНКИ ## level = ", level, " element = ", el)
            #отсечение ветвей решения, оценка которых больше рекорда
            if sorted_mu[index] > self.record and self.record != 0:
                continue
            #элемент цикла добавляется в массив пройденных листьев, убирается из массива вариантов альтернатив
            branch.append(el)
            variants.remove(el)
            print("### ШАГ ПО ДЕРЕВУ ### level = ", level, " element = ", el)
            self.it = self.it + 1
            #рекурсия
            self.findbest(self, variants, branch, sorted_beta[index], len(branch))
            #обновление значений из буферных переменных, для того, чтобы цикл for правильно работал. Переменные variants, branch в рекурсии меняют своё значение
            variants = list(sorted_variants)
            branch = list(sorted_branch)
            
    #функция подсчёта значения мю. branch - массив выбранных альтернатив, beta_k - значение бета текущей альтернативы
    def countmu(self, branch, beta_k):
        #копии массивов значений альфа и ню из глобальных переменных
        save_alpha = list(self.alpha)
        save_nu = list(self.nu)
        print("countmu(): branch = ", branch, " beta_k = ", beta_k, " save_alpha = ", save_alpha, " save_nu = ", save_nu)
        #сортировка альтернатив по порядку, для правильного прохождения цикла while
        sorted_branch = sorted(branch)
        i = len(sorted_branch)
        while i != 0:
            save_alpha.pop(sorted_branch[i-1]-1)
            save_nu.pop(sorted_branch[i-1]-1)
            i = i - 1
        #когда подсчитывается альтернатива последнего уровня
        if len(save_alpha) == 0:
            save_alpha.append(0)
            save_nu.append(0)
        cntmu = beta_k + sum(save_alpha) + min(save_nu)
        print("return cntmu = ", cntmu)
        return cntmu
         
    #функция подсчёта значения D. branch - массив выбранных альтернатив
    def countd(self, branch):
        print("countd(): branch =", branch)
        sump = 0
        sump1 = 0
        last = int(branch[-1])
        for i in range(0, len(branch)-1):
            el = int(branch[i])
            print(self.p[last-1][el-1], "p")
            sump += self.p[last-1][el-1]
        #все возможные альтернативы
        alternatives = []
        for i in range(0, self.nalt):
            alternatives.append(int(i+1))
        #вычитание множества
        for i in range(0, len(branch)-1):
            el = int(branch[i])
            alternatives.remove(el)
        for i in range(0, len(alternatives)):
            el = int(alternatives[i])
            print(self.p1[last-1][el-1], "p1")
            sump1 += self.p1[last-1][el-1]
        cntd = sump + sump1
        print("return cntd = ", cntd)
        return cntd
 
######################################################################
#функция создания матрицы смежности
def makematrix(a):
        Matrix = np.array([])
        #единичная матрица
        Matrix = np.eye(len(a))
        for i in range(0, len(a)):
            it = i
            while it != 0:
                it = it - 1
                Matrix[a[it]-1,a[i]-1] = 1
        return Matrix

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

#начало программы, ввод исходных данных
n = int()
m = int()
data = []
if len(sys.argv) > 1:
    data = dict()
    with open("./" + sys.argv[1], "r") as read_file:
        data = json.load(read_file)
    n = data["n"]
    m = data["m"]
    p = np.zeros((n, n))
    preferences = list(data["preferences"])
    for el in preferences:
        p += makematrix(el)
else:
    n = int(input("Введите число альтернатив: "))
    m = int(input("Введите число экспертов: "))
    while m % 2 == 0:
        print("Ошибка, m должно быть нечётным числом")
        m = int(input("Введите число экспертов: "))
    #нулевая матрица n x n
    p = np.zeros((n, n))
    print("Введите через пробел индивидуальные предпочтения экспертов, например: 1 2 3 4")
    preferences = []
    itm = 0
    check = []
    for i in range(0, n):
        check.append(int(i+1))
    while itm != m:
        try:
            print("Введите индивидуальные предпочтения для эксперта ", itm + 1)
            el = list(map(int, input().split()))
            checkel = list(el)
            checkel.sort()
            if checkel == check:
                itm = itm + 1
                preferences.append(el)
                p += makematrix(el)
            else:
                print("Введённое количество или названия альтернатив не соответствуют формату")
        except ValueError:
            print("Ошибка, вводите числа")
checklider = []
for i in range(0, n+1):
    checklider.append(int(i))
boolcheklider = 0
lider = int()
while boolcheklider != 1:
    lider = int(input("Введите альтернативу-лидер, если она есть, в противном случае число 0: "))
    for i in range(0, n+1):
        if checklider[i] == lider:
            boolcheklider = 1
start_time = time.time()
alpharez = 0
copyp = p.copy()
if lider != 0:
    for i in range(0,n):
        alpharez += p[lider-1, i]
    alpharez = (alpharez - m) * 2
    newpref = list()
    p = np.zeros((n-1, n-1))
    for i in range(0,m):
        newexp = list()
        for j in range(0,n):
            if preferences[i][j] == lider:
                continue
            if preferences[i][j] > lider:
                newexp.append(preferences[i][j]-1)
                continue
            newexp.append(preferences[i][j])
        newpref.append(newexp)
        p += makematrix(newexp)
    print(p)
    n = n - 1
#вычисление матрицы Р1
p1 = m - p
#вычисление матрицы P*
pz = p.copy()
for i in range(0, n):
    for j in range(0, n):
        if p1[i,j] < pz[i,j]:
            pz[i,j] = int(p1[i,j])
#вычисление матрицы G
g = makeG(p, n)
pt = np.transpose(p)
c = p - pt
for i in range(0, n):
    for j in range(0, n):
        if c[i, j] < 0:
            c[i, j] = 0
#создание массивов для посчёта оценок
alpha = []
beta = []
gamma = []
nu = []
mu = []
#подсчёт значений альфа
for i in range(0, n):
    alphai = 0
    for j in range(0, n):
        alphai += int(pz[i,j])
    alpha.append(alphai)
#подсчёт значений бета
for i in range(0, n):
    betai = 0
    for j in range(0, n):
        betai += int(p1[i,j])
    beta.append(betai)
#подсчёт значений гамма
for i in range(0, n):
    gammai = 0
    for j in range(0, n):
        if j == i:
            continue
        gammai += int(p[i,j])
    gamma.append(gammai)
#подсчёт значений ню
for i in range(0, n):
    nui = gamma[i] - alpha[i]
    nu.append(nui)
#подсчёт значений мю
for i in range(0, n):
    savenu = list(nu)
    savealphai = list(alpha)
    savenu.remove(nu[i])
    savealphai.remove(alpha[i])
    mui = beta[i] + sum(savealphai) + min(savenu)
    mu.append(mui)
print("mu: ", mu)
#массив возможных альтернатив
alternatives = []
for i in range(0, n):
    alternatives.append(int(i+1))
#создание дерева для решения задачи
tree()
#присвоение переменных объекта посчитанным значениям
tree.p = list(p)
tree.p1 = list(p1)
tree.alpha = list(alpha)
tree.beta = list(beta)
tree.gamma = list(gamma)
tree.nu = list(nu)
tree.mu = list(mu)
tree.nalt = int(n)
#запуск решения задачи
tree.start(tree, alternatives)
end_time = time.time()
if lider != 0:
    tree.record += alpharez
    for i in range(0, len(tree.records)):
        for j in range(0, n):
            if tree.records[i][j] >= lider:
                tree.records[i][j] += 1
        tree.records[i].append(lider)
    p = copyp.copy()
    n = n + 1
    p1 = m - p
    # вычисление матрицы P*
    pz = p.copy()
    for i in range(0, n):
        for j in range(0, n):
            if p1[i, j] < pz[i, j]:
                pz[i, j] = int(p1[i, j])
    # вычисление матрицы G
    g = makeG(p, n)
    pt = np.transpose(p)
    c = p - pt
    for i in range(0, n):
        for j in range(0, n):
            if c[i, j] < 0:
                c[i, j] = 0
    # создание массивов для посчёта оценок
    alpha = list()
    for i in range(0, n):
        alphai = 0
        for j in range(0, n):
            alphai += int(pz[i, j])
        alpha.append(alphai)
print(preferences)
print("G:\n", g, "\nC:\n", c, "\nP:\n", p, "\nP1:\n", p1, "\nPz:\n", pz)
Ilist = list()
for i in range(0, n):
    ct = 0
    for j in range(0, n):
        if g[i, j] == 1:
            ct = ct + 1
    Ilist.append(ct)
di = {k: v for k, v in enumerate(Ilist, start=1)}
sorted_tuple = sorted(di.items(), key=lambda x: x[1])
sI = sorted(Ilist)
print("I(ro) = ", sI, " ", sorted_tuple)
check = []
for i in range(0, n):
    check.append(int(i+1))
print("I2(ro) = ", [a - b for a, b in zip(sI, check)])
print("Рекорд = ", tree.record, "Последовательности: ", tree.records)
suma = 0
for i in alpha:
    print("alpha = ", i)
    suma += i
print("alpha sum = ", suma)
print(tree.it, " ", len(tree.records), " ", end_time - start_time, " ", suma, " ", tree.record)