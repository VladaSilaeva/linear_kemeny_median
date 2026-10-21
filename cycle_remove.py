import sys
from collections import defaultdict


def build_edges(n, C):
    """Строит список дуг (start, end, weight, original_index)."""
    edges = []
    for i in range(n):
        for j in range(n):
            w = C[i][j]
            if w != 0:   # предполагаем, что 0 означает отсутствие дуги
                edges.append((i, j, w))
    # Сохраняем исходный порядок для стабильности при равных весах
    for idx, e in enumerate(edges):
        edges[idx] = e + (idx,)
    # Сортировка по весу, при равенстве – по исходному индексу
    edges.sort(key=lambda x: (x[2], x[3]))
    # Убираем вспомогательный индекс, оставляем (start, end, weight)
    edges = [(s, e, w) for s, e, w, _ in edges]
    return edges

def find_all_cycles(n, edges):
    """Возвращает список всех простых контуров в лексикографическом порядке.
       Каждый контур – список индексов дуг в порядке обхода."""
    N = len(edges)
    # Список инцидентности: для каждой вершины список индексов исходящих дуг
    adj = [[] for _ in range(n)]
    for idx, (s, e, w) in enumerate(edges):
        adj[s].append(idx)
    # Для лексикографического порядка индексы дуг уже отсортированы по весу,
    # поэтому adj[s] автоматически будет в порядке возрастания индексов.
    cycles = []

    for i in range(N):
        start = edges[i][0]
        end = edges[i][1]
        path = [i]
        visited = {start, end}

        def dfs(cur_v, path, visited, min_idx):
            for e_idx in adj[cur_v]:
                if e_idx < min_idx:
                    continue
                if e_idx in path:
                    continue
                nxt = edges[e_idx][1]
                if nxt == start:
                    # нашли цикл
                    cycles.append(path + [e_idx])
                elif nxt not in visited:
                    path.append(e_idx)
                    visited.add(nxt)
                    dfs(nxt, path, visited, min_idx)
                    visited.remove(nxt)
                    path.pop()

        dfs(end, path, visited, i)

    # Дополнительная сортировка для гарантии лексикографического порядка
    cycles.sort()
    return cycles

def find_minimal_covers(cycles, edges):
    """Возвращает список минимальных по стоимости покрытий контуров.
       Каждое покрытие – список индексов дуг (глобальных)."""
    if not cycles:
        return [[]]  # пустое покрытие

    # Преобразуем циклы в множества дуг для быстрой проверки
    cycles_sets = [set(cycle) for cycle in cycles]
    # Все дуги, входящие хотя бы в один цикл
    used_edges = set()
    for cs in cycles_sets:
        used_edges.update(cs)
    relevant = sorted(used_edges)   # глобальные индексы, отсортированные по возрастанию
    # Веса для быстрого доступа
    weight = {idx: edges[idx][-1] for idx in relevant}

    best_cost = float('inf')
    best_solutions = []

    # Рекурсивный перебор с ветвями и границами
    def branch(selected, last_idx, uncovered, cur_cost):
        nonlocal best_cost, best_solutions
        if not uncovered:
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_solutions = [selected.copy()]
            elif cur_cost == best_cost:
                best_solutions.append(selected.copy())
            return
        if cur_cost >= best_cost:
            return

        # Собираем всех кандидатов из непокрытых циклов
        candidates = set()
        for uc in uncovered:
            candidates.update(cycles_sets[uc])
        # Оставляем только дуги с индексом > last_idx и не выбранные ранее
        candidates = [e for e in candidates if e > last_idx and e not in selected]
        candidates.sort()   # перебираем в порядке возрастания индекса (веса)
        for e in candidates:
            new_uncovered = [uc for uc in uncovered if e not in cycles_sets[uc]]
            selected.append(e)
            branch(selected, e, new_uncovered, cur_cost + weight[e])
            selected.pop()

    # Запускаем деревья для каждого возможного первого элемента
    for idx, first in enumerate(relevant):
        # Начальные непокрытые циклы – все, кроме тех, что содержат first
        uncovered = [i for i, cs in enumerate(cycles_sets) if first not in cs]
        branch([first], first, uncovered, weight[first])

    return best_cost, best_solutions

def invert_edges(n, edges, cover):
    """Строит новый граф, инвертируя дуги из cover.
       Возвращает список смежности (для каждой вершины список соседей)."""
    # Исходный граф как множество дуг (start, end)
    # Для простоты построим словарь исходящих дуг
    graph = defaultdict(list)
    for idx, (s, e, w) in enumerate(edges):
        if idx in cover:
            # инвертируем
            graph[e].append(s)
        else:
            graph[s].append(e)
    # Преобразуем в обычный список списков для всех вершин
    adj = [[] for _ in range(n)]
    for v in range(n):
        adj[v] = list(graph[v])
    return adj

def topological_sorts(n, adj):
    """Генерирует все топологические сортировки ориентированного ациклического графа.
       Возвращает список порядков (каждый порядок – список вершин)."""
    indeg = [0] * n
    for u in range(n):
        for v in adj[u]:
            indeg[v] += 1

    results = []
    current = []

    def backtrack():
        # Находим все вершины с нулевой полустепенью захода
        zeros = [v for v in range(n) if indeg[v] == 0]
        if not zeros:
            if len(current) == n:
                results.append(current[::-1])
            return
        for v in zeros:
            # выбираем v
            current.append(v)
            indeg[v] = -1   # временно удаляем
            for nb in adj[v]:
                indeg[nb] -= 1
            backtrack()
            # восстанавливаем
            for nb in adj[v]:
                indeg[nb] += 1
            indeg[v] = 0
            current.pop()

    backtrack()
    return results

def get_min_loop_split(C, verbose=0,graph=None,i=1):
    # 1. Чтение матрицы
    n=len(C)

    # 2. Построение списка дуг, сортировка по весу
    edges = build_edges(n, C)
    N = len(edges)
    if verbose:
        print(f"Найдено {N} дуг.", end=" ")
    if graph is not None:
        graph[-1][i] = [N, None]

    weights = [w for (_, _, w) in edges]

    # 3. Поиск всех простых контуров
    cycles = find_all_cycles(n, edges)
    if verbose:
        print(f"Найдено {len(cycles)} контуров.", end=" ")
    if graph is not None:
        graph[-1][i] = [N, len(cycles)]

    # 4. Поиск минимальных покрытий
    cost, covers = find_minimal_covers(cycles, edges)
    if verbose:
        print(f"Найдено {len(covers)} минимальных покрытий со стоимостью {cost},")

    # 5. Для каждого покрытия строим инвертированный граф и находим все линейные порядки
    all_orders = []
    if not cycles:
        # Нет контуров – исходный граф уже ациклический
        adj = [[] for _ in range(n)]
        for s, e, _ in edges:
            adj[s].append(e)
        orders = topological_sorts(n, adj)
        all_orders.extend(orders)
    else:
        for cover in covers:
            # Инвертируем дуги из cover
            adj = invert_edges(n, edges, set(cover))
            # Проверка ацикличности (построение топологической сортировки)
            orders = topological_sorts(n, adj)
            all_orders.extend(orders)

    # Убираем дубликаты порядков (если они совпадают)
    unique_orders = list({tuple(order) for order in all_orders})
    unique_orders.sort()  # для упорядоченного вывода

    # Вывод результатов: каждый линейный порядок на отдельной строке
    return cost, unique_orders

if __name__ == "__main__":
    C = [[0, 3, 0, 1, 1, 0, 0, 3, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [1, 3, 0, 1, 1, 1, 0, 5, 3, 0, 0, 1],
         [0, 5, 0, 0, 3, 0, 0, 1, 1, 0, 0, 1],
         [0, 3, 0, 0, 0, 0, 0, 3, 1, 0, 0, 3],
         [1, 3, 0, 1, 1, 0, 0, 1, 3, 0, 0, 0],
         [1, 3, 1, 1, 3, 1, 0, 3, 1, 0, 0, 1],
         [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 5, 1, 5, 3, 1, 1, 5, 1, 0, 1, 1],
         [3, 5, 1, 1, 1, 1, 1, 3, 3, 0, 0, 3],
         [0, 3, 0, 0, 0, 1, 0, 3, 3, 0, 0, 0]]
    ans = get_min_loop_split(C, verbose=1)
    for lin in ans[1]:
        print(lin)