import numpy as np

n=5
C=[[0,0,1,0,0],
   [1,0,0,1,1],
   [0,1,0,0,0],
   [0,0,0,0,1],
   [0,0,0,0,0]]


def scc(C, n=None,v=0):
    def matrix_multiply(matrix_a, matrix_b):
        # Проверка размеров матриц
        if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
            raise ValueError("Матрицы должны быть одинакового размера.")

        n = len(matrix_a)
        result_matrix = [[0 for _ in range(n)] for _ in range(n)]

        # Перемножение матриц
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result_matrix[i][j] += int(bool(matrix_a[i][k] * matrix_b[k][j]))
                result_matrix[i][j]=int(bool(result_matrix[i][j]))
        return result_matrix
    def matrix_E(n):
        E = [[0] * n for _ in range(n)]
        for i in range(n):
            E[i][i] = 1
        return E

    def matrix_sum(M):
        return sum([np.array(m) for m in M])

    def matrix_trans(M):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

    def matrix_and(a, b):
        return np.array(a) & np.array(b)
    def process_matrix(matrix):
        def mark_and_get_nonzeros(row_idx):
            """Ищет индексы ненулевых элементов в указанной строке,
               одновременно помечая строку и столбцы нулевыми значениями."""
            indices = []
            for col_idx, value in enumerate(matrix[row_idx]):
                if value != 0:
                    indices.append(col_idx)
                    # Обнуляем весь столбец
                    for r in range(len(matrix)):
                        matrix[r][col_idx] = 0
            # Обнуляем всю строку
            matrix[row_idx][:] = [0] * len(matrix[row_idx])
            return indices

        result = []
        rows_to_check = list(range(len(matrix)))
        while rows_to_check:
            current_row = rows_to_check.pop(0)
            non_zero_cols = mark_and_get_nonzeros(current_row)
            if non_zero_cols:
                result.append(non_zero_cols)
                # Убираем из списка проверяемых строк те, которые уже стали пустыми
                rows_to_check[:] = [row for row in rows_to_check if sum(matrix[row]) > 0]
        return result
    if n is  None: n=len(C)
    A = []
    A.append(matrix_E(n))
    A.append([row.copy() for row in C])
    for i in range(n):
        A.append(matrix_multiply(A[i], C))
    for i in range(n + 1):
        if v:
            print(i)
            print(*A[i], sep='\n',end='\n\n')
    T_D = matrix_sum(A)
    if v:
        print(T_D, sep='\n', end='\n\n')
    S_D = matrix_and(T_D, matrix_trans(T_D))
    if v:
        print(S_D, sep='\n', end='\n\n')
    return process_matrix(S_D)


def scc2(C,v=0):
    n=len(C)
    C_bool=np.array(np.matrix(C),dtype=bool)
    if v:
        print('C')
        print(C_bool)
    A=[np.array(np.eye(n),dtype=bool)]
    for i in range(n):
        A_i=np.matmul(A[-1],C_bool)
        if v:
            print(f'A{i}')
            print(np.array(A_i,dtype=int))
        A.append(A_i)
    T_D=np.array(sum(A),dtype=bool)
    if v:
        print('T_D')
        print(np.array(T_D,dtype=int))
    S_D=T_D&T_D.T
    if v:
        print('S_D')
        print(np.array(S_D,dtype=int))
    D=[]
    k=-1
    is_in_D=[0]*n
    while sum(is_in_D)!=n:
        for i in range(n):
            if is_in_D[i]:
                continue
            D.append([])
            for j in range(n):
                if is_in_D[j]:
                    continue
                if S_D[i][j]:
                    D[-1].append(j)
                    is_in_D[j]=1
    if v:
        print(D)
    return D
C= [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0]]
print(scc2(C))