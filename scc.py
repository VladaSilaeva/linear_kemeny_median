from graph_solve import get_C
from lab6 import get_Pm
import numpy as np
#C=get_C('test_20.txt')
#n=len(C)
P,m=get_Pm('test20.txt')
n=len(P)
C=[[0]*n for i in range(n)]
for i in range(len(C)):
    for j in range(len(C[0])):
        c=P[i][j]-P[j][i]
        if c>0:
            C[i][j]=1
A=[np.linalg.matrix_power(np.matrix(C), i) for i in range(n)]
for i in range(n):
    print(i)
    print(A[i])
T_D=sum(A)
print('T(D)')
print(T_D)
print('S(D)')
S_D=T_D & T_D.T
print(S_D[0])
