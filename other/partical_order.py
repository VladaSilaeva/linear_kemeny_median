C=[[0, 0, 1, 0, 1],
[0, 0, 1, 1, 1],
[0, 0, 0, 0, 0],
[0, 0, 1, 0, 1],
[0, 0, 0, 0, 0]]
"""C=[[0, 1, 1, 0],
[0, 0, 0, 0],
[0, 1, 0, 0],
[0, 1, 0, 0]]"""
n=len(C)
def complete_order(I=[], in_I=[]):
    print(f'I={I}')
    if len(in_I)==0:
        in_I=[0]*n
    else:
        if len(I) == n:
            return [I]
        in_I[I[-1]]=1
    print(in_I)
    v_k=[]
    for i in range(n):
        if not in_I[i]:
            #print(f'i={i}')
            l=True
            for j in range(n):
                #print(f'\tj={j}, {1-in_I[j]}, {1-C[i][j]}')
                if (not in_I[j]) and C[j][i]:
                    l=False
                    break
            if l:
                v_k.append(i)
    ans=[]
    print(f'v_k={v_k}')
    for v in v_k:
        ans += complete_order(I+[v], [i for i in in_I])
    return ans

print(complete_order())