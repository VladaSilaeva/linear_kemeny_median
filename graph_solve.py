from lab6 import get_Pm
"""P,m=get_Pm('test_loop.txt')
C=P-P.T
for i in range(len(C)):
    for j in range(len(C[0])):
        if C[i][j]<0: 
            C[i][j]=0"""
"""
C=[[0,1,0,1],
   [1,0,1,0],
   [0,0,1,1],
   [0,1,0,0]]"""
def get_C(filename):
    f = open(filename, 'r')
    c=[]
    for line in f.readlines():
        if len(line.strip()):
            c.append([int(c_i) for c_i in line.split()])
        else:
            break
    f.close()
    return c

C=get_C('test_lab7.txt')
"""C=[[0,5,0,3],
   [1,0,2,0],
   [0,0,0,0],
   [2,6,1,0]]"""
"""
C=[[0,0,0,0,0],
   [2,0,0,2,0],
   [0,2,0,0,0],
   [0,0,2,0,0],
   [2,0,0,0,0]]"""
n=len(C)
ordered_arcs=[]
cost=[]
inc_list=[]
arcs_cost=dict()
for i in range(n):
    for j in range(n):
        if C[i][j]:
            arcs_cost[(i,j)]=C[i][j]
ordered_arcs=[k for k,v in sorted(arcs_cost.items(), key=lambda item: item[-1])]
cost=[v for k,v in sorted(arcs_cost.items(), key=lambda item: item[-1])]
arcs_index=dict()
for i in range(len(ordered_arcs)):
    arcs_index[ordered_arcs[i]]=i
print(f'ordered_arcs={ordered_arcs}')
for i in range(len(ordered_arcs)):
    print(f'{ordered_arcs[i]}->x{i}; ', end='')
print()
print(f'cost={cost}')
for arc in ordered_arcs:
    next=[]
    for j in range(n):
        if C[arc[1]][j]:
            next.append(arcs_index[(arc[1],j)])
    inc_list.append(sorted(next))
print(f'inc_list={inc_list}')
for i in range(len(inc_list)):
    print(f'x{i}: ', end='')
    for j in inc_list[i]:
        print(f'x{j}, ',end='')
    print()
print()

def find_loops(I, v=False):
    if len(I)>1:
        path_vertex=[ordered_arcs[i][1] for i in I]
        if v:
            print(f"\t\tvertex [i{ordered_arcs[I[0]][0]}]",end='')
            for i in path_vertex:
                print(f"-i{i}",end='')
            print()
        if (ordered_arcs[I[0]][0])==path_vertex[-1]:
            if v: print(f"loop {['x'+str(i) for i in I]}")
            return [I]
        if path_vertex[-1] in path_vertex[:-1]:
            if v: print(f"break (i{path_vertex[-1]} пройдена дважды)")
            return []
    ans=[]
    for arc in inc_list[I[-1]]:
        if arc < I[0]: # для 4 дуги считаем без 0,1,2,3
            continue
        if v: print(f"{['x'+str(i) for i in I]}+x{arc}")
        ans+=find_loops(I+[arc])
    return ans


a=[]
loops=[]
k=0
for i in range(len(ordered_arcs)):
    a.append([])
    for j in range(k):
        if i in loops[j]:
            a[i].append(j)
    cur_loops=find_loops([i])
    m=len(cur_loops)
    a[i]+=list(range(k,k+m))
    loops+=cur_loops
    k+=m
print(f"count of loops={len(loops)}")
print(f'loops={loops}')
for k in range(len(loops)):
    print(f"eta{k}={['x'+str(i) for i in loops[k]]}")
print(f'A={a}')
for i in range(len(a)):
    print(f"A{i}={['eta'+str(k) for k in a[i]]}\tcost={cost[i]}")


def solve(A,l=[],c=0, best=[], v=False):
    if v: print(['x'+str(i) for i in A])
    if not len(l):
        l=[0]*len(loops)
        if v: print(f"\t{l}")
    else:
        if v: print(f"\t{l}")
        empty=True
        for l_j in l:
            if not l_j:
                empty=False
                break
        #print(empty)
        if empty:
            if len(best) and best[0]<c:
                if v: print(f"best({best[0]})<cur({c})")
                return best
            if len(best):
                if best[0]==c:
                    if v: print(f"best({c})=cur({c}): +{['x'+str(i) for i in A]}")
                    best[1].append(A)
                else:
                    if v: print(f"best({best[0]})>cur({c})")
                    best[0]=c
                    best[1]=[A]
            else:
                best=[c,[A]]
                if v: print(f"first best({c}): {['x'+str(i) for i in A]}")
            return best
    cur=-1
    if len(A):
        cur=A[-1]
    for i in range(cur+1,len(a)):
        c1 = c + cost[i]
        if len(best) and c1>best[0]:
            return best
        l1=[l_j for l_j in l]
        for j in a[i]:
            l1[j]=1
        best = solve(A+[i],l1,c1,best)
    return best
ans=solve([])
print(f"best cost={ans[0]}")
for l in ans[1]:
    print(['x'+str(i)+": "+str(ordered_arcs[i]) for i in l])


def complete_order(I=[], in_I=[], b=[],v=False):
    if v: print(f'I={I}')
    if len(in_I)==0:
        if v:
            print(f'b={b}')
            print(C)
        in_I=[0]*n
    else:
        if len(I) == n:
            return [I]
        in_I[I[-1]]=1
    if v: print(in_I)
    v_k=[]
    for i in range(n):
        if not in_I[i]:
            #print(f'i={i}')
            l=True
            for j in range(n):
                #print(f'\tj={j}, {1-in_I[j]}, {1-C[i][j]}')
                if (not in_I[j]) and C[j][i]:
                    if (j, i) in b:
                        continue
                    l=False
                    break
            if l:
                v_k.append(i)
    ans=[]
    if v: print(f'v_k={v_k}')
    for v in v_k:
        ans += complete_order(I+[v], [i for i in in_I])
    return ans
for l in ans[1]:
    b=[ordered_arcs[i] for i in l]
    for i in range(n):
        for j in range(n):
            print(int(C[i][j]!=0 and (i,j) not in b), end=', ')
        print()
    print(complete_order(b=b,v=True))
    print()

