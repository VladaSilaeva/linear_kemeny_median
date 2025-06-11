import graphviz


p = graphviz.Graph(name='parent',engine='twopi')#['circo', 'dot', 'fdp', 'neato', 'osage', 'patchwork', 'sfdp', 'twopi']
N=32
n=N
C=[[0]*n for i in range(N)]
sq=[(i+1)**2 for i in range(int((2*N)**0.5))]
draw_edges=dict()
for i in range(1, N+1):
    for s in sq:
        if 0 < s-i < i:
            C[i-1][s-i-1]=C[s-i-1][i-1]=1
            print(f'{i}+{s-i}={s}')
            draw_edges[(i,s-i)]=False

print(*C,sep='\n')
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
#print(f'cost={cost}')
for arc in ordered_arcs:
    next=[]
    for j in range(n):
        if C[arc[1]][j]:
            next.append(arcs_index[(arc[1],j)])
    inc_list.append(sorted(next))
"""print(f'inc_list={inc_list}')
for i in range(len(inc_list)):
    print(f'x{i}: ', end='')
    for j in inc_list[i]:
        print(f'x{j}, ',end='')
    print()
print()"""

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
k=0
for l in loops:
    if len(l)==N:
        print(l)
        print('\t', end='')
        for a in l:
            i, j = ordered_arcs[a]
            if j < i: i, j = j, i
            print(f'({i+1} {j+1})', end='-')
            if k==0:
                p.edge(f'{i+1}',f'{j+1}',label=f'{i+j+2}',color='blue')
                draw_edges[(i+1,j+1)] = True
            """elif k==1:
                p.edge(f'{j + 1}', f'{i + 1}', label=f'{i + j + 2}', color='red')
                draw_edges[(i + 1, j + 1)] = True"""
        print()
        k+=1
for (i,j),is_draw in draw_edges.items():
    if not is_draw:
        #p.edge(f'{i + 1}', f'{j + 1}', label=f'{i + j + 2}')
        print(f'{i + 1}', f'{j + 1}',)
#print(f'loops={loops}')
print(p.source)
p.format='png'
p.render(directory='doctest-output', view=True)