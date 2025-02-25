from lab6 import get_Pm
"""P,m=get_Pm('test_lab6.txt')
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
inc_list=dict()
arcs_cost=dict()
n=len(C)
arcs=[]
for i in range(n):
    for j in range(n):
        if C[i][j]:
            arcs_cost[i*n+j]=C[i][j]
            arcs.append(i*n+j)
            inc_list[i*n+j]=[j*n+k for k in range(n) if C[j][k]]
print(inc_list)
loops=[]
print(arcs)
arcs=sorted(arcs_cost.items(), key=lambda item: item[-1])
print(arcs)
arcs=[k for k,v in sorted(arcs_cost.items(), key=lambda item: item[-1])]
print(arcs)

def find_loops(I):
    if len(I)>1:
        vertex=[i%n for i in I]
        print(f'\t\tvertex [{I[0]//n}]-{vertex}')
        if (I[0]//n)==vertex[-1]:
            print(f"loop {I}")
            return [I]
        if vertex[-1] in vertex[:-1]:
            print(f'break {I}')
            return []
    ans=[]
    for arc in inc_list[I[-1]]:
        if arc < I[0]:
            continue
        print(f'{I}+{arc}')
        ans+=find_loops(I+[arc])
    return ans


a=dict()
for arc in arcs:
    a[arc]=[]
    for l in loops:
        if arc in l:
            a[arc].append(l)
    cur_loops=find_loops([arc])
    a[arc]+=cur_loops
    loops+=cur_loops
print(len(loops))
print(loops)
print(a)


def solve(A, best=[]):
    pass


