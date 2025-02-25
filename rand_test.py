import random
n=20
eq=False
m=10
l=1
def randomPermutation(n):
    a=[str(i) for i in range(1,n+1)]
    for i in range(n-1,0,-1):
        j=random.randint(0,i)
        a[i],a[j]=a[j],a[i]
    return a

f=open('test.txt', 'w')
print(f'n={n}')
if eq:
    f.write('\n')
else:
    c=[random.random() for _ in range(m)]
    s=sum(c)
    c=[round(i/s*m,l) for i in c]
    c[-1] = round(m - sum(c[:-1]),l)
    print(f'm={sum(c)}')
    c_=' '.join([str(i) for i in c])
    f.write(c_+'\n')
    print(f'c=[{c_}]')
print('r=[')
for i in range(m):
    perm=' '.join(randomPermutation(n))
    f.write(perm+'\n')
    print(f'\t<{perm}>,')
print('  ]')
f.close()