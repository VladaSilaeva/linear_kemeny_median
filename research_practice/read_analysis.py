"""Вывод графиков по результатам analysis.py"""
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

def plot_bar(name):
    with open(name+'/data.json') as json_file:
        d_k = json.load(json_file)
    f=dict()
    for d,k in d_k:
        if d[0] not in f.keys():
            f[d[0]]=[]
        f[d[0]].append(k)
    colors=['maroon','crimson']
    for k in sorted(f.keys()):
        for i in range(len(f[k])):
            plt.bar([str(k)],[f[k][i]],bottom=sum(f[k][:i]),color=colors[(len(f[k])-i)%2])
    x=sorted(list(f.keys()))
    y=[sum(f[i]) for i in x]
    print(f)
    plt.plot([str(i) for i in  x],y, color='black',marker='o')
    plt.show()
#plot_bar('analysis/analysis_k1000000_n35_m3_2025_05_09_10_40_08096038')


def open_data(K,n,m,k=-1):
    if k==-1:
        files=glob.glob(f'analysis/analysis_k{K}_n{n}_m{m}_*')
    else:
        files=glob.glob(f'analysis/analysis_k{K}_{k}_n{n}_m{m}_*')
    d=[]
    if len(files):
        with open(files[0]+'/data.json') as json_file:
            d=json.load(json_file)
    else:
        print(K,n,m,k)
    return d

def plot_1(K,n,k_,data):
    colors = mpl.color_sequences['tab10']
    #colors=['black','dimgrey','darkgrey','gainsboro']
    f=[]
    ni=[]
    x_min=n
    x_max=0
    y_min=0
    y_max=0
    for i in range(len(data)):
        f.append(dict())
        ni.append(0.0)
        for d,k in data[i][1]:
            if d[0] not in f[i].keys():
                f[i][d[0]]=[]
            ni[i]+=d[0]*k/K
            f[i][d[0]].append(k*100/K)
        x=sorted(list(f[i].keys()))
        print(x)
        x_min=min(x_min,x[0])
        x_max=max(x_max,x[-1])
        y=[sum(f[i][j]) for j in x]
        print(y)
        y_max=max(y_max,max(y))
        print(data[i][0], ni[i])
        print(f[i])
        if data[i][0]:
            plt.plot([int(j) for j in x],y, marker='o',label=f'm={data[i][0]}',color=colors[i])
            plt.axvline(x=ni[i], ymin=0, ymax=100, color=colors[i], linestyle='--')
            plt.text(ni[i], i * (y_max // len(data)), f"$\widetilde{{n}}={ni[i]:.1f}$", rotation=40,
                     bbox=dict(boxstyle="square", ec=colors[i], fc='w', ))
            """plt.text(ni[i],i*(y_max//len(data)),f"$\widetilde{{n}}_{{m={data[i][0]}}}={ni[i]:.1f}$",rotation=40,
             bbox=dict(boxstyle="square",ec=colors[i], fc='w', ))"""
            """
            plt.text(ni[i], i, f"$\widetilde{{n}}={ni[i]:.1f}$", rotation=40,
                     bbox=dict(boxstyle="square", ec=colors[i], fc='w', ))"""
    plt.legend()
    plt.xlabel("размер максимальной компоненты сильной связности")
    plt.ylabel(f"частота встречаемости (%)")
    plt.grid(ls=':')
    plt.xticks(list(range(x_min,x_max,(x_max-x_min)//6))+[x_max])
    #plt.xticks(range(x_min, x_max + 1))
    #plt.yticks([i*y_max/6 for i in range(6)]+[y_max])
    #plt.yticks(range(0,int(y_max)+1,10))
    plt.title(f"n={n}, количество тестов={K}"+(bool(k_)*f", степень несогласованности {k_}"))
    plt.show()
data=[]
K = 100000
n=35
"""for m in (5,6,7):
    data.append([m,open_data(K,n,m)])
plot_1(K,n,0,data)"""
data=[]
"""
for m in (5,6,7):
    data.append([m,open_data(K,n,m,10)])
plot_1(K,n,0,data)"""
for k_ in range(5,0,-1):
    data=[]
    K = 100000
    n=35
    for m in (5,6,7):
        data.append([m,open_data(K,n,m,k_)])
    plot_1(K,n,k_,data)
def plot_4(K,n,m,data):
    colors = mpl.color_sequences['tab20b']
    f=[]
    ni=[]
    x_min=n
    x_max=0
    y_min=0
    y_max=0
    for i in range(len(data)):
        f.append(dict())
        ni.append(0.0)
        for d,k in data[i][1]:
            if d[0] not in f[i].keys():
                f[i][d[0]]=[]
            ni[i]+=d[0]*k/K
            f[i][d[0]].append(k*100/K)
        x=sorted(list(f[i].keys()))
        print(x)
        x_min=min(x_min,x[0])
        x_max=max(x_max,x[-1])
        y=[sum(f[i][j]) for j in x]
        print(y)
        y_max=max(y_max,max(y))
        print(data[i][0], ni[i])
        print(f[i])
        if data[i][0]:
            plt.plot([int(j) for j in x],y, marker='o',label=f'p={data[i][0]}',color=colors[i])
            plt.axvline(x=ni[i], ymin=0, ymax=100, color=colors[i], linestyle='--')
            plt.text(ni[i],i*(y_max//len(data)),f"$\widetilde n={ni[i]:.3f}$",rotation=40,
             bbox=dict(boxstyle="square",
                       ec=colors[i],
                       fc='w',
                       ))
    plt.legend()
    plt.xlabel("размер максимальной компоненты сильной связности")
    plt.ylabel(f"частота встречаемости (%)")
    plt.grid(ls=':')
    plt.xticks(list(range(x_min,x_max,(x_max-x_min)//6))+[x_max])
    #plt.xticks(range(x_min, x_max + 1))
    #plt.yticks([i*y_max/6 for i in range(6)]+[y_max])
    plt.title(f"n={n}, m={m}, количество тестов={K}, p - кол-во переставленных пар")
    plt.show()
for m in (3,5,7,9):
    data=[]
    K = 100000
    n=35
    for k_ in range(1,13):
        data.append([k_,open_data(K,n,m,k_)])
    #plot_4(K,n,m,data)
def plot_2(K,n,k_,data):
    colors = mpl.color_sequences['tab10']
    f=[]
    ni=[]
    x_min=n
    x_max=n
    y_min=0
    y_max=0
    for i in range(len(data)):
        f.append(dict())
        ni.append(0.0)
        for d,k in data[i][1]:
            if d[0] not in f[i].keys():
                f[i][d[0]]=[]
            ni[i]+=d[0]*k/K
            f[i][d[0]].append(k)
        x=sorted(list(f[i].keys()))
        print(x)
        x_min=min(x_min,x[0])
        y=[sum(f[i][j]) for j in x]
        print(y)
        y_max=max(y_max,max(y))
        print(data[i][0], ni[i])
        print(f[i])
        if data[i][0]:
            plt.plot([int(j) for j in x],y, marker='o',label=f'm={data[i][0]}')
            plt.axvline(x=ni[i], ymin=0, ymax=100, color=colors[i], linestyle='--')
            plt.text(ni[i],i*(y_max//len(data)),f"n={ni[i]:.3f}",rotation=40,
             bbox=dict(boxstyle="square",
                       ec=colors[i],
                       fc='w',
                       ))
    plt.legend()
    plt.xlabel("размер максимальной компоненты сильной связности")
    plt.ylabel(f"частота встречаемости (количество тестов из {K})")
    plt.grid(ls=':')
    plt.xticks(range(x_min,x_max+1,(x_max-x_min)//6))
    plt.title(f"n={n}, количество тестов={K}, {k_} пар переставлено")
    plt.show()

def plot_3(K,n,data):
    g=[]
    for i in range(len(data)):
        g.append(dict())
        for d,k in data[i][1]:
            l=len(d)
            if l not in g[i].keys():
                g[i][l]=[]
            g[i][l].append(k*100/K)
        x=sorted(list(g[i].keys()))
        y=[sum(g[i][j]) for j in x]
        print(data[i][0],g[i])
        if data[i][0]<13:
            plt.plot([str(j) for j in  x],y, marker='o',label=f'm={data[i][0]}')
    plt.legend()
    plt.xlabel("количество компонент сильной связности")
    plt.ylabel("%")
    plt.title(f"N={n}, K={K}")
    plt.show()
