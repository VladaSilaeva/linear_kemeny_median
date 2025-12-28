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
            plt.text(ni[i], i * (y_max // len(data)), f"$\widetilde{{n}}={ni[i]:.3f}$", rotation=40,
                     bbox=dict(boxstyle="square", ec=colors[i], fc='w', ))
            """plt.text(ni[i],i*(y_max//len(data)),f"$\widetilde{{n}}_{{m={data[i][0]}}}={ni[i]:.1f}$",rotation=40,
             bbox=dict(boxstyle="square",ec=colors[i], fc='w', ))"""
            """
            plt.text(ni[i], i, f"$\widetilde{{n}}={ni[i]:.3f}$", rotation=40,
                     bbox=dict(boxstyle="square", ec=colors[i], fc='w', ))"""
    plt.legend()
    plt.xlabel("размер максимальной компоненты сильной связности")
    plt.ylabel(f"частота встречаемости (%)")
    plt.grid(ls=':')
    print(f'x_min={x_min},x_max={x_max},step={(x_max-x_min)//6}')
    plt.xticks(list(range(x_min,x_max,max((x_max-x_min)//6, 1)))+[x_max])
    #plt.xticks(range(x_min, x_max + 1))
    #plt.yticks([i*y_max/6 for i in range(6)]+[y_max])
    #plt.yticks(range(0,int(y_max)+1,10))
    plt.title(f"$n={n}$, количество тестов=${K}$"+(bool(k_)*f", степень несогласованности $k={k_}$"))
    plt.show()
data=[]
K = 10000
n=35
k_s=list(range(10, (35**2-35)//2, 10))
m_s=(5,6,7)

for k_ in k_s:
    data=[]
    for m in m_s:
        data.append([m,open_data(K,n,m,k_)])
    plot_1(K,n,k_,data)
def plot_4(K,n,m,data):
    print(f'plot4 n={n} m={m}')
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
        #print(x)
        x_min=min(x_min,x[0])
        x_max=max(x_max,x[-1])
        y=[sum(f[i][j]) for j in x]
        #print(y)
        y_max=max(y_max,max(y))
        print(data[i][0], ni[i])
        #print(f[i])
        if data[i][0]:
            plt.plot([int(j) for j in x],y, marker='o',label=f'k={data[i][0]}',color=colors[i%20])
            plt.axvline(x=ni[i], ymin=0, ymax=100, color=colors[i%20], linestyle='--')
            plt.text(ni[i],i*(y_max//len(data)),f"$\widetilde n={ni[i]:.3f}$",rotation=40,
             bbox=dict(boxstyle="square",
                       ec=colors[i%20],
                       fc='w',
                       ))
    plt.legend()
    plt.xlabel("размер максимальной компоненты сильной связности")
    plt.ylabel(f"частота встречаемости (%)")
    plt.grid(ls=':')
    plt.xticks(list(range(x_min,x_max,(x_max-x_min)//6))+[x_max])
    #plt.xticks(range(x_min, x_max + 1))
    #plt.yticks([i*y_max/6 for i in range(6)]+[y_max])
    plt.title(f"n={n}, m={m}, количество тестов={K}, k - степень несогласованности")
    plt.show()
for m in m_s:
    data=[]
    for k_ in k_s[::5]:
        data.append([k_,open_data(K,n,m,k_)])
    plot_4(K,n,m,data)

def plot_5(K,n,m,data,part=False):
    print(f'plot5 n={n} m={m}')
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
        #print(x)
        x_min=min(x_min,x[0])
        x_max=max(x_max,x[-1])
        y=[sum(f[i][j]) for j in x]
        #print(y)
        y_max=max(y_max,max(y))
        print(data[i][0], ni[i])
        #print(f[i])
        """if data[i][0]:
            plt.plot([int(j) for j in x],y, marker='o',label=f'k={data[i][0]}',color=colors[i%20])
            plt.axvline(x=ni[i], ymin=0, ymax=100, color=colors[i%20], linestyle='--')
            plt.text(ni[i],i*(y_max//len(data)),f"$\widetilde n={ni[i]:.3f}$",rotation=40,
             bbox=dict(boxstyle="square",
                       ec=colors[i%20],
                       fc='w',
                       ))"""
    """plt.legend()
    plt.xlabel("размер максимальной компоненты сильной связности")
    plt.ylabel(f"частота встречаемости (%)")
    plt.grid(ls=':')
    plt.xticks(list(range(x_min,x_max,(x_max-x_min)//6))+[x_max])
    #plt.xticks(range(x_min, x_max + 1))
    #plt.yticks([i*y_max/6 for i in range(6)]+[y_max])
    plt.title(f"n={n}, m={m}, количество тестов={K}, k - степень несогласованности")
    plt.show()"""
    if part:
        plt.plot([d[0]/(n*(n-1)/2)*100 for d in data],ni)
        plt.xlabel("$\\frac{2k}{n^2-n}$ - доля несогласованности (%)")
    else:
        plt.plot([d[0] for d in data],ni)
        plt.xlabel("$k$ - степень несогласованности")
    plt.ylabel(f"$\widetilde n$ - среднеожидаемый размер макс. компоненты")
    plt.title(f"n={n}, m={m}, количество тестов={K}")
    plt.show()


for m in m_s:
    data=[]
    for k_ in k_s:
        data.append([k_,open_data(K,n,m,k_)])
    plot_5(K,n,m,data,part=True)