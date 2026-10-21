"""Вывод графиков по результатам lin_analysis.py"""
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

from prettytable import PrettyTable

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

def find_dir(K,n,m,model="M3", matrix_subsolve=True,k=-1,i=0,v=1):
    name_dir0 = f"lin_analysis_{model}"
    if not matrix_subsolve:
        name_dir0 += "_graph"
    if k==-1:
        files=glob.glob(name_dir0 + f'/lin_analysis_k{K}_n{n}_m{m}_*')
    else:
        files=glob.glob(name_dir0 + f'/lin_analysis_k{K}_{k}_n{n}_m{m}_*')
    if not len(files):
        if v:print(f"not found K={K}, n={n}, m={m}" + (k != -1)*f", k={k}")
        return None
    else:
        if i>=len(files):
            i=-1
        if v:print(f"found dir (i={i}): {files[i]}")
        return files[i]

def open_data(K,n,m,model="M3",matrix_subsolve=True,k=-1,v=1):
    dir=find_dir(K,n,m,model=model,matrix_subsolve=matrix_subsolve,k=k,v=v)
    d=None
    if dir is not None:
        with open(dir+'/data.json') as json_file:
            d=json.load(json_file)
    return d

def open_exec_times(K,n,m,model="M3",matrix_subsolve=True,k=-1,v=1):
    dir=find_dir(K,n,m,model=model,matrix_subsolve=matrix_subsolve,k=k,v=v)
    exec_times=None
    if dir is not None:
        with open(dir+'/exec_times.json') as json_file:
            exec_times=json.load(json_file)
    return exec_times

def open_index0(K,n,m,model="M3",matrix_subsolve=True,k=-1,v=1):
    dir=find_dir(K,n,m,model=model,matrix_subsolve=matrix_subsolve,k=k,v=v)
    index0=None
    if dir is not None:
        with open(dir+'/index0.json') as json_file:
            index0=json.load(json_file)
    return index0

def open_len_ans(K,n,m,model="M3",matrix_subsolve=True,k=-1,v=1):
    dir=find_dir(K,n,m,model=model,matrix_subsolve=matrix_subsolve,k=k,v=v)
    len_ans=None
    if dir is not None:
        with open(dir+'/len_ans.json') as json_file:
            len_ans=json.load(json_file)
    return len_ans

def open_n_max(K,n,m,model="M3",matrix_subsolve=True,k=-1,v=1):
    dir=find_dir(K,n,m,model=model,matrix_subsolve=matrix_subsolve,k=k,v=v)
    n_max=None
    if dir is not None:
        with open(dir+'/n_max.json') as json_file:
            n_max=json.load(json_file)
    return n_max

def plot_n_max_exec_time(K,n,m,k_):
    exec_time = open_exec_times(K, n, m, k=k_)
    if exec_time is None:
        return
    n_max = open_n_max(K, n, m, k=k_)
    colors = mpl.color_sequences['tab10']
    #colors=['black','dimgrey','darkgrey','gainsboro']

    n_max_s = dict()
    for i in range(K):
        if n_max[i] not in n_max_s.keys():
            n_max_s[n_max[i]] = []
        n_max_s[n_max[i]].append(exec_time[i])
    print(f"K={K}, n={n}, m={m}, k={k_} (r={2*k_/(n*(n-1))}), time={min(exec_time)}-{max(exec_time)}")
    f=[]
    x=[]
    is_label=False
    for nm, ets in sorted(n_max_s.items()):
        if is_label:
            plt.bar(nm, max(ets), color="red")
            plt.bar(nm, min(ets), color="blue")
        else:
            plt.bar(nm, max(ets), color="red",label="max")
            plt.bar(nm, min(ets), color="blue",label="min")
            is_label=True
        #print(nm, min(ets), sum(ets)/len(ets), max(ets))
        et = sum(ets)/len(ets)
        f.append(et)
        x.append(nm)
    plt.plot(x,f,label='среднее время',color="black")
    print()
    plt.legend()
    plt.xlabel("размер максимальной компоненты сильной связности")
    plt.ylabel(f"время решения (c)")
    plt.grid(ls=':')
    plt.title(f"$n={n}, m={m}$, количество тестов=${K}$"+(bool(k_)*f", \nстепень несогласованности $k={k_}$"))
    #plt.show()



def plot_k_exec_time(n,m, k_s=None, Ks=None):
    colors = mpl.color_sequences['tab10']
    # colors=['black','dimgrey','darkgrey','gainsboro']
    if k_s is None:
        k_s = list(range(580, 10, -10))
    if Ks is None:
        Ks=[1000,100,10]
    et=[[] for _ in range(len(k_s))]
    for i,k_ in enumerate(k_s):
        for j,K in enumerate(Ks):
            exec_time = open_exec_times(Ks[j], n, m, k=k_)
            et[i].append(exec_time)
            if exec_time is not None:
                t = sum(exec_time)/K
                print(k_, t)
                plt.bar(k_, t,color=colors[j])
    #plt.plot(k_s,et,label='среднее время',color="black")
    print()
    #plt.legend()
    plt.xlabel("степень несогласованности k")
    plt.ylabel(f"время решения (c)")
    plt.grid(ls=':')

    plt.title(f"$n={n}, m={m}$")
    plt.show()

if __name__=="__main__":
    t_k = []
    K = 10
    Ks = [1000, 100, 10, 1]
    n = 35
    k_s = list(range(10, 590, 10))
    #k_s=[50,100,120,140,560,570,580]
    k_s_K = {10: [130, 140, 150, 570], 100: [30, 60, 100, 110, 120, 560, 570, 580, 590],
             1000: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 570, 580, 590]}
    m_s = (5, 6, 7, 8)


    tab = PrettyTable(['m']+[f"k={k} (r={2*k/(n*(n-1)):.2f})" for k in k_s])
    tab.align['m'] = "r"
    for i,m in enumerate(m_s):
        t_k.append([])
        for j,k_ in enumerate(k_s):
            t = (None,None)
            for K in Ks:
                if k_==40:break
                if k_ ==150 and m==5:
                    break
                exec_time = open_exec_times(K, n, m, model="M3", matrix_subsolve=False, k=k_,v=0)
                if exec_time is not None:
                    t=[(round(min(exec_time),4),round(sum(exec_time)/K,4),round((max(exec_time)),4))[1],K]
                    break
            t_k[i].append(t)
        tab.add_row([m]+[ts for ts,K in t_k[i]])
        #print(t_k[i])
    print(tab.get_string())
    """for m in m_s:
        plot_k_exec_time(n,m)"""