from rand_test import get_test
from matrix_solve import MatrixSolve, get_R, get_P, get_C, scc
import datetime

n=35
m=9

N=5
def gener_N(n,m,N,Ni):
    print(f"start gener_N(n={n}, m={m}, N={N}, Ni={Ni})")
    k = 0#[]
    try:
        while N > 0:
            if k%100==0:
                date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
                print(f'\tk={k}\t{date_str}')
            str_test, n, m, c, r = get_test(n, m)
            P = get_P(get_R(r, n, m), c, n, m)
            C = get_C(P, n)
            D = scc(C, n)
            k+=1#k.append([len(Di) for Di in D])
            l = len(D)
            ni = max([len(Di) for Di in D])
            if ni < Ni:
                print(l, ni)
                print(D)
                date_str=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%f")
                print(date_str)
                f = open(f'test_n{n}_ni{ni}_D{l}_{date_str}.txt', 'w')
                f.write(str_test)
                f.write(f'\n{D}')
                f.close()
                N -= 1
        print(f'\tk={k}')
    except KeyboardInterrupt:
        print(f'\tk={k}')
#gener_N(20,9,5, 16)
#gener_N(25,9,5, 16)
gener_N(35,9,2, 19)
gener_N(30, 9, 5, 19)





