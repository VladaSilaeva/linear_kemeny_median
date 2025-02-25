import numpy as np
def r2R(n,r,m, R):
    for k in range(m):
        a = np.array([[-1] * n for _ in range(n)])
        for i in r[k]:
            a[i - 1][i - 1] = 1
            for j in range(n):
                if a[i - 1][j] == -1:
                    a[i - 1][j] = 1
            for j in range(n):
                if a[j][i - 1] == -1:
                    a[j][i - 1] = 0
        R.append(a)

def get_P(n, c, R, m):
    P = np.zeros((n, n))
    for k in range(m):
        P += c[k] * R[k]
    return P

class MatrixSolve:
    def __init__(self, n, r, output_file, log_file):
        self.output = 'res_default' if output_file is None else output_file
        self.log = 'log_default' if log_file is None else log_file
        self.n = n
        self.m = len(r)
        self.P = np.zeros((n, n))
        if self.m and (len(r[0]) == 2):
            R = []
            for k in range(self.m):
                a = np.array([[-1] * n for _ in range(n)])
                for i in r[0][k]:
                    a[i - 1][i - 1] = 1
                    for j in range(n):
                        if a[i - 1][j] == -1:
                            a[i - 1][j] = 1
                    for j in range(n):
                        if a[j][i - 1] == -1:
                            a[j][i - 1] = 0
                R.append(a)

            for k in range(self.m):
                self.P += r[1][k] * R[k]
        else:
            self.P=np.array(r).T

        self.P1 = np.zeros((n, n))
        self.P1[:] = self.m
        self.P1 -= self.P

        self.Px = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.Px[i][j] = min(self.P[i][j], self.P1[i][j])

    def solve(self):
        f = open(self.log, 'w')
        
        f.write(f'P = {self.P}\n')
        f.write(f'P1 = {self.P1}\n')
        f.write(f'Px = {self.Px}\n')
        f.close()
        return self.m

    
def main():
    n = 5
    r = [[3, 5, 2, 4, 1], [4, 5, 2, 3, 1], [1, 3, 5, 2, 4], [1, 2, 4, 3, 5], [5, 3, 2, 1, 4], [1, 4, 5, 2, 3]]
    c = [0.5, 1, 2, 0.5, 0.5, 1.5]
    m = len(r)

if __name__ == "__main__":
    main()