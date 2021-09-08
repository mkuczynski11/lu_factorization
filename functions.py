import numpy as np
from numpy import inf, linalg as LA
def create_equations(a1:int, a2:int, a3:int, N:int) -> np.ndarray:
    A:np.ndarray = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if(i == j):
                A[i][j] = a1
            elif(i == j-1 or i == j+1):
                A[i][j] = a2
            elif(i == j-2 or i == j+2):
                A[i][j] = a3
    return A

def create_vector(f:int, N:int) -> np.ndarray:
    b = np.fromfunction(lambda i: np.sin(i*(f+1)), (N,), dtype=float)
    return b

def jacobi(A:np.ndarray, b:np.ndarray, res:float) -> tuple[np.ndarray, int, list[float]]:
    size = b.size
    r = np.ones(size)
    n_r = r.copy()
    cur_res = LA.norm(np.matmul(A,r) - b)
    res_tab = [cur_res]
    i = 0
    while(cur_res > res and i < 1000):
        for j in range(size):
            sum_left, sum_right = 0,0
            # for k in range(0, j, 1):
            #     sum_left += A[j][k]*r[k]
            # for k in range(j+1, b.size, 1):
            #     sum_right += A[j][k]*r[k]
            n_r[j] = (b[j] - np.sum(A[j][0:j]*r[0:j]) - np.sum(A[j][j+1:size]*r[j+1:size]))/A[j][j]

        r = n_r.copy()
        cur_res = LA.norm(np.matmul(A,r) - b)
        res_tab.append(cur_res)
        i += 1
        if(cur_res == inf):
            return np.r,i,res_tab
    if(i == 1000):
        return r,i,res_tab
    return r,i,res_tab

def gauss_seidel(A:np.ndarray, b:np.ndarray, res:float) -> tuple[np.ndarray, int, list[float]]:
    size = b.size
    r = np.ones(size)
    n_r = r.copy()
    cur_res = LA.norm(np.matmul(A,r) - b)
    res_tab = [cur_res]
    i = 0
    while(cur_res > res and i < 1000):
        for j in range(size):
            # sum_left, sum_right = 0,0
            # for k in range(0, j, 1):
            #     sum_left += A[j][k]*n_r[k]
            # for k in range(j+1, b.size, 1):
            #     sum_right += A[j][k]*r[k]
            n_r[j] = (b[j] - np.sum(A[j][0:j]*n_r[0:j]) - np.sum(A[j][j+1:size]*r[j+1:size]))/A[j][j]

        r = n_r.copy()
        cur_res = LA.norm(np.matmul(A,r) - b)
        res_tab.append(cur_res)
        i += 1
    if(i == 1000):
        return r,i,res_tab
    return r,i,res_tab

def forward_sub(X:np.ndarray, b:np.ndarray) -> np.ndarray:
    size = int(np.sqrt(X.size))
    Y:np.ndarray = np.zeros(size, dtype=float)
    Y[0] = b[0] / X[0][0]
    for i in range(1,size):
        tmp = b[i]
        for j in range(0,i):
            tmp -= X[i][j] * Y[j]
        Y[i] = tmp / X[i][i]
    return Y

def back_sub(X:np.ndarray, b:np.ndarray) -> np.ndarray:
    size = int(np.sqrt(X.size))
    Y:np.ndarray = np.zeros(size, dtype=float)
    Y[-1] = b[-1] / X[-1][-1]
    for i in range(2,size+1):
        tmp = b[-i]
        for j in range(1,i+1):
            tmp -= X[-i][-j] * Y[-j]
        Y[-i] = tmp / X[-i][-i]
    return Y

def lu(A:np.ndarray, b:np.ndarray) -> tuple[np.ndarray, float]:
    U = np.copy(A)
    L = np.eye(b.size)
    m = b.size
    for k in range(0,m-1):
        for j in range(k+1, m):
            L[j][k] = U[j][k]/U[k][k]
            U[j][k:m] = U[j][k:m] - L[j][k]*U[k][k:m]
            # for i in range(k,m,1):
            #     U[j][i] = U[j][i] - L[j][k]*U[k][i]
    y = forward_sub(L, b)
    x = back_sub(U,y)
    res = LA.norm(np.matmul(A,x) - b)
    return x, res