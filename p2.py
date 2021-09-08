#Martin Kuczyński
import numpy as np
import functions as fn
import time
import matplotlib.pyplot as pt

#Define constraints
#Index = 180199
d = 9
c = 9
e = 1
f = 0
N = 999
res = 10**(-9)

def main():
    #Zadanie A
    print("Task A ------------------------------")
    A = fn.create_equations(5+e, -1, -1, N)
    b = fn.create_vector(f, N)
    
    #Zadanie B
    print("Task B ------------------------------")
    print("Task B: Starting Jacobi method computing")
    start = time.time()
    solution_jacobi,iterations_jacobi,residuum_jacobi = fn.jacobi(A, b, res)
    end = time.time()
    # print(f'Jacobi method did {iterations_jacobi} iterations and eneded up with the result of:')
    print(f'Jacobi method did {iterations_jacobi} iterations and residuum ={residuum_jacobi[-1]}')
    print(f'Jacobi took {end-start} seconds')
    # print(solution_jacobi)
    print("Task B: Starting Gauss_Seidel method computing")
    start = time.time()
    solution_gauss_seidel,iterations_gauss_seidel,residuum_gauss_seidel = fn.gauss_seidel(A, b, res)
    end = time.time()
    # print(f'Gauss_Seidl method did {iterations_gauss_seidel} iterations and eneded up with the result of:')
    print(f'Gauss_Seidl method did {iterations_gauss_seidel} iterations and residuum = {residuum_gauss_seidel[-1]}')
    print(f'Gauss_Seidl took {end-start} seconds')
    # print(solution_gauss_seidel)

    pt.plot(np.arange(len(residuum_jacobi)),residuum_jacobi, label="Jacobi")
    pt.plot(np.arange(len(residuum_gauss_seidel)),residuum_gauss_seidel, label="Gauss-Seidel")
    pt.yscale('log')
    pt.xlabel("Ilość iteracji")
    pt.ylabel("Norma wektora residuum")
    pt.legend()
    pt.savefig('./images/ZadanieB_2.png')
    pt.clf()

    #Zadanie C
    print("Task C ------------------------------")
    A = fn.create_equations(3, -1, -1, N)
    print("Task C: Starting Jacobi method computing")
    start = time.time()
    solution_jacobi,iterations_jacobi,residuum_jacobi = fn.jacobi(A, b, res)
    end = time.time()
    # print(f'Jacobi method did {iterations_jacobi} iterations and eneded up with the result of:')
    print(f'Jacobi method did {iterations_jacobi} iterations and ended with residuum ={residuum_jacobi[-1]}')
    print(f'Jacobi took {end-start} seconds')
    # print(solution_jacobi)
    print("Task C: Starting Gauss_Seidel method computing")
    start = time.time()
    solution_gauss_seidel,iterations_gauss_seidel,residuum_gauss_seidel = fn.gauss_seidel(A, b, res)
    end = time.time()
    # print(f'Gauss_Seidl method did {iterations_gauss_seidel} iterations and eneded up with the result of:')
    print(f'Gauss_Seidl method did {iterations_gauss_seidel} iterations and ended with residuum = {residuum_gauss_seidel[-1]}')
    print(f'Gauss_Seidl took {end-start} seconds')
    # print(solution_gauss_seidel)

    pt.plot(np.arange(len(residuum_jacobi)),residuum_jacobi, label="Jacobi")
    pt.plot(np.arange(len(residuum_gauss_seidel)),residuum_gauss_seidel, label="Gauss-Seidel")
    pt.yscale('log')
    pt.xlabel("Ilość iteracji")
    pt.ylabel("Norma wektora residuum")
    pt.legend()
    pt.savefig('./images/ZadanieC_2.png')
    pt.clf()

    #Zadanie D
    print("Task D ------------------------------")
    print("Task D: Starting LU method computing")
    start = time.time()
    solution_lu,residuum_lu = fn.lu(A, b)
    end = time.time()
    # print(f'LU method eneded up with the result of:')
    print(f'LU method ended with residuum = {residuum_lu}')
    print(f'LU took {end-start} seconds')
    # print(solution_lu)
    
    #Zadanie E
    print("Task E ------------------------------")
    n_tab = np.linspace(500,4000,num=8, dtype=int)
    jacobi_tab = np.zeros(n_tab.size)
    gauss_seidel_tab = np.zeros(n_tab.size)
    lu_tab = np.zeros(n_tab.size)
    print("Task E: Starting simulation for:")
    print(n_tab)
    for i in range(n_tab.size):
        A = fn.create_equations(5+e,-1,-1,n_tab[i])
        b = fn.create_vector(f,n_tab[i])

        print(f'Task E: Jacobi --------')
        print(f'Task E: Starting Jacobi method for {n_tab[i]}')
        start = time.time()
        fn.jacobi(A,b,res)
        end = time.time()
        jacobi_tab[i] = end-start
        print(f'Task E: Finished Jacobi method for {n_tab[i]} with time {end-start}')

        print(f'Task E: Gauss-Seidel --------')
        print(f'Task E: Starting Gauss-Seidel method for {n_tab[i]}')
        start = time.time()
        fn.gauss_seidel(A,b,res)
        end = time.time()
        gauss_seidel_tab[i] = end-start
        print(f'Task E: Finished Gauss-Seidel method for {n_tab[i]} with time {end-start}')

        print(f'Task E: LU --------')
        print(f'Task E: Starting LU method for {n_tab[i]}')
        start = time.time()
        fn.lu(A,b)
        end = time.time()
        lu_tab[i] = end-start
        print(f'Task E: Finished LU method for {n_tab[i]} with time {end-start}')

    pt.plot(n_tab,jacobi_tab, label="Jacobi")
    pt.plot(n_tab,gauss_seidel_tab, label="Gauss-Seidel")
    pt.plot(n_tab,lu_tab, label="LU")
    pt.xlabel("N")
    pt.ylabel("Czas[s]")
    pt.legend()
    pt.savefig('./images/ZadanieE_3.png')
    pt.clf()
    return

if __name__ == "__main__":
    main()