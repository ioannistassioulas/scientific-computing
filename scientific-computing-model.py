#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:28:22 2024

@author: ioannisangelotassioulas
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import timeit as time
from scipy import linalg

# Working on a new scientific computing model that helps compute the solutiuon
# to the discrete Helmholtz Equation

# Question 1
# Exact solution
def exact_solution(x, y):
    return (x * (1 - x) * y**3 * (1 - y))+ np.exp(x)

def f_function(x, y): # create the function f that guarantees u
    # return 2*y**3*(1-y) - 6*x*y*(1-2*y)*(1-x) + np.exp(x) - (2j*(x*(1-x)*y**3*(1-y) + np.exp(x)))
    return (-2 * y**3 * (1 - y) + 6 * x * (1 - x) * y * (1 - 2 * y) + np.exp(x)) - 2j * (x * (1 - x) * y**3 * (1 - y) + np.exp(x))

def f_sol(x, y, N):
    h = 1 / N
    f_sol = np.zeros(((N+1), (N+1)))

    for i in range(1, N):
        for j in range(1, N):
            f_sol[i][j] = h**2 * f_function(x[i], y[j])

    # Apply boundary conditions
    # x=0 and x=1 boundaries
    f_sol[0, :] = 1.0  # u = 1 at x=0
    f_sol[N, :] = np.exp(1)  # u = e at x=1
    # y=0 and y=1 boundaries
    f_sol[:, 0] = np.exp(x)  # u = e^x at y=0
    f_sol[:, N] = np.exp(x)  # u = e^x at y=1

    return f_sol.flatten()

def A(N, c):
    h = 1 / N

    # Construct the diagonal and off-diagonal values using sparse matrices
    main_diag = (4 - h**2 * c * 1j) * np.ones(N+1)  # Main diagonal values
    off_diag = -np.ones(N)  # Off-diagonal values for the first superdiagonal and subdiagonal
    A_sparse = sp.sparse.diags([main_diag, off_diag, off_diag], [0, 1, -1], shape=(N+1, N+1), format="csc")

    # # Create sparse matrices for the diagonals
    # main_diag_sparse = sp.sparse.diags([main_diag, off_diag, off_diag], [0, 1, -1], shape=(N+1, N+1), format="csc")  # Main diagonal

    # Apply boundary conditions: Set the first and last rows/columns to zero
    A_sparse[0, 0] = 1
    A_sparse[0, 1:] = 0
    A_sparse[-1, -1] = 1
    A_sparse[-1, :-1] = 0

    # Correct the boundary values
    A_sparse[0, 0] = h**2
    A_sparse[N, N] = h**2

    return A_sparse

def B(N):
    B = sp.sparse.diags([-1], [0], shape=(N+1, N+1), format="csc")
    B = B.tolil()  # Convert to List of Lists format for modification

    # Set boundary rows to zero
    B[0, :] = 0
    B[N, :] = 0
    return B.tocsc()

def create_matrix(N,c):
    h = 1 / N  # step division
    
    # discretize the grid
    x = np.arange(0, 1+h, h)
    y = np.arange(0, 1+h, h)
    
    # find the correct 
    f = f_sol(x, y, N)
    
    # construct the discretized block matrix
    I = sp.sparse.eye(N+1, format='csc')
    A_block = A(N,c)
    B_block = sp.sparse.diags([-1, -1], [1, -1], shape=(N+1, N+1), format='csc')

    A_big = sp.sparse.kron(I, A_block, format="csc")  # Main diagonal blocks
    B_big = sp.sparse.kron(B_block, I, format="csc")  # Off-diagonal
    matrix = (A_big + B_big) / h**2
    
    return matrix, f
        
A1, f = create_matrix(3, -2)

# Plot the exact solution
N = 4
h = 1 / N

X = np.arange(0, 1+h, h)
Y = np.arange(0, 1+h, h)

x, y = np.meshgrid(X, Y)
z = (x * (1 - x) * y**3 * (1 - y)) + np.exp(x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, label="Exact Solution")

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#################################################

# Question 2
# Direct solver via LU-Decomposition
def error_finder(N):
    h = 1 / (N + 1)
    X = np.linspace(0, 1, N+1)
    Y = np.linspace(0, 1, N+1)
    
    # Exact solution
    u_exact = exact_solution(*np.meshgrid(X, Y)).flatten()
    
    # Solve
    A_mat, f_rhs = create_matrix(N, 2)  # c=2 as per problem statement
    u_h = sp.sparse.linalg.spsolve(A_mat, f_rhs)
    
    # Calculate maximum error
    error = np.max(np.abs(u_exact - u_h))
    return error

start = time.timeit()
print(error_finder(16) - np.exp(1))
end = time.timeit()
print(f"Time elapse: {end - start}")

start = time.timeit()
print(error_finder(32) - np.exp(1))
end = time.timeit()
print(f"Time elapse: {end - start}")

start = time.timeit()
print(error_finder(64) - np.exp(1))
end = time.timeit()
print(f"Time elapse: {end - start}")

start = time.timeit()
print(error_finder(128) - np.exp(1))
end = time.timeit()
print(f"Time elapse: {end - start}")

start = time.timeit()
print(error_finder(256) - np.exp(1))
end = time.timeit()
print(f"Time elapse: {end - start}")

answers = np.array([error_finder(2**(n+4)) for n in range(5)]) - np.exp(1)
h_values = np.array([1/(2**(n+4)) for n in range(5)])
test = np.array([2**(n+4) for n in range(5)])
print(test)
plt.plot(test, h_values, label="O(h^2) - Theoretical Errorr")
plt.plot(test, answers, label="Real Simulation Error")
plt.title("Scaling of Error")
plt.xlabel("Iteration Step")
plt.ylabel("Error")
plt.legend()
plt.show()

###########################################
# Question 3

# discretize and create my own matrix



# solve problem using Gauss Seidel
def gauss_seidel(A, u):
    '''
    Take a single step in the Gauss Seidel iteration method

    Parameters
    ----------
    A : 2-dimensional n x n integer array
        Matrix which represents the discretized operator
    u : previous iteration - trial function
    Returns
    -------
    u : 1-dimensional n length array
        The next iteration to the Gauss Seidel iteration method
    '''
    
    n = len(A)  # define the size of the matrix
    # take one step in Gauss Seidel
    for i in range(n):
        u[i] = (f[i] - np.matmul(A[i][1:i-1], u[1:i-1]) - np.matmul(A[i][i+1:n], u[i+1:n])) / A[i][i]
    return u

# def iteration_sequence(A, f, u, err=0.1):
#     '''
#     Iteration sequence exists for two reasons:
#         1) Recalculate the residual and check the stopping criteria
#         2) Iterate over the Gauss Seidel Step 
#     Parameters
#     ----------
#     A : n x n two dimensional integer array
#         Discretized operator which which to iterate over
#     f : n length integer array
#         RHS of the equation
#     u : n length integer array
#         initial guess of solution to equation
#     e : integer
#         stopping criteria
        
#     Returns
#     -------
#     u : final solution

#     '''
#     residual = f - np.matmul(A, u)  # calculate residual for stopping condition
#     u_next = gauss_seidel(A, u)  # calculate next step
#     print(np.linalg.norm(residual)/np.linalg.norm(f))
#     stopping = np.linalg.norm(residual) / np.linalg.norm(f)
#     if(stopping <= err):
#         return np.array(np.zeros(len(A)))
        
#     return np.vstack((u_next, iteration_sequence(A, f, u_next, err)))


# iteration_record = iteration_sequence(A1, f, np.ones(16))



# Checking to see if everything is working