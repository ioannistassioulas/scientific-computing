#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:28:22 2024

@author: ioannisangelotassioulas
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg

# Working on a new scientific computing model that helps compute the solutiuon
# to the discrete Helmholtz Equation

# Question 1
# Exact solution
def exact_solution(x, y):
    return (x * (1 - x) * y**3 * (1 - y))+ np.exp(x)

def f_function(x, y): # create the function f that guarantees u
    return 2*y**3*(1-y) - 6*x*y*(1-2*y)*(1-x) + np.exp(x) - (2j*(x*(1-x)*y**3*(1-y) + np.exp(x)))

def f_sol(x, y, N):
    f_sol = np.zeros(((N+1), (N+1)))
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            f_sol[i][j] = f_function(x[i], y[j])
    
    # Add certain boundary conditions
    f = np.vstack(([np.exp(k) for k in x[1:N]], np.zeros((N-3, N-1)), [np.exp(k) for k in x[1:N]]))
    g = np.hstack((np.vstack([np.exp(0)]*(N-1)), np.zeros((N-1, N-3)), (np.vstack([np.exp(1)]*(N-1)))))
    tot = np.pad(f + g, ((1, 1), (1, 1)), 'constant', constant_values = ((0, 0), (0, 0)))
    
    f_sol += tot
    
    # set boundary nodes to boundary conditions
    v = [np.exp(k) for k in x]
    
    f_sol[0] = v
    f_sol[N] = v
    f_sol[:, 0] = np.array([np.exp(x[0])])
    f_sol[:, N] = np.array([np.exp(x[N])])

    return f_sol.flatten()

def A(N, c):
    h = 1 / (N+1)
    A = (4 - (h**2 * c*1j)) * np.eye(N+1) - np.eye(N+1, k = 1) - np.eye(N+1, k = -1)
    
    A[:1, :] = 0
    A[-1:, :] = 0
    A[:, :1] = 0
    A[:, -1:] = 0

    
    A[0][0] = 1/(h**2)
    A[N][N] = 1/(h**2)
    return A

def B(N):
    B = -1*np.eye(N+1)
    B[0][0] = 0
    B[N][N] = 0
    
    return B

def create_matrix(N,c):
    h = 1 / N  # step division
    
    # discretize the grid
    x = np.arange(0, 1+h, h)
    y = np.arange(0, 1+h, h)
    
    f = f_sol(x, y, N)
    
    # construct the discretized block matrix
    blocks_A = [A(N,c) for i in range(N+1)]
    blocks_B = [B(N) for i in range(N)]
    matrix_A = sp.sparse.block_diag(blocks_A, format="csr").toarray()
    
    offset = np.empty((0, N+1), int)
    matrix_B1 = sp.linalg.block_diag(offset, *blocks_B, offset.T)
    matrix_B2 = sp.linalg.block_diag(offset.T, *blocks_B, offset)
    
    matrix = matrix_A + matrix_B1 + matrix_B2  

    
    #delete boundaries
    matrix[:N+1, :] = 0
    matrix[-(N+1):, :] = 0
    matrix[:, :(N+1)] = 0
    matrix[:, -(N+1):] = 0
    
    for i in range(len(matrix)):
        matrix[i][i] = (1 / (N+1))**2 if matrix[i][i] == 0 else matrix[i][i]
    
    return np.array(matrix), f
        
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


def direct_solver(A, f):
    
    # placeholders for the solutions
    y = np.zeros(len(A))
    u = np.zeros(len(A))
    
    # LU decomposition
    P, L, U = sp.linalg.lu(A)
    
    # solution from LU decomposition via forward and backward substitution
    for i in range(0, len(A)):
        y[i] = f[i] - np.matmul(L[i][1:i-1], y[1:i-1])
        
    for j in range(len(A)-1, -1, -1):
        u[j] = (y[j] - np.matmul(U[j][j+1:len(A)-1], u[j+1:len(A)-1])) / U[j][j]
    
    # Return the final solution
    return u

u_exact = np.array([])
for i in X:
    for j in Y:
        u_exact = np.append(u_exact, exact_solution(i, j))
print(u_exact)
u_16 = direct_solver(create_matrix(16, 2))


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

def iteration_sequence(A, f, u, err=0.1):
    '''
    Iteration sequence exists for two reasons:
        1) Recalculate the residual and check the stopping criteria
        2) Iterate over the Gauss Seidel Step 
    Parameters
    ----------
    A : n x n two dimensional integer array
        Discretized operator which which to iterate over
    f : n length integer array
        RHS of the equation
    u : n length integer array
        initial guess of solution to equation
    e : integer
        stopping criteria
        
    Returns
    -------
    u : final solution

    '''
    residual = f - np.matmul(A, u)  # calculate residual for stopping condition
    u_next = gauss_seidel(A, u)  # calculate next step
    print(np.linalg.norm(residual)/np.linalg.norm(f))
    stopping = np.linalg.norm(residual) / np.linalg.norm(f)
    if(stopping <= err):
        return np.array(np.zeros(len(A)))
        
    return np.vstack((u_next, iteration_sequence(A, f, u_next, err)))


iteration_record = iteration_sequence(A1, f, np.ones(16))



# Checking to see if everything is working