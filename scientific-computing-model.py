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

# Plot the exact solution
X = np.arange(0, 1, 0.1)
Y = np.arange(0, 1, 0.1)
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
def direct_homebrew(A, n):
    '''
    My homebrew of the direct solver, built according to the specifications of
    the pseudocode given in the textbook. To be used to compare to the scipy
    black box.
    
    Keywords:
    A: Matrix to be decomposed
    n: length of the square matrix
    '''
    L = np.identity(n)
    for k in range(n):
        if A[k][k] == 0:  # make sure that we can still pivot
            return 0
        else:
            maximum = np.argmax(A, axis = 0)[k]
            place = A[k]
            A[k] = A[maximum]
            A[maximum] = place
                
            for i in range(k+1, n):
                L[i][k] = (A[i][k] / A[k][k])  #  alpha coefficient
                A[i][k] = L[i][k]
                for j in range(k+1, n):
                    A[i][j] = A[i][j] - (L[i][k] * A[k][j])
    return L

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
    return u, L, U


###########################################
# Question 3

# discretize and create my own matrix

def A(N):
    A = 4 * np.eye(N+1) - np.eye(N+1, k = 1) - np.eye(N+1, k = -1)
    
    A[:1, :] = 0
    A[-1:, :] = 0
    A[:, :1] = 0
    A[:, -1:] = 0

    
    A[0][0] = 1/((N+1)**2)
    A[N][N] = 1/((N+1)**2)
    return A

def B(N):
    B = -1*np.eye(N+1)
    B[0][0] = 0
    B[N][N] = 0
    
    return B

def create_matrix(N):
    h = 1 / (N+1)  # step division
    
    # discretize the grid
    x = np.arange(0, 1, h)
    y = np.arange(0, 1, h)
    X, Y = np.meshgrid(x, y)
    
    # construct the discretized block matrix
    blocks_A = [A(N) for i in range(N+1)]
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
    
    return np.array(matrix)
        
print(create_matrix(3))

A = create_matrix(3)
f = np.random.randint(1, 10, size=len(A))
print(f.shape)
sol, lower, upper= direct_solver(A,f)


L_home  = direct_homebrew(A, len(A))

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

def iteration_sequence(A, f, u, err=0.028):
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
    if(np.linalg.norm(residual)/np.linalg.norm(f) < err):
        return np.array(np.zeros(len(A)))
    
    return np.vstack((u_next, iteration_sequence(A, f, u_next)))


iteration_record = iteration_sequence(A, f, np.ones(16))
print(iteration_record)


# Checking to see if everything is working