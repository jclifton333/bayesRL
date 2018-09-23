"""
Numba implementations of linear algebra...hopefully this will speed things up.
"""
from numba import njit, jit
import numpy as np


@njit
def sse(u, v):
  sum_ = 0.0
  n = u.size
  for i in range(n):
    sum_ += (u[i] - v[i])**2
  return sum_


@njit
def matrix_dot_vector(M, v):
  n, m = M.shape
  res = np.zeros(n)
  for row in range(n):
    for col in range(m):
      res[row] += M[row, col] * v[col]
  return res


@njit
def vector_dot_vector(u, v):
  res = 0.0
  n = u.size
  for i in range(n):
    res += u[i]*v[i]
  return res


@njit
def vector_outer_vector(u, v):
  n = u.size
  res = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      res[i, j] = u[i] * v[j]
  return res


@njit
def matrix_dot_matrix(A, B):
  nrow_a, ncol_a = A.shape
  nrow_b, ncol_b = B.shape
  res = np.zeros((nrow_a, ncol_b))
  for rownum in range(nrow_a):
    row = A[rownum, :]
    for colnum in range(ncol_b):
      col = B[:, colnum]
      res[rownum, colnum] = vector_dot_vector(row, col)
  return res


@njit
def sherman_woodbury(A_inv, u, v):
  # outer = np.outer(u, v)
  outer = vector_outer_vector(u, v)
  A_inv_times_outer = matrix_dot_matrix(A_inv, outer)
  # A_inv_times_outer = np.dot(A_inv, outer)
  num = matrix_dot_matrix(A_inv_times_outer, A_inv)
  A_inv_times_u = matrix_dot_vector(A_inv, u)
  # A_inv_times_u = np.dot(A_inv, u)
  denom = 1.0 + vector_dot_vector(A_inv_times_u, v)
  # denom = 1.0 + np.dot(A_inv_times_outer, v)
  return A_inv - num / denom

