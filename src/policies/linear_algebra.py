"""
Numba implementations of linear algebra...hopefully this will speed things up.
"""
from numba import njit, jit
import numpy as np
import pdb


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


def update_linear_model(X, y, Xprime_X, Xprime_X_inv, x_new, X_dot_y, y_new):
  X_new = np.vstack((X, x_new.reshape(1, -1)))
  X_dot_y_new = X_dot_y + y_new * x_new

  if Xprime_X_inv is None:  # Can't do fast update
    Xprime_X_new = np.dot(X.T, X)
    Xprime_X_inv_new = np.linalg.inv(Xprime_X_new + 0.01*np.eye(X.shape[1]))
  else:
    # Compute new beta hat and associated matrices
    Xprime_X_inv_new = sherman_woodbury(Xprime_X_inv, x_new, x_new)
    Xprime_X_new = Xprime_X + np.outer(x_new, x_new)

  beta_hat_new = matrix_dot_vector(Xprime_X_inv_new, X_dot_y_new)

  # Compute new sample covariance
  n, p = X_new.shape
  yhat = matrix_dot_vector(X_new, beta_hat_new)
  # sigma_hat = np.sum((yhat - y_new)**2) / (n - p)
  y = np.append(y, y_new)
  sigma_hat = np.sqrt(sse(yhat, y) / np.max((1.0, n - p)))
  sample_cov = sigma_hat * Xprime_X_inv_new

  return {'beta_hat': beta_hat_new, 'Xprime_X_inv': Xprime_X_inv_new, 'X': X_new, 'y': y, 'X_dot_y': X_dot_y_new,
          'sample_cov': sample_cov, 'sigma_hat': sigma_hat, 'Xprime_X': Xprime_X_new}
