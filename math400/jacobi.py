import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt


"""
Author: Kenny Kong
Date: 2021

-----
Iterative Jacobi and Gauss-Seidel methods for linear systems
"""


WPC = 15 # set working decimal precision
RPC = 10 # set return value decimal precision
PPC = 8  # set print decimal precision

_TOL_ = 5e-6
#_ITR_LIMIT_ = 1000


def jac_itr(A, x0, b):
	"""
	Jacobi method for square linear systems of form Ax=b. 

	Parameter(s)
	------------
	A : array-like
		Coefficient matrix, must be square. 

	x0 : array-like
		Initial guess of solution x. 

	b : array-like
		Right-hand-side of system. 

	Returns
	-------
	xnew : ndarray
		Numerical solution to the system Ax=b within plus/minus _TOL_. 
	"""

	A = np.array(A)   # square coefficient matrix
	x0 = np.array(x0) # intital guess for solution
	b = np.array(b)   # right-hand-side

	# Pre-conditions
	m, n = A.shape
	assert m == n           # check : A is square
	assert b.shape[0] == n  # check : b is shape-compatible
	assert x0.shape[0] == n # check : x0 is shape-compatible

	# A = L + D + U
	D_inv = LA.inv(np.diag(np.diag(A), k=0))
	print('D = ',LA.inv(D_inv))
	L = np.tril(A, k=-1)
	print('L = ',L)
	U = np.triu(A, k=1)
	print('U = ',U)
	

	diff = _TOL_ + 99.9 # dummy value : initial error
	xold = np.array(x0)
	xnew = np.matmul(np.matmul(-D_inv, L+U),xold) + np.matmul(D_inv, b)
	m = 1    # counter : number of Jacobi iterations
	print('Jacobi method. Tolerance = ',round(_TOL_,PPC),'.\n0:',xold,'\n1 : ',xnew,
			sep='')

	while (diff >= _TOL_):
		'''
		# start Jacobi method itr k
		for i in range(n):
			xnew[i] = (b[i]-np.sum(A[i,0:i]*xold[0:i])-np.sum(A[i,i+1:n]*xold[i+1:n]))/A[i,i]
		# end Jacobi method itr k
		'''
		xnew = np.matmul(np.matmul(-D_inv, L+U),xold) + np.matmul(D_inv, b)
		print('WTF : ',np.matmul(-D_inv, L+U))
		m += 1
		diff = LA.norm(xnew-xold, np.inf)
		print(m,' : ',np.round(xnew,PPC),' , diff = ',round(diff,PPC), sep='')
		xold = xnew

	return np.around(xnew, RPC)


def gs_itr(A, x0, b):
	"""
	Gauss-Seidel method for linear systems of form Ax=b. 

	Parameter(s)
	------------
	A : array-like
		Coefficient matrix, must be square. 

	x0 : array-like
		Initial guess of solution x. 

	b : array-like
		Right-hand-side of system. 

	Returns
	-------
	xnew : ndarray
		Numerical solution to the system Ax=b within plus/minus _TOL_. 
	"""

	A = np.array(A)   # square coefficient matrix
	x0 = np.array(x0) # intital guess for solution
	b = np.array(b)   # right-hand-side

	# Pre-conditions
	m, n = A.shape
	assert m == n
	assert b.shape[0] == n
	assert x0.shape[0] == n

	# A = L + D + U
	D = np.diag(np.diag(A), k=0)
	print('D = ',D)
	L = np.tril(A, k=-1)
	print('L = ',L)
	U = np.triu(A, k=1)
	print('U = ',U)

	diff = _TOL_ + 99.9 # dummy value : initial error
	xold = np.array(x0)
	xnew = (np.matmul(np.matmul(-LA.inv(L + D),np.matmul(U,xold)),xold) + 
			np.matmul(LA.inv(L + D),b))
	m = 1    # counter : number of Jacobi iterations
	print('Gauss-Seidel method. Tolerance = ',round(_TOL_,PPC),'.\n1 : ',xold, sep='')

	while (diff >= _TOL_):
		'''
		# start G-S method itr k
		for i in range(n):
			xnew[i] = (b[i]-np.sum(A[i,0:i]*xnew[0:i])-np.sum(A[i,i+1:n]*xold[i+1:n]))/A[i,i]
		# end G-S method itr k
		'''
		xnew = (np.matmul(np.matmul(-LA.inv(L+D),np.matmul(U,xold)),xold) + 
				np.matmul(LA.inv(L+D),b))
		m += 1
		diff = LA.norm(xnew-xold, np.inf)
		print(m,' : ',np.round(xnew,PPC),' , diff = ',round(diff,PPC), sep='')
		xold = np.array(xnew)

	return np.around(xnew, RPC)


def __main__():
	A = np.array( [ [4.0, -1.0, 0, -2.0, 0, 0], 
					[-1.0, 4.0, -1.0, 0, -2.0, 0], 
					[0, -1.0, 4.0, 0, 0, -2.0], 
					[-1.0, 0, 0, 4.0, -1.0, 0], 
					[0, -1.0, 0, -1.0, 4.0, -1.0], 
					[0, 0, -1.0, 0, -1.0, 4.0]] )

	x0 = np.zeros(6)

	b = np.array([-1.0, 0, 1.0, -2.0, 1.0, 2.0])

	gs = gs_itr(A,x0,b)
	print(gs)
	print('Check: ',np.round(np.matmul(A,gs),PPC))
	jac = jac_itr(A,x0,b)
	print(jac)
	print('Check: ',np.round(np.matmul(A,jac),PPC))

	return 0


if __name__ == '__main__':
	__main__()






















