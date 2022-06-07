import numpy as np
import matplotlib.pyplot as plt


"""
Author: Kenny Kong
Date: 2021

-----
Gaussian elimination using Python
"""


WPC = 15 # set working decimal precision
RPC = 10 # set return value decimal precision
PPC = 8  # set print decimal precision


def backsub(U, b):
	"""
	Back-substitution for square upper triangular system Ux = b

	Parameter(s)
	------------
	U : array-like
		Upper triangular square coefficient matrix

	b : array-like
		Right hand side
	"""
	U = np.around(U,WPC)
	b = np.around(b,WPC)
	m, n = U.shape

	assert m >= n
	assert b.shape[0] == m
	assert np.allclose(U, np.triu(U)) # check if U is upper triangular

	x = np.zeros((m,))
	x[-1] = np.around(b[-1]/U[-1,-1],WPC)
	print('\nBack-sub. routine started.', sep='')

	rhs = 0
	for i in range(-2, -m-1, -1): # row back-sub itr
		rhs = b[i]
		for j in range(i, 0, 1):
			rhs = rhs - U[i,j]*x[j]
		x[i] = round(rhs/U[i,i],WPC)

	return np.around(x,RPC)


def foresub(L, b):
	"""
	Foreward-substitution for lower triangular system Lx = b
	Especially for use with LU_decomp=True
	"""
	L = np.around(L,WPC)
	b = np.around(b,WPC)
	m, n = L.shape

	assert m >= n
	assert b.shape[0] == m
	assert np.allclose(L, np.triu(L)) # check if L is lower triangular

	x = np.zeros((m,))
	x[-1] = np.around(b[-1]/L[-1,-1],WPC)
	print('\nForeward-sub. routine started.', sep='')

	rhs = 0
	for i in range(-2, -m-1, -1): # row foreward-sub itr
		rhs = b[i]
		for j in range(i, 0, 1):
			rhs = rhs - L[i,j]*x[j]
		x[i] = round(rhs/L[i,i],WPC)

	return np.around(x,RPC)


def spp_elim(A, b, LU_decomp=True):
	"""
	Gaussian elimination with scaled partial pivoting for square matrix A.

	Parameter(s)
	------------
	A : array-like

	b : array-like

	LU_decomp : bool, optional

	Returns
	-------
	_res_ : array-like
		Numerical solution to the system Ax=b. 

	P, L, U : array-like, optional
	Optionally returns PLU decomposition of A, i.e. PA = LU (working)
	"""
	A = np.around(A,WPC)
	b = np.around(b,WPC)
	m, n = A.shape
	assert m == n
	assert b.shape[0] == m

	P = np.arange(0, m)
	M = np.identity(m)
	L = np.identity(m)
	U = np.array((m,m))
	
	print('\nGaussian elimination with scaled partial pivoting.\n')
	print('A init:\n',np.around(A,PPC), sep='')
	print('b init: ',np.around(b,PPC), sep='')
	print()

	for j in range(n): # col pivoting itr
		pivitr = np.argmax( 
						[A[i,j]/np.max(A[i,:]) for i in range(j,n)] 
				) # maximizes A[j+pivitr,j]/max{A[j+pivitr,j:m]} 

		A[[j,j+pivitr],:] = A[[j+pivitr,j],:] # row swap (LHS)
		b[[j,j+pivitr]] = b[[j+pivitr, j]] # row swap (RHS)

		#print('Pivot Col ',j,': \nswap rows ',j,' and ',j+pivitr,':\n',np.around(A,PPC), sep='')
		#print('b = ',np.around(b,PPC), sep='')
		#print()
		
		for i in range(j+1,n): # row elimination itr
			if not (A[j,j] == 0):
				mult = round(A[i,j]/A[j, j],WPC) # multiplier
				A[i,:] = np.around(A[i,:]-mult*A[j,:],WPC) # row op LHS
				b[i] = np.around(b[i]-mult*b[j],WPC) # row op RHS
				#print('R',i,'->R',i,'-(',round(mult,PPC),')(R',j,')', sep='')

		print()
		print('A = ',np.around(A,PPC), sep='')
		print('b = ',np.around(b,PPC), sep='')
		print()

	_res_ = backsub(A,b)
	if not LU_decomp:
		return _res_

	return _res_, np.around(P,RPC), np.around(L,RPC), np.around(U,RPC)


# Gaussian elimination with partial pivoting for square mx A
def pp_elim(A, b, LU_decomp=True):
	"""
	Gaussian elimination with partial pivoting for square matrix A.

	Parameter(s)
	------------
	A : array-like

	b : array-like

	LU_decomp : bool, optional

	Returns
	-------
	_res_ : array-like
		Numerical solution to the system Ax=b. 

	P, L, U : array-like, optional
	Optionally returns PLU decomposition of A, i.e. PA = LU (not working)
	"""
	A = np.around(A,WPC)
	b = np.around(b,WPC)
	m, n = A.shape
	assert m == n
	assert b.shape[0] == m
	
	print('\nGaussian elimination with partial pivoting.\n')
	print('A init:\n',np.around(A,PPC))
	print('b init: ',np.around(b,PPC), sep='')
	print()

	for j in range(n): # col pivoting itr
		piv_itr = np.argmax(A[j:,j]) # row ind of largest entry in col j
		A[[j,j+piv_itr],:] = A[[j+piv_itr,j],:] # row swap (LHS)
		b[[j,j+piv_itr]] = b[[j+piv_itr, j]] # row swap (RHS)

		#print('Pivot Col ',j,': \nswap rows ',j,' and ',j+piv_itr,':\n',np.around(A,PPC), sep='')
		#print('b = ',np.around(b,PPC), sep='')
		#print()
		
		for i in range(j+1,n): # row elimination itr
			if not (A[j,j] == 0):
				mult = round(A[i,j]/A[j, j],WPC) # multiplier
				A[i,:] = np.around(A[i,:]-mult*A[j,:],WPC) # row op LHS
				b[i] = np.around(b[i]-mult*b[j],WPC) # row op RHS
				#print('R',i,'->R',i,'-(',round(mult,PPC),')(R',j,')', sep='')

		#print()
		#print('A = ',np.around(A,PPC), sep='')
		#print('b = ',np.around(b,PPC), sep='')
		#print()

	_res_ = backsub(A,b)
	if not LU_decomp:
		return _res_

	return _res_


def __main__():
	"""
	A1 = [[0.2115, 2.296, 2.715, 3.215], 
		[0.4371, 3.916, 1.683, 2.852], 
		[6.099, 4.324, 23.2, 1.578], 
		[4.623, 0.8926, 15.32, 5.305]]
	b = [8.438, 8.888, 35.2, 26.14]

	#soln_1bi = np_elim(A1)
	#print('solution for (1bi):\n',soln_1bi)

	soln_pp = pp_elim(A1, b)
	soln_spp = spp_elim(A1, b)
	print('Solution with partial pivoting:\n', np.around(soln_pp,RPC))
	print('\nCheck:\nAx = ',np.around(np.matmul(A1, soln_pp),RPC), sep='')
	print()
	print('Solution with scaled partial pivoting:\n', np.around(soln_spp,RPC))
	print('\nCheck:\nAx = ', np.around(np.matmul(A1, soln_spp),RPC), sep='')
	print()
	print('b = ',np.around(np.array(b),RPC), sep='')
	print()
	"""


if __name__ == '__main__':
	__main__()



















