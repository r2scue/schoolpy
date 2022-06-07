import math
import numpy as np
from numpy.random import default_rng
import numpy.linalg as LA
import matplotlib.pyplot as plt


"""
Author: Kenny Kong
Date: 2021

-----
Monte Carlo integration Python module
"""


_DIM_ = 3
_N_ = int(1e6)
_a_ = np.array([1,-3,-1])
_b_ = np.array([4,4,1])
_E_ = 42 # volume


def _rho_(x):
	return math.exp(x[2])


def _inE_(_x_):
	"""
	Defines E

	Parameter(s)
	------------
	x : array-like
		Point to check if inside volume E

	Returns
	-------
	res : bool
		True if x in E and False otherwise
	"""

	# Defining V
	# ----------

	#if V is a sphere
	'''
	assert _DIM_ == 3
	center = np.array([0,0,0])
	radius = 1

	return LA.norm(x-center,2) <= radius
	'''

	#if V is a box or prism
	'''
	a = np.full((_DIM_,), _a_)
	b = np.full((_DIM_,), _b_)
	return all((x > a and x < b))
	'''

	# V torus defined...
	x = _x_[0]
	y = _x_[1]
	z = _x_[2]
	return all([z**2 + (math.sqrt(x**2 + y**2) - 3)**2 <= 1,
				x >= 1,
				y >= -3])
	
	
def mci(f, n_s, inE=_inE_):
	"""
	Integrates f(x, y, z) over volume E

	Pre-condition(s)
	----------------
	E is defined in _inE_. 

	Parameter(s)
	------------
	f : function
		function to be integrated

	n_s : int, optional
		number of Monte Carlo integration iterations

	dim : int
		number of spatial dimensions which V has
	"""

	rng = default_rng() # RNG
	nV = 0 # Initial count of random numbers in V
	fS = 0 # Initial sum of f in V
	x = np.zeros((_DIM_,))

	for i in range(n_s):
		# start mci itr i
		x = rng.uniform(_a_, _b_, size=(_DIM_,))
		if inE(x):
			nV += 1
			fS += _rho_(x)
		print(i+1,': x=',x,', in=',inE(x),', nV=',nV,', fS=',fS)
		# end mci itr i

	return _E_*fS/n_s # Vol(E) ~= (Vol(Cube)) * points in E / points in E or cube


def __main__():
	print(mci(_rho_,_N_))


if __name__ == '__main__':
	__main__()



















	# End of file