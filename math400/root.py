import time
import math
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt
from gelim import pp_elim


"""
Author: Kenny Kong
Date: 2021

-----
Rootfinding using Python, solving f(x) = 0
"""

PPC = 8 # set print precision
MAX_ITR = 50


# f
def _f_(x):
	"""
	f(x)

	x : float
	"""
	return math.tanh(x)


# f prime
def _fp_(x):
	"""
	f'(x)

	x : float
	"""
	return 1 - (math.tanh(x)**2)


def newton(x0, _TOL_=1e-8, make_plot=False):
	"""
	Solves f(x) = 0

	Parameter(s)
	------------
	x0 : array-like

	_TOL_ : float, optional

	make_plot : bool, optional
	"""
	xo = x0 # old x
	xn = xo - _f_(xo)/_fp_(xo) # new x
	n = 1 # iteration counter
	temp = [_fp_(xo)]
	print('Newton\'s method\nInitial guess: ',x0,'\nItr 1: c = ',xn,', f(c) = ',_f_(xn),
		sep='')

	while (abs(_f_(xn)) > _TOL_) and (n < MAX_ITR):
		xo = xn
		if (_fp_(xo) != 0):
			xn = xo - _f_(xo)/_fp_(xo) # Newton's method equation
			temp.append(_fp_(xo))
		else: 
			temp.append(0)
			print('SkipItr: division by zero')
		n += 1
		print('Itr ',n,': c=',round(xn,PPC),', f(c)=',round(abs(_f_(xn)),PPC),
			', f\'(c)=',temp[-1],sep='')

	# plots the tangent at each iteration of the method
	if make_plot:
		plt.figure()
		plt.title('Newton\'s method: f\'(c[i]) for i=1,2,...')
		plt.xlabel('i')
		plt.ylabel('f\'(c[i])')
		plt.plot(range(1,len(temp)+1),temp)
		plt.show()

	return xn


def _F_(_x_):
	x = _x_[0]
	y = _x_[1]
	z = _x_[2]

	# 1.
	return [ x**2 + y**2 + z**2 - 1,
			 x**2 + z**2 - 0.25,
			 x**2 + y**2 - 4*z]


def _JF_(_x_):
	"""
	Jacobian of _F_

	_x_ : array-like
	"""
	x = _x_[0]
	y = _x_[1]
	z = _x_[2]

	# 1.
	J11,J12,J13 = (2*x, 2*y, 2*z)
	J21,J22,J23 = (2*x, 0, 2*z)
	J31,J32,J33 = (2*x, 2*y, -4)

	J = [[J11, J12, J13],
		 [J21, J22, J23],
		 [J31, J32, J33]]
	return J


def _G_(_x_):
	x = _x_[0]
	y = _x_[1]
	z = _x_[2]

	# 2.
	return [ x**2 + y**2 + z**2 - 10,
			x + 2*y - 2,
			x + 3*z - 9 ]


def _JG_(_x_):
	"""
	Jacobian of _G_

	_x_ : array-like
	"""
	x = _x_[0]
	y = _x_[1]
	z = _x_[2]

	# 2.
	J11,J12,J13 = (2*x, 2*y, 2*z)
	J21,J22,J23 = (1, 2, 0)
	J31,J32,J33 = (1, 0, 3)

	J = [[J11, J12, J13],
		 [J21, J22, J23],
		 [J31, J32, J33]]
	return J


def _H_(_x_):
	x = _x_[0]
	y = _x_[1]
	z = _x_[2]

	# 3.
	return [ x**2 + 50*x + y**2 + z**2 - 200,
			x**2 + 20*y + z**2 - 50,
			-x**2 - y**2 + 40*z + 75 ]


def _JH_(_x_):
	"""
	Jacobian of _H_

	_x_ : array-like
	"""
	x = _x_[0]
	y = _x_[1]
	z = _x_[2]

	#3. 
	J1 = (2*x+50, 2*y, 2*z)	# Row 1
	J2 = (2*x, 20, 2*z)		# Row 2
	J3 = (-2*x, -2*y, 40)	# Row 3

	J = [J1,
		 J2,
		 J3]
	return J


def newton_sys(x0, _FN_, _JAC_, _TOL_=5e-6):
	"""
	Newton's method for square systems _FN_(x) = [0 ... 0]
	
	Parameter(s)
	------------
	x0 : array-like

	_FN_ : function

	_JAC_ : function

	_TOL_ : float, optional
	"""
	xo = np.array(x0) 
	J = np.array(_JAC_(xo))
	yn = pp_elim(-J, _FN_(xo), LU_decomp=False) # elimination with partial pivoting
	xn = yn+xo
	fc = _FN_(xn)
	tol = np.full(xn.shape, _TOL_)
	zero = np.zeros(xn.shape)
	n = 1
	print('Newton\'s method for systems.\nItr 1: x0=',np.round(xo,PPC),', c=',np.round(xn,PPC),
		', F(c)=',np.round(fc,PPC),sep='')

	while (not np.all(np.abs(fc) <= tol)) and (n < MAX_ITR):
		xo = xn		# xold
		J = np.array(_JAC_(xo))
		yn = pp_elim(-J, _FN_(xo), LU_decomp=False)	# yn = xn - xo
		xn = yn+xo	# xnew
		fc = _FN_(xn)
		n += 1
		print('Itr ',n,': c=',np.round(xn,PPC), ', F(c)=',np.round(fc,PPC),sep='')

	return xn


def secant(x0, x1, _TOL_=1e-8):
	"""
	Solves f(x) = 0
	"""
	xoo = x0
	xo = x1
	xn = (xoo*_f_(x0) - xo*_f_(xoo)) / (_f_(xo) - _f_(xoo))
	n = 1
	print('Secant method\nInitial guesses x0, x1: ',x0,', ',x1,'\nItr 1: ',xn,sep='')

	while (abs(_f_(xn)) > _TOL_) and (n < MAX_ITR):
		xoo = xo
		xo = xn
		if (_f_(xo)-_f_(xoo) != 0):
			xn = xo - _f_(xo)*(xo-xoo)/(_f_(xo)-_f_(xoo))
		n += 1
		print('Itr ',n,': c = ',round(xn,PPC), ', f(c) = ',round(abs(_f_(xn)),PPC),sep='')

	return xn


def bisection(x0, x1, _TOL_=1e-8):
	"""
	Solves f(x) = 0

	Parameter(s)
	------------
	x0, x1 : float
	"""
	a = min([x0,x1])
	b = max([x0,x1])
	c = (a + b) / 2
	fa = _f_(a)
	fb = _f_(b)
	fc = _f_(c)
	n = 1
	print('Start bisection method.\nItr 1 : a = ',a,', b = ',b,', c = ',c,
			', f(c) = ',fc,sep='')

	while (abs(c) > _TOL_) and (n < MAX_ITR):
		if fa*fc < 0:
			b = c
		else:
			a = c
		c = (a + b) / 2
		fa = _f_(a)
		fb = _f_(b)
		fc = _f_(c)
		n += 1
		print('Itr ',n,' : a = ',round(a,PPC),', b = ',round(b,PPC),', c = ',
			round(c,PPC),', f(c) = ',round(fc,PPC),sep='')

	return c


def bisectionNewton(x0, x1, s=0.1, _TOL_=1e-8):
	a = min([x0,x1])
	b = max([x0,x1])
	c = (a + b) / 2
	fa = _f_(a)
	fb = _f_(b)
	fc = _f_(c)
	n = 1
	print('Start bisection method.\nItr 1 : a = ',a,', b = ',b,', c = ',c,
			', f(c) = ',fc,sep='')

	while (abs(a-b) > s*(abs(x1-x0))) and (n < MAX_ITR):
		if fa*fc < 0:
			b = c
		else:
			a = c
		c = (a + b) / 2
		fa = _f_(a)
		fb = _f_(b)
		fc = _f_(c)
		n += 1
		print('Itr ',n,' : a = ',round(a,PPC),', b = ',round(b,PPC),', c = ',
			round(c,PPC),', f(c) = ',round(fc,PPC),sep='')

	return newton(c, _TOL_)


def __main__():
	
	print('Problem 2a.')
	newton(1.08, make_plot=False)
	newton(1.09, make_plot=False) # diverges
	print('\nProblem 2b.')
	secant(1.08, 1.09)
	secant(1.09, 1.1)
	secant(1, 2.3) # diverges
	secant(1, 2.4) # diverges
	print('\nProblem 2c.')
	bisection(-5,3)
	print('\nProblem 2d.')
	bisectionNewton(-10,15)
	#print('\nProblem 3.1')
	#r1=newton_sys( [1,1,1], _F_, _JF_ )
	#print('\nProblem 3.2')
	#r2=newton_sys( [2,0,2], _G_, _JG_ )
	print('\nProblem 3.3')
	r3=newton_sys( [2,2,2], _H_, _JH_ )
	print('Check: H(x) = ',np.round(_H_(r3),PPC))


if __name__ == '__main__':
	__main__()















# End of file