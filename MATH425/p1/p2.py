# Kenny Kong MATH 425 Group 6 Project 1 Problem 2
## Christina Cai, Jiajian Li, Jocelyn Padilla, Danny Lopez
### intermediate calculations done/verified on TI-nspire


import numpy as np 
import matplotlib.pyplot as plt 


# 2.a.)
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) 
y = np.array([0, 8.8, 29.9, 62, 104.7, 159.1, 222, 294.5, 380.4, 471.1, 571.7, 
			686.8, 809.2]) # observation vector

## design matrix fns
def f0(x):
	return 1


def f1(x):
	return x


def f2(x):
	return x**2


def f3(x):
	return x**3


A = np.zeros((13, 4)) # init design matrix

for i in range(13): # populate design matrix
	A[i,0] = f0(t[i])
	A[i,1] = f1(t[i])
	A[i,2] = f2(t[i])
	A[i,3] = f3(t[i])

## solve normal equations for least squares solution beta
AT_A = np.matmul(np.transpose(A), A)
AT_y = np.transpose(A).dot(y)
beta = np.linalg.solve(AT_A, AT_y) # parameter vector

## Least squares cubic fit 
def y_hat(t):
	return beta[0] + beta[1]*t + beta[2]*(t**2) + beta[3]*(t**3)

y_e = y_hat(t)

print('Least squares estimate for velocity when t = 4.5:\n', 
		(beta[1] + 2*beta[2]*(4.5) + 3*beta[3]*(4.5**2)), '\n', sep='')

fig = plt.figure()
ax = plt.axes()
plt.scatter(t, y, label='y')
plt.plot(t, y_e, c='g', lw=2, ls='-', alpha=0.3, label='y_e')
'''
plt.title('Project 1 Problem 2 - least-squares cubic fit on data')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

plt.show()
'''
W = np.diag([1, 1, 1, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

AT_W_A = np.matmul(np.transpose(A), np.matmul(W, A))
AT_W_y = np.matmul(np.transpose(A), W).dot(y)
beta_w = np.linalg.solve(AT_W_A, AT_W_y)

def y_w_hat(t):
	return beta_w[0] + beta_w[1]*t + beta_w[2]*(t**2) + beta_w[3]*(t**3)

y_w_e = y_w_hat(t)

print('Weighted least squares estimate for velocity when t = 4.5:\n', 
		(beta_w[1] + 2*beta_w[2]*(4.5) + 3*beta_w[3]*(4.5**2)), '\n', sep='')

plt.plot(t, y_w_e, c='m', lw=2, ls='--', alpha=0.35, label='y_w_e')

plt.title('Project 1 Problem 2 - weighted/ordinary least-squares cubic fit on data')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

plt.show()














