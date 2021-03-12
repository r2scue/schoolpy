# Kenny Kong MATH 425 Group 6 Project 1 Problem 2
## Christina Cai, Jiajian Li, Jocelyn Padilla, Danny Lopez
### intermediate calculations done/verified on TI-nspire

import numpy as np 
import matplotlib.pyplot as plt


data = np.genfromtxt('heightweight.txt', usecols=(1,2))
# col names of data: height, weight

# design matrix
A = np.ones(np.shape(data))
A[:,1] = data[:,0]

Q = np.zeros(np.shape(data))
R = np.zeros((2, 2))

# QR factorization for A
R[0,0] = np.linalg.norm(A[:,0]) 	# r11 = ||v1|| = ||x1||
Q[:,0] = A[:,0] / R[0,0] 			# u1 = v1 / r11
R[0,1] = Q[:,0].dot(A[:,1]) 		# r12 = <u1, x2>

Q[:,1] = A[:,1] - R[0,1] * Q[:,0] 	# v2 = x2 - r12 * u1
R[1,1] = np.linalg.norm(Q[:,1]) 	# r22 = ||v2||
Q[:,1] = Q[:,1] / R[1,1] 			# u2 = v2 / r22

beta = np.linalg.solve(R, np.transpose(Q).dot(data[:,1]))
print(beta)

#print('Q:', Q, '\nR:', R, sep='\n')

def y_hat(h):
	return beta[0] + beta[1]*h

fig = plt.figure()
ax = plt.axes()

plt.scatter(data[:,0], data[:,1], alpha=0.5, label='y')
plt.plot(data[:,0], y_hat(data[:,0]), c='m', label='y_e')

plt.title('Project 1 Problem 3 - least-squares linear fit on height/weight data')
plt.xlabel('height')
plt.ylabel('weight')
plt.legend()

plt.show()






















