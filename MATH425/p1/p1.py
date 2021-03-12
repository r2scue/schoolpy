# Kenny Kong MATH 425 Group 6 Project 1 Problem 1
## Christina Cai, Jiajian Li, Jocelyn Padilla, Danny Lopez
### intermediate calculations done/verified on TI-nspire


import numpy as np
import matplotlib.pyplot as plt


pi = np.pi

# tiles represented by a1, a2, a3, a4
## tile corners in homogeneous coordinates
a1 = np.array([[0.3036, 0.6168, 0.7128, 0.7120, 0.9377, 0.7120, 0.3989, 
			0.3028, 0.3036, 0.5293], 
			[0.1960, 0.2977, 0.4169, 0.1960, 0.2620, 0.5680, 0.6697, 
			0.7889, 0.5680, 0.5020], 
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
a2 = a1.copy()
a3 = a2.copy()
a4 = a3.copy()

# part a.i.)
## rotation by pi radians about (0.712, 0.432)
h1 = np.array([[-1, 0, 1.424], 
				[0, -1, 0.864], 
				[0, 0, 1]])
print('Rotation by pi radians about (0.712, 0.432) matrix in homogeneous', 
	'coordinates: \n', h1, '\n', sep='')

# part a.ii.)
## composition: reflection across y=0.618, translation by (0.4084, 0)
h2 = np.array([[1, 0, 0.4084], 
				[0, -1, 1.236], 
				[0, 0, 1]])
print('Reflection across y = 0.618, translation by (0.4084, 0)', 
	' composition matrix in homogeneous coordinates: \n', h2, '\n', 
	sep='')

# part a.iii.)
## composition: reflection across x=0.5078, translation by (0, 0.1)
h3 = np.array([[-1, 0, 1.0156], 
				[0, 1, 0.1], 
				[0, 0, 1]])
print('Reflection across x = 0.5078, translation by (0, 0.1)', 
	' composition matrix in homogeneous coordinates: \n', h3, '\n', 
	sep='')

# part a.iv.)
## translation by (0, 0.472)
h4 = np.array([[1, 0, 0], 
				[0, 1, 0.472], 
				[0, 0, 1]])
print('Translation by (0, 0.472) matrix in homogeneous coordinates: \n', 
	h4, sep='')

# apply h1-h4 to each corner position vector of tiles 1-4 resp'ly
for i in range(10):
	a4[:, i] = np.matmul(h4, a4[:, i])
	a3[:, i] = np.matmul(h3, a3[:, i])
	a2[:, i] = np.matmul(h2, a2[:, i])
	a1[:, i] = np.matmul(h1, a1[:, i])

fig = plt.figure()
ax = plt.axes()

## convert to rectangular from homogeneous coordinates
ax.fill(a1[2,0]*a1[0,:], a1[2, 0]*a1[1,:]) # store pattern itr to plot
ax.fill(a2[2,0]*a2[0,:], a2[2, 0]*a2[1,:])
ax.fill(a3[2,0]*a3[0,:], a3[2, 0]*a3[1,:])
ax.fill(a4[2,0]*a4[0,:], a4[2, 0]*a4[1,:])
# base pattern

# part b.)
## translation by (0, 0.7441n), n = 1, 2, 3 using matrix mult'n
h5 = np.array([[1, 0, 0], 
				[0, 1, 0.7441], 
				[0, 0, 1]])

## apply h5 to pattern tiles for n = 1, 2, 3 and plot in rectangular coordinates
b1 = a1.copy()
b2 = a2.copy()
b3 = a3.copy()
b4 = a4.copy()

for n in range(1, 4):

	for i in range(10):
		b4[:, i] = np.matmul(h5, b4[:, i])
		b3[:, i] = np.matmul(h5, b3[:, i])
		b2[:, i] = np.matmul(h5, b2[:, i])
		b1[:, i] = np.matmul(h5, b1[:, i])

	ax.fill(b1[2,0]*b1[0,:], b1[2, 0]*b1[1,:]) # store pattern itr to plot
	ax.fill(b2[2,0]*b2[0,:], b2[2, 0]*b2[1,:])
	ax.fill(b3[2,0]*b3[0,:], b3[2, 0]*b3[1,:])
	ax.fill(b4[2,0]*b4[0,:], b4[2, 0]*b4[1,:])
# column of base pattern

# part c.)
## translation by (-0.8168n, 0), n = 1, 2, 3, 4, 5 using matrix mult'n
h6 = np.array([[1, 0, -0.8168], 
				[0, 1, 0], 
				[0, 0, 1]])

## repeat part b.) and
## apply h6 to pattern tiles for n = 1, 2, 3, 4, 5 and plot in rectangular 
## coordinates

b1 = a1.copy() # store base pattern
b2 = a2.copy()
b3 = a3.copy()
b4 = a4.copy()

for m in range(1, 6): # apply h6 (hz translation) for m = 1, 2, 3, 4, 5
	c1 = b1.copy() # pattern iterator
	c2 = b2.copy()
	c3 = b3.copy()
	c4 = b4.copy()

	for i in range(10): # does one hz translation (h6 once)
		c4[:, i] = np.matmul(h6, c4[:, i])
		c3[:, i] = np.matmul(h6, c3[:, i])
		c2[:, i] = np.matmul(h6, c2[:, i])
		c1[:, i] = np.matmul(h6, c1[:, i])

	b1 = c1.copy() # store current hz translation of base for iteration
	b2 = c2.copy()
	b3 = c3.copy()
	b4 = c4.copy()

	ax.fill(c1[2,0]*c1[0,:], c1[2, 0]*c1[1,:]) # store pattern itr to plot
	ax.fill(c2[2,0]*c2[0,:], c2[2, 0]*c2[1,:])
	ax.fill(c3[2,0]*c3[0,:], c3[2, 0]*c3[1,:])
	ax.fill(c4[2,0]*c4[0,:], c4[2, 0]*c4[1,:])

	for n in range (1, 4): # apply h5 (vt translation) for n = 1, 2, 3
		for i in range(10): # does one vt translation (h5 once)
			c4[:, i] = np.matmul(h5, c4[:, i])
			c3[:, i] = np.matmul(h5, c3[:, i])
			c2[:, i] = np.matmul(h5, c2[:, i])
			c1[:, i] = np.matmul(h5, c1[:, i])

		ax.fill(c1[2,0]*c1[0,:], c1[2, 0]*c1[1,:]) # store pattern itr to plot
		ax.fill(c2[2,0]*c2[0,:], c2[2, 0]*c2[1,:])
		ax.fill(c3[2,0]*c3[0,:], c3[2, 0]*c3[1,:])
		ax.fill(c4[2,0]*c4[0,:], c4[2, 0]*c4[1,:])
# final pattern

plt.title('Project 1 Problem 1 - final pattern')
#plt.title('Project 1 Problem 1 - column of base pattern')
#plt.title('Project 1 Problem 1 - base pattern')


plt.xlabel('x')
plt.ylabel('y')

plt.show() # plot




















