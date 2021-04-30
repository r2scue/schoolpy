################################################
##### Kenny Kong MATH 425 Project 2 Part A #####


import numpy as np 
import matplotlib.pyplot as plt 



## data step 

train = np.genfromtxt('handwriting_training_set.txt', delimiter=',').T
train_lab = np.genfromtxt('handwriting_training_set_labels.txt')
train_lab[train_lab==10] = 0
#unique, counts = np.unique(train_lab, return_counts = True)
#print(np.asarray((unique, counts)).T)

test = np.genfromtxt('handwriting_test_set.txt', delimiter=',').T
test_lab = np.genfromtxt('handwriting_test_set_labels.txt')
test_lab[test_lab==10] = 0


#print(train.shape, train_lab.shape, sep='\n')
#print(test.shape, test_lab.shape, sep='\n')

a0 = train[:,0:400]
a1 = train[:,400:800]
a2 = train[:,800:1200]
a3 = train[:,1200:1600]
a4 = train[:,1600:2000]
a5 = train[:,2000:2400]
a6 = train[:,2400:2800]
a7 = train[:,2800:3200]
a8 = train[:,3200:3600]
a9 = train[:,3600:4000]



## classifier training and testing

a0_bar = a0 - np.tile((1 / 400) * a0.sum(axis=1), (400, 1)).T
a1_bar = a1 - np.tile((1 / 400) * a1.sum(axis=1), (400, 1)).T
a2_bar = a2 - np.tile((1 / 400) * a2.sum(axis=1), (400, 1)).T
a3_bar = a3 - np.tile((1 / 400) * a3.sum(axis=1), (400, 1)).T
a4_bar = a4 - np.tile((1 / 400) * a4.sum(axis=1), (400, 1)).T
a5_bar = a5 - np.tile((1 / 400) * a5.sum(axis=1), (400, 1)).T
a6_bar = a6 - np.tile((1 / 400) * a6.sum(axis=1), (400, 1)).T
a7_bar = a7 - np.tile((1 / 400) * a7.sum(axis=1), (400, 1)).T
a8_bar = a8 - np.tile((1 / 400) * a8.sum(axis=1), (400, 1)).T
a9_bar = a9 - np.tile((1 / 400) * a9.sum(axis=1), (400, 1)).T

u0, s0, vt0 = np.linalg.svd(a0)
u1, s1, vt1 = np.linalg.svd(a1)
u2, s2, vt2 = np.linalg.svd(a2)
u3, s3, vt3 = np.linalg.svd(a3)
u4, s4, vt4 = np.linalg.svd(a4)
u5, s5, vt5 = np.linalg.svd(a5)
u6, s6, vt6 = np.linalg.svd(a6)
u7, s7, vt7 = np.linalg.svd(a7)
u8, s8, vt8 = np.linalg.svd(a8)
u9, s9, vt9 = np.linalg.svd(a9)

### testing

test_res = np.zeros(4000)

for i in range(1000):
	for j in range(4):
		z = np.zeros(10)
		z[0] = np.linalg.norm(test[:,i]-np.matmul(u0[:,0:(j+1)*5], np.matmul(u0[:,0:(j+1)*5].T, test[:,i])))
		z[1] = np.linalg.norm(test[:,i]-np.matmul(u1[:,0:(j+1)*5], np.matmul(u1[:,0:(j+1)*5].T, test[:,i])))
		z[2] = np.linalg.norm(test[:,i]-np.matmul(u2[:,0:(j+1)*5], np.matmul(u2[:,0:(j+1)*5].T, test[:,i])))
		z[3] = np.linalg.norm(test[:,i]-np.matmul(u3[:,0:(j+1)*5], np.matmul(u3[:,0:(j+1)*5].T, test[:,i])))
		z[4] = np.linalg.norm(test[:,i]-np.matmul(u4[:,0:(j+1)*5], np.matmul(u4[:,0:(j+1)*5].T, test[:,i])))
		z[5] = np.linalg.norm(test[:,i]-np.matmul(u5[:,0:(j+1)*5], np.matmul(u5[:,0:(j+1)*5].T, test[:,i])))
		z[6] = np.linalg.norm(test[:,i]-np.matmul(u6[:,0:(j+1)*5], np.matmul(u6[:,0:(j+1)*5].T, test[:,i])))
		z[7] = np.linalg.norm(test[:,i]-np.matmul(u7[:,0:(j+1)*5], np.matmul(u7[:,0:(j+1)*5].T, test[:,i])))
		z[8] = np.linalg.norm(test[:,i]-np.matmul(u8[:,0:(j+1)*5], np.matmul(u8[:,0:(j+1)*5].T, test[:,i])))
		z[9] = np.linalg.norm(test[:,i]-np.matmul(u9[:,0:(j+1)*5], np.matmul(u9[:,0:(j+1)*5].T, test[:,i])))
		test_res[i+(j*1000)] = np.argmin(z)

test_5_res = test_res[0:1000]
test_10_res = test_res[1000:2000]
test_15_res = test_res[2000:3000]
test_20_res = test_res[3000:4000]



## classifier evaluation

### graphs and singular values
'''
pct = np.array([np.sum(test_5_res == test_lab), 
			 np.sum(test_10_res == test_lab), 
			 np.sum(test_15_res == test_lab), 
			 np.sum(test_20_res == test_lab)]) / 10

fig = plt.figure()
ax = plt.axes()
ax.plot(np.array([5, 10, 15, 20]), pct, c='b')
ax.scatter(np.array([5, 10, 15, 20]), pct, c='k')
ax.set_xlabel('number of basis vectors')
ax.set_ylabel('percentage of correctly classified digits')
ax.set_title('A.i.')
plt.show()

dsr5 = np.zeros(10) # digit success rates: 5 basis vectors
dsr10 = np.zeros(10)
dsr15 = np.zeros(10)
dsr20 = np.zeros(10)
for i in range(10):
	dsr5[i] = np.sum(np.logical_and(test_5_res==i, test_lab==i))/np.sum(test_lab==i)
	dsr10[i] = np.sum(np.logical_and(test_10_res==i, test_lab==i))/np.sum(test_lab==i)
	dsr15[i] = np.sum(np.logical_and(test_15_res==i, test_lab==i))/np.sum(test_lab==i)
	dsr20[i] = np.sum(np.logical_and(test_20_res==i, test_lab==i))/np.sum(test_lab==i)
fig = plt.figure()
ax = plt.axes(xlim=(0, 9))
ax.plot(np.arange(10), dsr5, c='b')
ax.plot(np.arange(10), dsr10, c='g')
ax.plot(np.arange(10), dsr15, c='r')
ax.plot(np.arange(10), dsr20, c='m')
ax.set_xlabel('digits')
ax.set_ylabel('percent correct by number of basis vectors used')
ax.set_title('A.ii.')
plt.show()
'''
#print(s0[0:20], s1[0:20], s2[0:20], s3[0:20], s4[0:20], s5[0:20], s6[0:20], 
#	  s7[0:20], s8[0:20], s9[0:20], sep='\n\n')
''' console output for print call

Kennys-MacBook-Pro:p2code rescue$ python3 p2_a.py
[124.01037317  42.78066987  33.21935988  24.83475696  20.66246868
  19.30227614  17.00823575  16.30237637  15.74896455  14.24987001
  13.15704621  12.66033224  12.43196206  11.22746093  10.66083544
  10.21221811   9.47684616   8.98091363   8.77328406   8.593473  ]

[79.87287296 39.49222833 22.86861214 14.18854868 13.07773609 11.52661168
 10.62709009  9.14929202  8.81758382  7.45478559  7.13932553  6.67726886
  6.25381584  5.92581386  5.73187047  5.36376952  5.01701487  4.97839731
  4.77690351  4.71424346]

[103.65635378  37.32498327  27.67683039  22.76166927  21.64286476
  20.02384168  18.89184471  18.02356756  15.97695568  15.00364437
  14.23906172  14.04114404  12.82955937  12.38541874  11.95593221
  11.51030707  11.44766584  11.08660785  10.52171607  10.14728933]

[104.73737483  33.68163624  28.74574163  22.41244916  20.94709396
  17.86019157  16.65702759  15.54722681  14.61384862  14.51717887
  13.62135056  11.84116654  11.67886416  11.47219681  10.92289187
  10.48700295  10.06542662   9.50283611   9.21973868   9.117898  ]

[91.94492275 31.27711424 27.86363978 22.22709177 19.92553988 18.79267552
 17.8917106  15.46484725 14.98931334 14.34815841 13.35440106 12.73842106
 10.79605689 10.48324615 10.24556657 10.08243301  9.29643273  9.23721159
  8.88871969  8.47414952]

[90.78030444 39.62913717 27.43469727 25.54099619 19.44886994 17.9835211
 16.31817968 15.5578312  15.1228879  13.99160196 13.21532996 13.00694331
 11.9865957  11.08207485 10.80914185 10.31881528 10.09885613 10.02973272
  9.53700463  9.4828665 ]

[103.23469962  36.32520809  25.1985839   24.66272765  20.13625539
  17.89852831  16.458192    14.33655896  13.70565303  13.09220353
  12.16242603  11.75615759  11.44306349  11.09054956  10.84026909
   9.68963826   9.27576242   8.99415018   8.5877757    8.17844918]

[92.95691955 35.85075165 29.61942447 21.99814371 19.4814692  17.20792318
 14.85317221 13.67901166 12.85903307 12.3104336  11.61210642 10.82153936
 10.52438899  9.92649386  9.32625578  9.13833124  8.67362273  8.35490546
  7.90198457  7.38033646]

[107.15264904  34.76939194  23.81307079  22.24416866  18.74339007
  17.20574971  16.98740943  14.92733695  14.02843787  13.85377325
  13.17892505  12.7468115   12.196597    11.79720737  11.47151599
  11.29292143  10.31008023  10.06911638   9.84069058   9.10848717]

[94.44057418 35.58559554 24.97694473 22.21742321 18.63494268 16.83438684
 16.31458867 15.33457831 13.79840206 12.64828038 12.4007117  11.89784147
 11.42546367 10.99320378 10.62978775  9.65706216  8.98259456  8.67178935
  8.53584568  7.97563675]

'''


















