################################################
##### Kenny Kong MATH 425 Project 2 Part B #####


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

test_res = np.zeros(1000)
stage_classified = np.zeros(1000)

stage1_ct = 0
for i in range(1000):
	z = np.zeros(10)
	z[0] = np.linalg.norm(test[:,i]-u0[:,0]*np.matmul(u0[:,0].T, test[:,i]))
	z[1] = np.linalg.norm(test[:,i]-u1[:,0]*np.matmul(u1[:,0].T, test[:,i]))
	z[2] = np.linalg.norm(test[:,i]-u2[:,0]*np.matmul(u2[:,0].T, test[:,i]))
	z[3] = np.linalg.norm(test[:,i]-u3[:,0]*np.matmul(u3[:,0].T, test[:,i]))
	z[4] = np.linalg.norm(test[:,i]-u4[:,0]*np.matmul(u4[:,0].T, test[:,i]))
	z[5] = np.linalg.norm(test[:,i]-u5[:,0]*np.matmul(u5[:,0].T, test[:,i]))
	z[6] = np.linalg.norm(test[:,i]-u6[:,0]*np.matmul(u6[:,0].T, test[:,i]))
	z[7] = np.linalg.norm(test[:,i]-u7[:,0]*np.matmul(u7[:,0].T, test[:,i]))
	z[8] = np.linalg.norm(test[:,i]-u8[:,0]*np.matmul(u8[:,0].T, test[:,i]))
	z[9] = np.linalg.norm(test[:,i]-u9[:,0]*np.matmul(u9[:,0].T, test[:,i]))
	
	zs = np.sort(z)
	if zs[0] < 3.5 and zs[1]-zs[0] > 0.3:
		test_res[i] = np.argmin(z)
		stage1_ct += 1
	else:
		stage_classified[i] = 1
		z[0] = np.linalg.norm(test[:,i]-np.matmul(u0[:,0:10], np.matmul(u0[:,0:10].T, test[:,i])))
		z[1] = np.linalg.norm(test[:,i]-np.matmul(u1[:,0:10], np.matmul(u1[:,0:10].T, test[:,i])))
		z[2] = np.linalg.norm(test[:,i]-np.matmul(u2[:,0:10], np.matmul(u2[:,0:10].T, test[:,i])))
		z[3] = np.linalg.norm(test[:,i]-np.matmul(u3[:,0:10], np.matmul(u3[:,0:10].T, test[:,i])))
		z[4] = np.linalg.norm(test[:,i]-np.matmul(u4[:,0:10], np.matmul(u4[:,0:10].T, test[:,i])))
		z[5] = np.linalg.norm(test[:,i]-np.matmul(u5[:,0:10], np.matmul(u5[:,0:10].T, test[:,i])))
		z[6] = np.linalg.norm(test[:,i]-np.matmul(u6[:,0:10], np.matmul(u6[:,0:10].T, test[:,i])))
		z[7] = np.linalg.norm(test[:,i]-np.matmul(u7[:,0:10], np.matmul(u7[:,0:10].T, test[:,i])))
		z[8] = np.linalg.norm(test[:,i]-np.matmul(u8[:,0:10], np.matmul(u8[:,0:10].T, test[:,i])))
		z[9] = np.linalg.norm(test[:,i]-np.matmul(u9[:,0:10], np.matmul(u9[:,0:10].T, test[:,i])))
		test_res[i] = np.argmin(z)

stage2_ct = 1000-stage1_ct
print('stage 1 (stage 1 min. residual significant) classifications:', stage1_ct, 
		'\nstage 2 (stage 1 min. residual not significant) classifications', stage2_ct)
print('overall success rate: ', np.sum(test_res==test_lab) / 10, '%')
print('stage 1 classifier success rate: ', 
	 100*(np.sum(test_res[stage_classified==0]==test_lab[stage_classified==0])/stage1_ct), '%')
print('stage 2 classifier success rate: ', 
	 100*(np.sum(test_res[stage_classified==1]==test_lab[stage_classified==1])/stage2_ct), '%')

''' residual < 3 use 1 basis vector: console output for print call

Kennys-MacBook-Pro:p2code rescue$ python3 p2_b.py
stage 1 (residual < 3) classifications: 89 
stage 2 (residual >= 3) classifications 911
overall success rate:  94.0 %
stage 1 classifier success rate:  95.50561797752809 %
stage 2 classifier success rate:  93.85290889132821 %

'''


