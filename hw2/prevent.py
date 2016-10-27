import numpy as np
import csv
import math
import sys
#define functions
def sigmoid_arr(x):
	return 1.0 / (1.0 + np.exp(-1*x))
#read in training data
train_X = np.load('table.npy')
answer = np.load('ans.npy')
one = np.ones((4001,1), dtype = np.float)
w = np.ones((58, 1), dtype = np.float) * 0.0001	#w should be initialized small
#start training
#using adam optimizer
#define paremeters
Eta = 0.001
beta1, beta2 = 0.9, 0.999
epsilon = 1.0e-08
m_0 = np.zeros((58,1), dtype = np.float)
v_0 = m_0
t = 0
print(train_X.shape,w.shape)
for i in range(0, 10000, 1):
	Xw = np.dot(train_X, w)
	f_wb = sigmoid_arr(Xw)
	gradient = -np.dot(train_X.T, answer - f_wb)/4001.0
	###ADAM
	t += 1
	lr_t = Eta * math.sqrt(1-(beta2**t)) / (1-(beta1**t))
	m_0  = beta1 * m_0 + (1-beta1) * gradient
	v_0  = beta2 * v_0 + (1-beta2) * (np.square(gradient))
	w = w - lr_t * m_0 / (np.sqrt(v_0)+epsilon)
	###ADAM
	#w = w - Eta * gradient
	#L = -1*(np.dot(answer.T,np.log(f_wb))+np.dot((one-answer).T,np.log(one-f_wb)))
	L = np.mean(-(answer*np.log(f_wb+1e-20)+(1-answer)*np.log(1-f_wb+1e-20)))
	if(i % 10 == 0):
		print L, t
for i in xrange(4001):
	if f_wb[i] >= 0.5: f_wb[i] = 1
	else: f_wb[i] = 0
print 'accuracy'
print (1-np.mean(np.abs(f_wb - answer)))
#np.save(sys.argv[1], w)
