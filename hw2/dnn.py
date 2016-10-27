import numpy as np
import csv
import sys
###3-layer neural network
###activation fuction = sigmoid
###adam optimizer

def sigmoid(x):
	
	return 1.0/(1+np.exp(-x))
def d_sigmoid(x):
	#return np.multiply(sigmoid(x), 1-sigmoid(x))
	return (1.0/(1+np.exp(-x))) * (1-(1.0/(1+np.exp(-x))))

#read in training data
train_x = np.load('table.npy')#(4001, 58)
answer = np.load('ans.npy') #(4001,  1)

#declare parameters
#w_1 = np.ones((58, 40), dtype = np.float)*0.0001
#w_2 = np.ones((40, 1), dtype = np.float)*0.0001
np.random.seed(0)
w_1 = np.random.rand(58,36)
w_2 = np.random.rand(36,1)
np.save('w_1.npy', w_1)
np.save('w_2.npy', w_2)
m_1 = np.zeros((58,36), dtype = np.float)
v_1 = np.zeros((58,36), dtype = np.float)
m_2 = np.zeros((36, 1), dtype = np.float)
v_2 = np.zeros((36, 1), dtype = np.float)
e, t = 1e-8, 0
Eta, reg = 0.01, 0.05
beta1, beta2 = 0.9, 0.999

for i in range(0, 2000, 1):
	#forward propagation
	z1 = np.dot(train_x, w_1)	#(4001, 40)
	a1 = sigmoid(z1)				#(4001, 40)
	a1.T[0].fill(1)				# bias
	z2 = np.dot(a1, w_2)			#(4001,  1)
	a2 = sigmoid(z2)				#(4001,  1)
	#backward propagation
	delta_3 = d_sigmoid(z2) * (-(answer/(a2+1e-30)+(1-answer)/(a2-1+1e-30)))
	#delta_3 = -(answer - a2)	#(4001,  1)
	#delta_2 = d_sigmoid(z1) * np.dot(delta_3, w_2.T)
	delta_2 = np.multiply(d_sigmoid(z1), np.dot(delta_3, w_2.T))
										#(4001, 40)
	dw2 = np.dot(a1.T, delta_3)			#(40,  1)
	dw1 = np.dot(train_x.T, delta_2)		#(58, 40)
	#regularization
	dw1 += reg * w_1
	dw2 += reg * w_2
	###ADAM_w1	
	t += 1
	lr_t = Eta * np.sqrt(1-(beta2**t))/(1-(beta1**t))
	m_1 = beta1 * m_1 + (1-beta1) * dw1
	v_1 = beta2 * v_1 + (1-beta2) * np.square(dw1)
	w_1 = w_1 - lr_t * m_1 / (np.sqrt(v_1)+e)
	###ADAM_w2
	m_2 = beta1 * m_2 + (1-beta1) * dw2
	v_2 = beta2 * v_2 + (1-beta2) * np.square(dw2)
	w_2 = w_2 - lr_t * m_2 / (np.sqrt(v_2)+e)
	#Loss
	L = np.mean(-(answer*np.log(a2+1e-20)+(1-answer)*np.log(1-a2+1e-20)))
	if (i % 10 == 0):
		print L, i
for i in xrange(4001):
	if a2[i] > 0.5: a2[i] = 1
	else: a2[i] = 0
print 'accuracy'
print (1-np.mean(np.abs(a2 - answer)))

wf = open(sys.argv[1], 'w')
wr = csv.writer(wf)

for row in w_1:
	wr.writerows([row])
wr.writerows(w_2.T)

