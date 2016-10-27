import numpy as np
import csv
import sys
def sigmoid_arr(x):
	#np.seterr(all='warn')
	return 1.0/(1.0+np.exp(-x))
###test
test = np.load('test.npy')
#w = np.load(sys.argv[1]+".npy")
f = open(sys.argv[1], 'r')
w_1 = np.zeros((58, 36), dtype = np.float)
w_2 = np.zeros((36, 1), dtype = np.float)
count = 0
for rows in csv.reader(f):
	if count == 58:
		w_2.T[0] = rows
	else:
		w_1[count] = rows
	count += 1
#print f_wb
wf = open(sys.argv[2], 'w')
wr  = csv.writer(wf)
row_data = [['id', 'label']]
wr.writerows(row_data)
count = 1
L = 0
for row in test:
	x = row
	z1 = np.dot(x, w_1)
	a1 = sigmoid_arr(z1)
	z2 = np.dot(a1, w_2)
	y  = sigmoid_arr(z2)
	label = 0
	if (y > 0.5):
		label = 1
	row_data = [[str(count),str(label)]]	
	wr.writerows(row_data)
	count += 1
wf.close()
