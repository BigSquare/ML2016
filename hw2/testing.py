import numpy as np
import csv
import sys
def sigmoid_arr(x):
	return 1.0/(1.0+np.exp(-x))
###test
test = np.load('test.npy')
w = np.load(sys.argv[1]+".npy")
#print f_wb
wf = open(sys.argv[2], 'w')
wr  = csv.writer(wf)
row_data = [['id', 'label']]
wr.writerows(row_data)
count = 1
L = 0
for row in test:
	x = row
	y = np.dot(x, w)
	y = sigmoid_arr(y)
	label = 0
	if (y > 0.5):
		label = 1
	row_data = [[str(count),str(label)]]	
	wr.writerows(row_data)
	count += 1
wf.close()
