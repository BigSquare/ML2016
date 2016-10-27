import csv
import time
import numpy as np
import matplotlib.pyplot as plt

#for table 1
w1, h1 = 5760, 18
table = [[0 for x in range(w1)] for y in range(h1)]

f = open('data/train.csv', 'r')
x, y = 0, 0

for row in csv.reader(f):
	if x != 0:
		table[x-1][y:y+24] = row[3:27]##[0:24] = 0~23
	x += 1
	if x == 19:
		x = 1
		y += 24
f.close()
table = np.array(table)
table[table == 'NR'] = 0
table = table.astype(np.float)
#TODO delete redundant data
for i in range(17,12,-1):
	table = np.delete(table, i, 0)
table = np.delete(table, 11, 0)
table = np.delete(table, 10, 0)
for i in range(0,4,1):
	table = np.delete(table, 0, 0)

#TODO deal with incontinuous months
#seperate each month
w3, h3 = 64, 5652
count_m = 0
skip_row = 0
train = [[0 for x in range(w3)] for y in range(h3)]
for rows in range(0, 5652, 1):
	for cols in range(0, 9, 1):
		temp = cols * 7
		train[rows][temp:temp+7] = table.T[rows+cols+skip_row*9][0:7]
		if cols == 8:
			train[rows][63] = 1
			if count_m == 470:
				count_m = -1
				skip_row += 1
	count_m += 1
train = np.array(train)
# read in complete
# start training
#y_ans = table[9][9:5760]
y_ans = [0 for x in range(5652)]
for i in range(0, 12, 1):
	y_ans[471*i:471*(i+1)] = table[5][9+i*480:480*(i+1)]
y_ans = np.array([y_ans])		  
#NOTE it is originally a 1-d array, need brackets
w_wgt = np.ones((64, 1), dtype = np.float)*0.01 

# define learning rate
Eta = [0.0000015, 0.000001, 0.0000001, 0.00000001]
lamda = [0.1, 10, 100, 1000]
#data for plotting
data_X = np.arange(300000)
data_Y = [0 for y in range(300000)]
Xw_store = np.dot(train, w_wgt)
ada_1 = [0 for x in range(64)]
ada_2 = np.zeros((64,1), dtype = np.float)

tStart = time.time()
#plt.figure(1)
for comp in range(0, 1, 1):
	w_wgt = np.ones((64, 1), dtype = np.float)*0.01
	for i in range(0, 300000, 1):
		Xw = np.dot(train, w_wgt)
		gradient = np.dot(np.transpose(train), Xw - np.transpose(y_ans))
		gradient_n = np.dot(2*Eta[0]/5652, gradient)
		#gradient_n = gradient_n + np.dot( w_wgt, 2*lamda[comp]*Eta[0])
		#ada_1 = np.square(gradient_n)
		#ada_2 = ada_1 + ada_2
		#temp_eta = np.sqrt(ada_2)
		w_wgt = w_wgt - gradient_n #comparing lamda and Eta, remember to change all parameters
		#data_Y[i] = np.sqrt(np.square(np.linalg.norm(y_ans.T-Xw))/5652)
		#print data_Y[i]
	#plt.plot(data_X, data_Y, label = 'lamda '+str(lamda[comp]))
	#plt.ylim([4,10])
#plt.xlabel('iteration')
#plt.ylabel('loss')
#plt.legend()
#plt.show()
tEnd = time.time()

print ('time: ', tEnd - tStart)
# get the correct answer
x_plus = np.linalg.pinv(train)
w = np.dot(x_plus, y_ans.T)

# compare train case with sudo inverse
i = 0
avg = 0
for row in train:
	x = row
	y = table[5][9+i]
	a1 = y - np.dot(x, w)
	a2 = y - np.dot(x, w_wgt)
	avg += abs(a1-a2)
	i += 1
print ('compare with sudo inverse:')
print avg/5652

#compare with test case answer
# read in test_X.csv
test = np.load('test_7.npy')


wf = open('kaggle_best.csv', 'w')
w = csv.writer(wf)
row_data = [['id','value']]
w.writerows(row_data)
count = 0
avg = 0
for row in test:
	x = row
	y = np.dot(x, w_wgt)
	#ans_y = table_ans[count]
	#temp = ans_y - y
	#avg += np.dot(temp, temp)
	row_data = [['id_'+str(count),str(y[0])]]
	w.writerows(row_data)
	count += 1
wf.close()
