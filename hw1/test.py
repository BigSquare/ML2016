import csv
import numpy as np

w1, h1 = 2160, 18
table = [[0 for x in range(w1)] for y in range(h1)]

f = open('data/test_X.csv', 'r')
x, y = 0, 0
for row in csv.reader(f):
	table[x][y:y+9] = row[2:11]
	x += 1
	if x == 18:
		x = 0
		y += 9
f.close()
#print table[0]

table = np.array(table)
table[table == 'NR'] = 0
table = table.astype(np.float)
for i in range(17,12,-1):
	table = np.delete(table,i,0)
table = np.delete(table,11,0)
table = np.delete(table,10,0)
for i in range(0,4,1):
	table = np.delete(table,0,0)

w2, h2 = 64, 240
test = [[0 for x in range(w2)] for y in range(h2)]
count_row, count_col = 0, 0
for row in table.T:
	test[count_row][count_col:count_col+7] = row
	count_col += 7
	if count_col == 63:
		test[count_row][63] = 1
		count_row += 1
		count_col = 0
test = np.array(test)
print (test[239])
np.save('test_7.npy', test)
