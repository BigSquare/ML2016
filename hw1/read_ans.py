import csv
import numpy as np

#create a 240*1 array
w, h = 1, 240
table = [[0 for x in range(w)] for y in range(h)]

f = open('answer_test.csv', 'r')
x = 0
for row in csv.reader(f):
	table[x] = row
	x += 1
table = np.array(table)
table = table.astype(np.float)
print table
np.save('ans.npy', table)
