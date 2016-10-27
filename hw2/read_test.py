import csv
import numpy as np
import sys

filename = sys.argv[1]
f = open(filename, 'r')

w, h = 58, 600
test = [[0 for x in range(w)] for y in range(h)]
x = 0
for row in csv.reader(f):
	test[x][0:57] = row[1:58]		
	test[x][57] = 1
	x += 1
test = np.array(test)
test = test.astype(np.float)
np.save('test.npy', test)
#print test
