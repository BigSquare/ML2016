import csv
import sys
import numpy as np

filename = sys.argv[1]

#NOTE path to train.csv
f = open(filename, 'r')

w, h = 58, 4001
table = [[0 for x in range(w)] for y in range(h)]
wa, ha = 1, 4001
answer = [[0 for x in range(wa)] for y in range(ha)]
#wt, ht = 58, 600
#test = [[0 for x in range(wt)] for y in range(ht)]
x = 0
for row in csv.reader(f):
	table[x][0:57] = row[1:58]
	table[x][57] = 1	#bias
	answer[x] = row[58]#label
	x += 1
table = np.array(table)
table = table.astype(np.float)
#print(table[0])
answer = np.array([answer])
answer = answer.astype(np.float)
np.save('table.npy', table)
np.save('ans.npy', answer.T)
f.close()



