import numpy as np
import csv
import sys
from keras.models import Sequential, load_model

test = np.load('test.npy')
test = test.reshape(10000,3,32,32)
model = load_model(sys.argv[1])
prediction = model.predict(test, batch_size=100, verbose=0)
#print prediction.shape
#print prediction[0]
wf = open(sys.argv[2], 'w')
wr = csv.writer(wf)
row_data = [['ID','class']]
wr.writerows(row_data)
count = 0
for row in prediction:
	x = row
	label = np.argmax(x)
	row_data = [[str(count), str(label)]]
	wr.writerows(row_data)
	count += 1
wf.close()
