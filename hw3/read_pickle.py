import pickle
import sys
import numpy as np

_dir = sys.argv[1]
all_label = pickle.load(open(_dir+'all_label.p', 'rb'))
all_label = np.array(all_label)
np.save('label.npy', all_label)

test_data = pickle.load(open(_dir+'test.p', 'rb'))
test_data = np.array(test_data['data'])
np.save('test.npy', test_data)

un_label = pickle.load(open(_dir+'all_unlabel.p', 'rb'))
un_label = np.array(un_label)
np.save('unlabel.npy', un_label)
