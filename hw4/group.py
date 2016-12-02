import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import csv
import sys

###reading in the titles
with open(sys.argv[1] + 'title_StackOverflow.txt') as f:
	content = f.readlines()
for i in xrange(len(content)):
	content[i] = content[i]

vectorizer = TfidfVectorizer(max_df = 0.5, max_features=None,
									  min_df = 2, stop_words='english')

X = vectorizer.fit_transform(content)
svd = TruncatedSVD(n_components = 20, n_iter = 10, random_state = None)
normalizer = Normalizer(copy = False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

km = KMeans(n_clusters=100, init='k-means++', max_iter=200, n_init=20, verbose = 0)
km.fit(X)
clustered = km.predict(X)

###predict on testing data
wr = open(sys.argv[2], 'w')
f  = open(sys.argv[1] + 'check_index.csv', 'r')
w  = csv.writer(wr)

row_data = [['ID', 'Ans']]
w.writerows(row_data)
count = -1

for row in csv.reader(f):
	if (count != -1):
		if ( clustered[int(row[1])] == clustered[int(row[2])]):
			row_data = [[str(count), str(1)]]
		else:
			row_data = [[str(count), str(0)]]
		w.writerows(row_data)
	if(count % 100000 == 0):
		print count
	count += 1

wr.close()
f.close()
