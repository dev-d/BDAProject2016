
# Script to demostrate how to run co-clustering using spark and scala.
# Assumption: we assume that sample data is loaded in a Hive table called datase.
# Use the loader script to load this data if you have not already done so 
# To execute use, spark-shell

val df_newsgroup = sqlContext.sql("select * from dataset")

val splitData = df_newsgroup.randomSplit(Array(0.8, 0.2))
val df_test = splitData(0)
val df_train = splitData(1)

val tokenizer = new org.apache.spark.ml.feature.Tokenizer
tokenizer.setInputCol("news")
tokenizer.setOutputCol("news_words")

val df_train_words = tokenizer.transform(df_train)

# Hashing Term-Frequency
val numFeatures=10000
val hashing_tf = new org.apache.spark.mllib.feature.HashingTF(numFeatures)
# taking the news_words column, converting the array to a sequence and then creating a new RDD with the hasing TF
val rdd_train_tf = hashing_tf.transform(df_train_words.select("news_words").map(_.toSeq))

# Inverse Document Frequency
rdd_train_tf.cache()
# fit to build the model on all the data, and then apply it line by line
val idf = new org.apache.spark.mllib.feature.IDF().fit(rdd_train_tf) 
val rdd_train_tfidf = idf.transform(rdd_train_tf)


val indexer = new org.apache.spark.ml.feature.StringIndexer().setInputCol("target").setOutputCol("target_indexed")
# FIXME this won't work, needs that the RDD be converted back to dataFrame. So not sure if this was the intention
# see http://spark.apache.org/docs/latest/sql-programming-guide.html#interoperating-with-rdds
indexer.fit(rdd_train_tfidf)
val indexed = indexer.transform(rdd_train_tfidf)

########Scala upto this point


# K-means Algorithm
from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt

val a=dataFrame.select("news_tfidf").map(lambda r : r[0]))

clusters = KMeans.train(a, 100, maxIterations=10, runs=10, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors

from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.linalg import SparseVector
import numpy

def error(point):
    center = clusters.centers[clusters.predict(point)]
    denseCenter = DenseVector(numpy.ndarray.tolist(center))
    return sqrt(sum([x**2 for x in (DenseVector(point.toArray()) - denseCenter)]))


WSSSE = a.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

########Python upto this point

#Bi-Clustering
from collections import defaultdict
import operator
import re
from time import time
import numpy as np
from scipy.sparse import dia_matrix
from sklearn.externals.six import iteritems
from scipy.sparse import issparse
from scipy.linalg import eigh as largest_eigh
from sklearn.utils.arpack import eigsh, svds
from sklearn.utils.validation import assert_all_finite, check_array


def _svd(array, n_components, n_discard):
	u, _, vt = svds(array, k=n_components, ncv=None)
	if np.any(np.isnan(vt)):
		# some eigenvalues of A * A.T are negative, causing
		# sqrt() to be np.nan. This causes some vectors in vt
		# to be np.nan.
		_, v = eigsh(safe_sparse_dot(array.T, array),
				ncv=None)
		vt = v.T
	if np.any(np.isnan(u)):
		_, u = eigsh(safe_sparse_dot(array, array.T),
				ncv=None)
	assert_all_finite(u)
	assert_all_finite(vt)
	u = u[:, n_discard:]
	vt = vt[n_discard:]
	return u, vt.T


def scale_normalize(X):
	row_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=1))).squeeze()
	col_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=0))).squeeze()
	row_diag = np.where(np.isnan(row_diag), 0, row_diag)
	col_diag = np.where(np.isnan(col_diag), 0, col_diag)
	if issparse(X):
		n_rows, n_cols = X.shape
		r = dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
		c = dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
		an = r * X * c
	else:
		an = row_diag[:, np.newaxis] * X * col_diag
	return an, row_diag, col_diag


def SpectralCoclustering(X,n_clusters):
	normalized_data, row_diag, col_diag = scale_normalize(X)
	n_sv = 1 + int(np.ceil(np.log2(n_clusters)))
	u, v = _svd(normalized_data, n_sv, n_discard=1)
	z = np.vstack((row_diag[:, np.newaxis] * u,col_diag[:, np.newaxis] * v))
	model = KMeans.train(z, 2, maxIterations=10,runs=10, initializationMode="random")
	labels = model.labels_
	n_rows = X.shape[0]
	row_labels_ = labels[:n_rows]
	column_labels_ = labels[n_rows:]
	rows = np.vstack(row_labels_ == c for c in range(n_clusters))
	columns = np.vstack(column_labels_ == c for c in range(n_clusters))
	return -centroid, n_rows,row_labels_,column_labels_,row_labels_,rows,columns


cocluster= SpectralCoclustering(X,n_clusters=len(categories),svd_method='arpack', n_svd_vecs=None,mini_batch=False,init='k-means++',n_init=10,n_jobs=1,random_state=0)

feature_names = vectorizer.get_feature_names()
document_names = list(newsgroups.target_names[i] for i in newsgroups.target)

def bicluster_ncut(i):
	rows= np.nonzero(cocluster[5][i])[0]
	cols= np.nonzero(cocluster[6][i])[0]
	if not (np.any(rows) and np.any(cols)):
		import sys
		return sys.float_info.max
	row_complement = np.nonzero(np.logical_not(cocluster[5][i]))[0]
	col_complement = np.nonzero(np.logical_not(cocluster[6][i]))[0]
	weight = X[rows][:, cols].sum()
	cut = (X[row_complement][:, cols].sum() +X[rows][:, col_complement].sum())
	return cut / weight

def most_common(d):
    return sorted(iteritems(d), key=operator.itemgetter(1), reverse=True)

bicluster_ncuts = list(bicluster_ncut(i) for i in range(len(newsgroups.target_names)))
best_idx = np.argsort(bicluster_ncuts)[:5]

print("Best biclusters:")

for idx, cluster in enumerate(best_idx):
	n_rows = X.shape[0]
	n_cols = X.shape[1]
	cluster_docs, cluster_words = np.nonzero(cocluster[5][cluster])[0],np.nonzero(cocluster[6][cluster])[0]
	if not len(cluster_docs) or not len(cluster_words):
		continue
	counter = defaultdict(int)
	for i in cluster_docs:
		counter[document_names[i]] += 1
	cat_string = ", ".join("{:.0f}% {}".format(float(c) / n_rows * 100, name)for name, c in most_common(counter)[:3])
	out_of_cluster_docs = cocluster[4] != cluster
	out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
	word_col = X[:, cluster_words]
	word_scores = np.array(word_col[cluster_docs, :].sum(axis=0)-word_col[out_of_cluster_docs, :].sum(axis=0))
	word_scores = word_scores.ravel()
	important_words = list(feature_names[cluster_words[i]]for i in word_scores.argsort()[:-11:-1])
	print("bicluster {} : {} documents, {} words".format(idx, n_rows, n_cols))
	print("categories   : {}".format(cat_string))
	print("words        : {}\n".format(', '.join(important_words)))
