import sklearn
print(sklearn.__version__)
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot

# creating datasets
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
for class_value in range(2):
    row_ix = where(y==class_value)
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
pyplot.show()


# dbscan clustering
from numpy import unique
from sklearn.cluster import DBSCAN

X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
model = DBSCAN(eps=0.30, min_samples=9)
yhat = model.fit_predict(X)
clusters = unique(yhat)
for cluster in clusters:
	row_ix = where(yhat == cluster)
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
pyplot.show()


# k-means clustering
from numpy import unique
from sklearn.cluster import KMeans

X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
model = KMeans(n_clusters=2)
model.fit(X)
yhat = model.predict(X)
clusters = unique(yhat)
for cluster in clusters:
	row_ix = where(yhat == cluster)
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
pyplot.show()


# gaussian mixture clustering
from numpy import unique
from sklearn.mixture import GaussianMixture

X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
model = GaussianMixture(n_components=2)
model.fit(X)
yhat = model.predict(X)
clusters = unique(yhat)
for cluster in clusters:
	row_ix = where(yhat == cluster)
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
pyplot.show()
