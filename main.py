import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# X will be a 2 dim array with values (-2, 0) across 100 rows and 2 columns
X = -2 * np.random.rand(100, 2)

# X1 will be a 2 dim array with values (1, 3) across 50 rows and 2 columns
X1 = 1 + 2 * np.random.rand(50, 2)

# X from rows 49 to 99 and both columns replaced by the values from X1 which have values in the space (1, 3)
X[50:100, :] = X1

# Set up a KMeans clustering algorithm with number of clusters = 2, init = 10 resulting in 10 different
# KMeans clustering algorithm runs. Finally fit the data in X using the defined Kmeans algorithm.
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

XClusterCenters = Kmean.cluster_centers_

# Prints the center of the clusters of X
print(XClusterCenters)

# Plots the X values with x values consisting of the first column of rows in X and y values consisting of
# the second column of rows in X. Marker size set to 50 & color of points set to blue.
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')

# The following 2 lines plot the known cluster centroids for X with the values for X
plt.scatter(XClusterCenters[0][0], XClusterCenters[0][1], s=225, c='g', marker='s')
plt.scatter(XClusterCenters[1][0], XClusterCenters[1][1], s=225, c='r', marker='s')

plt.show()

# Show which X points belong to cluster 1 and cluster 2
print(Kmean.labels_)

# ! For a new test value, determine which cluster it will belong to

# Define an array with values [-1, -1]
sampleValue = np.array([-1, -1])

# Reshape the array to make it 2-dimensional to simulate the shape of X. Array will now be [[-1, -1]].
test = sampleValue.reshape(1, -1)

# Predict the cluster where the test values (really only one value in test) belong
print("The test values belongs to clusters", Kmean.predict(test))

