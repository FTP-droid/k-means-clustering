import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# X will be a 2 dim array with values (-2, 0) across 100 rows and 2 columns
X = -2 * np.random.rand(100, 2)

# X1 will be a 2 dim array with values (1, 3) across 50 rows and 2 columns
X1 = 1 + 2 * np.random.rand(50, 2)

# X from rows 49 to 99 replaced by the values from X1 which have values in the space (1, 3)
X[50:100, :] = X1
