import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


A = np.array([[3,3], [2,2], [1 , 1]])
import ipdb
ipdb.set_trace()
_, pair_dist_arg = pairwise_distances_argmin_min(A, A, metric='euclidean')
