import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from future.utils import iteritems
from sortedcontainers import SortedList


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        # lazy fit function
        self.X = X
        self.y = y
        
    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # test points
            sl = SortedList() # stores (distance, class) tuples
            for j,xt in enumerate(self.X): # training points
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    # don't need to check, just add
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) )
        
