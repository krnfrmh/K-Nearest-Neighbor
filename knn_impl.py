import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from future.utils import iteritems
from sortedcontainers import SortedList
from generate_data import get_data

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
            
            # vote
            votes = {}
            for _, v in sl:
                # print "v:", v
                votes[v] = votes.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in iteritems(votes):
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y
    
    def predict_vectorize(self, X):
        N = len(X)
        y = np.zeros(N)

        # returns distances in a matrix of shape (N_test, N_train)
        distances = pairwise_distances(X, self.X)
        
        # get the minimum k elements' indexes
        idx = distances.argsort(axis=1)[:, :self.k]

        # determine the winning votes each row of idx contains indexes from 0..Ntrain corresponding to the indexes of the closest samples from the training set
        votes = self.y[idx]
        for i in range(N):
            y[i] = np.bincount(votes[i]).argmax()

        return y
        
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data()

    # display the data
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # get the accuracy
    model = KNN(3)
    model.fit(X, Y)
    print("Accuracy:", model.score(X, Y))
