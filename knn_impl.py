import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        
