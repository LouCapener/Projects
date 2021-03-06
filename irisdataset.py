# -*- coding: utf-8 -*-
"""IrisDataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hYwTtUeytq3l_toFLzKnxuqxcbjVhAoU
"""

# Louise Capener
# 200956103
# COMP534 Assignment 1


# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

import numpy as np
# numpy is used to perform the necessary mathematical operations

# The following code is taken from the ML book referenced above:

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        # random_state - this will be the seed for RandomState

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        # rgen - numpys random function is used to create a random number generator
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # These random numbers are drawn from a normal distribution
        self.errors_ = []
        epoch = 0

        for _ in range(self.n_iter):
            errors = 0
            epoch += 1
            # I created an epoch counter to keep track of epoch iterations
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                # If target is 1 and prediction is -1, then difference is 2
                # If target is -1 and prediction is -1 then there's no difference
                # I.e. dont need to update anything as we only update if we misclassify
                self.w_[1:] += update * xi
                # For weights we use update value multiplied by the features of each instance
                self.w_[0] += update
                # for bias we add the update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print("Error for epoch", epoch, "=",errors)
        return errors

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# The following code is not taken from the ML book, and represents my own work:
# ### Reading-in the Iris data

import pandas as pd
# pandas is used to read-in the Iris data

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
        # The Iris dataset is the most popular dataset taken from the UCI Machine Learning Repository
        # csv - this is filename extension for 'comma separated value' files
        # read - this pandas function reads the csv file into a DataFrame (i.e. 2D labeled data structure)
        # header=None - we already explicit column names, so we do not need to number them

# I create my y variables which will contain the specific data values I'd like to select
# These y variables will be the target values that are entered into the fit function
# Essentially, y is selecting the data relating to setosa and versicolor
y = df.iloc[0:100, 4].values
# iloc - a function that is part of the DataFrame class and uses indexes to locate data
# i.e. iloc retrieves the data for setosa and versicolor
y = np.where(y == 'Iris-setosa', -1, 1)
# np.where - a numpy function that returns elements depending on the condition
# i.e. in this case it returns -1 for data relating to setosa, and 1 for versicolor


# ### Training the perceptron model
ppn = Perceptron(eta=0.1, n_iter=4)

features_list = ([[0,1,2], [1,2,3], [0,2,3], [0,1,3]])
# Was previously np.array


for i in features_list:
  print()
  print("The error for combination", i, "is:")
  E = df.iloc[0:100,i].values
  error = ppn.fit(E,y)

# The feature that should be left out is feature 1, which is indicated by the fact that [2,3,4] had the lowest error
# Whereas, combinations that included feature 1 had more errors
# Therefore, every feature set apart from set [2,3,4] should be removed