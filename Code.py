#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:47:16 2022

@author: mohamaddalati
"""

### Assignment 5 - Agglomerative and K-Means Clustering 

# Import data 
import pandas as pd 
import numpy as np 
df = pd.read_csv('/Users/mohamaddalati/Desktop/INSY-662/Assignment5/cereals.csv')

"""
Task 0: We are going to develop clustering algorithms using the following variables: 'Calories', 'Protein', 'Fat', 'Fiber', 
'Carbo', 'Sodium', 'Sugars', 'Potass', and 'Vitamins'. Drop any observations that have one or more missing value for these variables.
Note: Clustering algorithms typically cannot handle observations with any missing value.
"""
X = df[ ['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins'] ]
X.dropna(axis=0, inplace = True)
# STANDARDIZE 
from sklearn.preprocessing import StandardScaler #ALWAYS STANDARDIZE 
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

"""
Task 1: First, perform agglomerative Clustering with complete-linkage. Report the number of cereals in each cluster when
 the number of cluster is 2.
"""
# Using SKLEARN package  (notice how u have to define the number of clustering in advance unlike Scipy)
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2, affinity='euclidean', linkage = 'complete')
cluster.fit_predict(X_std)
print(cluster.labels_)
# how each datapoint or observation is assigned to each data cluster 

""" 
Task 2: Then, use the same set of variables above to perform K-Mean Clustering with k=2. Report the number of cereals
 in each cluster.
"""
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters = 2)
model = kmeans.fit(X_std)
labels = model.predict(X_std)
np.bincount(labels) # to calculate the number of occurence of numpy array 

"""
Task 3: With the results from K-Mean Clustering, intuitively explain the characteristics of cereals in each cluster.
"""
kmeans.cluster_centers_

# Cereals in the first cluster are: 
# Note: negative means low, positive means high
Containing low Calories, low Protein, low Fat, low Fiber, high Carbo, low Sodium, low Sugars, low Potassium, and high Vitamins.
Note: High since >0, and Low since <0 


# Cereals in the second cluster are:
The opposite. They are Containing high Calories, high Protein, high Fat, high Fiber, low Carbo, high Sodium, high Sugars, high Potassium, and low Vitamins.























