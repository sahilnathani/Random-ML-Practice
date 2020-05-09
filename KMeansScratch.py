# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:06:44 2018

@author: Sahil Nathani
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('fivethirtyeight')

X = np.array([[1, 2], 
             [4, 8], 
             [5, 8], 
             [-2, 8], 
             [1, 6], 
             [9, 11]])

plt.scatter(X[:,0], X[:,1], s=100)
plt.show()

colors = 10*["g", "b", "k", "c"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
    
    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
                
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
          
            prev_centroids = dict(self.centroids)
         
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
             
            optimised = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100)>self.tol:
                    optimised = False
                    
            if optimised:
                break
                    
    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
   plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="g", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
        
unknowns = np.array([[1, 3], [8, 9], [0, 3], [5,4], [6, 4]])

for x in unknowns:
    classification = clf.predict(x)
    plt.scatter(x[0], x[1], marker=">", color='r', s=150, linewidths=5)
        

plt.show()        
        