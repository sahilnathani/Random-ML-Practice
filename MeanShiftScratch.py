# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 09:20:12 2018

@author: Sahil Nathani
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=15, centers=3, n_features=2)

''' X = np.array([[1, 2], 
             [1, 3], 
             [5, 8], 
             [4, 8], 
             [3, 6], 
             [9, 1],
             [8, 2],
             [10, 2],
             [9, 3]])'''

#plt.scatter(X[:,0], X[:,1], s=100)
#plt.show()

colors = 10*["g", "b", "k", "c"]

class MeanShift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
        
    def fit(self, data):
      if self.radius==None:
          all_data_centroids = np.average(data, axis=0)
          all_data_norm = np.linalg.norm(all_data_centroids)
          self.radius = all_data_norm/self.radius_norm_step
        
      centroids = {}
      
      for i in range(len(data)):
          centroids[i] = data[i]
         
      while True:
          new_centroids = []
          for i in centroids:
              in_bandwith = []
              centroid = centroids[i]
              for featureset in data:
                  distance = np.linalg.norm(featureset-centroid)
                  if distance == 0.0:
                     distance = 0.0000000000000001
                  weight_index = int(distance/self.radius)    
                  weights = [1 for i in range(self.radius_norm_step)][::-1]
                  if weight_index>self.radius_norm_step-1:
                      weight_index = self.radius_norm_step-1
                  to_add = (weights[weight_index]**2)*[featureset]    
                  in_bandwith += to_add
                  
              new_centroid = np.average(in_bandwith, axis=0)
              new_centroids.append(tuple(new_centroid))
              
          uniques = sorted(list(set(new_centroids)))
          
          to_pop = []
          for i in uniques:
              for ii in uniques:
                  if i==ii:
                      pass
                  elif np.linalg.norm(np.array(i)-np.array(ii))<=self.radius:
                      to_pop.append(ii)
                      break
                  
          for i in to_pop:
              try:
               uniques.remove(i)
              except:
                  pass
                      
          
          prev_centroids = dict(centroids)
          
          centroids = {}
          for i in range(len(uniques)):
              centroids[i] = np.array(uniques[i])
              
          optimised = True
          
          for i in centroids:
              if not np.array_equal(centroids[i], prev_centroids[i]):
                  optimised = False
                  
              if not optimised:
                  break
              
          if optimised:
              break
          
      self.centroids = centroids
      self.classifications = {}
      
      for i in range(len(self.centroids)):
          self.classifications = []
          
      for featueset in data:
          distances  = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
          classification = distances.index(min(distances))
          self.classifications[classification].append(featureset)
       
    def predict(self, data):
        distances  = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
        
clf=MeanShift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color=colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, marker='x', s=150, linewidth=5)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s= 150)

plt.show()

              