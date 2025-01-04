import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=pd.read_csv('./MlLibrary/unsupervised_data.csv').to_numpy()

X= (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    
    return centroids


def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for j in range(X.shape[0]):
        min_distance = float('inf')  # Initialize to a very large number
        for i in range(K):
            dist = np.sum((X[j] - centroids[i]) ** 2)
            if dist < min_distance:
                min_distance = dist
                idx[j] = i  
                    
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for i in range(K):
        sum=0
        c=0
        for j in range(m):
            if(idx[j]==i):
                c=c+1
                sum+=X[j]
        centroids[i]=sum/c
        
    return centroids
  def cost(X,centroids,idx):
    cost=0
    for i in range(X.shape[0]):
         cluster_idx = idx[i]
         cost += np.sum((X[i] - centroids[cluster_idx]) ** 2)
    return cost

def run_kMeans(X, initial_centroids, max_iters=10):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
      
    idx = np.zeros(m)
    

    for i in range(max_iters):
        
   
        print("K-Means iteration %d/%d" % (i+1, max_iters))
        
        
        idx = find_closest_centroids(X, centroids)
        
        centroids = compute_centroids(X, idx, K)
    
    return centroids, idx


def elbowpoint(X,K):
      costs = []
      for K in range(1, K + 1):
        init_centroids = kMeans_init_centroids(X, K)
        final_centroids, cluster_idx = run_kMeans(X, init_centroids, max_iters=10)
        totcost = cost(X, final_centroids, cluster_idx)
        costs.append(totcost)
      

      plt.figure(figsize=(8, 6))
      plt.plot(range(1, K + 1), costs, marker='o', linestyle='-')
      plt.title('Elbow Method for Optimal K')
      plt.xlabel('Number of Clusters (K)')
      plt.ylabel('cost')
      plt.xticks(range(1, K + 1))
      plt.grid(True)
      plt.show()



K= 20
inertias = elbowpoint(X, K)



init_centroids = kMeans_init_centroids(X, 4)
final_centroids, cluster_idx = run_kMeans(X, init_centroids, max_iters=10)

print("Final centroids:\n", final_centroids)
