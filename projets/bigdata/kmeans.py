import random

import numpy as np 


# 7(a)
def compute_centroids(points, cluster_ids):
    '''
    Compute centroids of clusters

    points : value of points (dimension 1), pyspark.rdd.RDD
    clusters_ids : ids of clusters associated with points, pyspark.rdd.RDD
    '''
    # create tuples (id, value)
    temp = cluster_ids.zip(points)
    # sum values for each key
    sum_by_cluster_id = temp.reduceByKey(lambda x, y: x + y)
    # .mapValues(lambda x: 1) used to count each elements by keys
    count_by_cluster_id = temp.mapValues(lambda x: 1).reduceByKey(lambda x, y: x+y)
    # compute means by key
    centroids = sum_by_cluster_id.join(count_by_cluster_id).mapValues(lambda x: x[0] / x[1])
    return   centroids


def squared_distances(value, means):
    # return [(value - mean) ** 2 for mean in means]  #  7(b)i
    return [np.sum((value - mean) ** 2) for mean in means]  # 8. 


def assign_clusters(points, centroids):
    '''
    Assign each points to a cluster, return a pyspark.rdd.RDD

    points : pyspark.rdd.RDD
    centroids : result of compute_centroids, pyspark.rdd.RDD
    '''
    # 7(b)ii
    means = centroids.values().collect()
    # 7(b)iii
    # search index of closest centroid of each point
    temp = points.map(lambda x: np.argmin(squared_distances(x, means))) 
    # recover index of cluster's centroid
    centroids_keys = centroids.keys().collect() # impossible to use RDD object
    # in .map, we keep the list for next line
    assigned_clusters = temp.map(lambda x: centroids_keys[x])
    return assigned_clusters


class UnidimensionalKmeans:

    def __init__(self, K, itermax):
        self.K = K
        self.itermax = itermax
        self.best_centroids = 'nothing yet'

    def fit(self, points):
        '''
        points : pyspark.rdd.RDD
        '''
        # initializing by assign random clusters to each point of points
        cluster_ids = points.map(lambda x: random.choice(range(self.K)))
        iteration = 0
        condition = True
        while condition & (iteration < self.itermax):
            centroids = compute_centroids(points, cluster_ids)
            cluster_ids_new = assign_clusters(points, centroids)

            # stop algorithm when clusters_ids doesn't change
            condition = (cluster_ids.collect() != cluster_ids_new.collect())

            # update index of clusters 
            cluster_ids = cluster_ids_new
            iteration += 1
        self.best_centroids = centroids  # save centroids 
        print('Done ! (in {} iterations)'.format(iteration))

    def predict(self, points):
        ''' 
        Return RDD with each points assigned to a cluster
        points : pyspark.rdd.RDD
        '''
        if not isinstance(self.best_centroids, str):
            return assign_clusters(points, self.best_centroids)
        else:
            print('Model need to be fitted before !')
            
            
class MultidimensionalKmeans(UnidimensionalKmeans):
    def __init__(self, K, itermax):
        super().__init__(K, itermax)

    def fit(self, points):
        '''
        points : pyspark.rdd.RDD
        '''
        points = points.map(lambda x : np.array(x)) # only change from super class
        cluster_ids = points.map(lambda x: random.choice(range(self.K)))
        iteration = 0
        condition = True
        while condition & (iteration < self.itermax):
            centroids = compute_centroids(points, cluster_ids)
            cluster_ids_new = assign_clusters(points, centroids)
            condition = (cluster_ids.collect() != cluster_ids_new.collect())
            cluster_ids = cluster_ids_new
            iteration += 1
        self.best_centroids = centroids
        print('Done ! (in {} iterations)'.format(iteration))
        
        
def cosin_distances(value, centroids):
     return [np.dot(value, centroid)/np.dot(centroid, centroid) for centroid in centroids]


def assign_clusters_spherical(points, centroids):
    '''
    Assign each points to a cluster, return a pyspark.rdd.RDD

    points : pyspark.rdd.RDD
    centroids : result of compute_centroids, pyspark.rdd.RDD
    '''
    means = centroids.values().collect()
    temp = points.map(lambda x: np.argmin(cosin_distances(x, means))) 
    # recover index of cluster's centroid
    centroids_keys = centroids.keys().collect() 
    assigned_clusters = temp.map(lambda x: centroids_keys[x])
    return assigned_clusters


class SphericalKmeans(UnidimensionalKmeans):
    def __init__(self, K, itermax):
        super().__init__(K, itermax)

    def fit(self, points):
        '''
        points : pyspark.rdd.RDD
        '''
        points = points.map(lambda x : np.array(x)/np.linalg.norm(np.array(x))) 
        # only change from super class, apply a normalization on vector
        cluster_ids = points.map(lambda x: random.choice(range(self.K)))
        iteration = 0
        condition = True
        while condition & (iteration < self.itermax): 
            centroids = compute_centroids(points, cluster_ids)
            # centroids may not be on the hypersphere
            cluster_ids_new = assign_clusters_spherical(points, centroids)
            condition = (cluster_ids.collect() != cluster_ids_new.collect())
            cluster_ids = cluster_ids_new
            iteration += 1
        self.best_centroids = centroids
        print('Done ! (in {} iterations)'.format(iteration))
        
    def predict(self, points):

        if not isinstance(self.best_centroids, str):
            return assign_clusters_spherical(points, self.best_centroids)
        else:
            print('Model need to be fitted before !')
            
if __name__ == '__main__':
    pass
