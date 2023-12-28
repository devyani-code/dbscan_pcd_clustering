
import numpy as np
import open3d
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

class MyDBScan:

    def downsample_data(self, pcd):
        downsample_pcd = pcd.voxel_down_sample(voxel_size = 0.4)
        print("\n Orignal number of pcd points : "+str(len(pcd.points))+"\n")
        print("Number of downsampled pcd points : "+str(len(downsample_pcd.points))+"\n")
        return downsample_pcd

    def segment_data(self, downsample_pcd):
        _, inliers = downsample_pcd.segment_plane(distance_threshold = 0.3, ransac_n = 3, num_iterations=250)
        inlier_cloud = downsample_pcd.select_by_index(inliers)
        outlier_cloud = downsample_pcd.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color([1,0,0])
        outlier_cloud.paint_uniform_color([0,0,1])
        return inlier_cloud, outlier_cloud
    
    def preprocess_lidar_data(self, path):
        pcd = open3d.io.read_point_cloud(path, format = 'auto')
        downsample_pcd = self.downsample_data(pcd)
        inlier_cloud, outlier_cloud = self.segment_data(downsample_pcd)
        data = np.asarray([outlier_cloud.points])
        data = data.reshape((len(outlier_cloud.points),3))
        return data, inlier_cloud, outlier_cloud

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1-p2)**2))
    
    def get_neighbors(self, X,pt,epsilon):
        neighbour=[]
        for index in range(X.shape[0]):
            if self.euclidean_distance(X[pt],X[index])<epsilon:
                neighbour.append(index)
        return neighbour
    
    def expand(self, X, clusters, point, neighbors, currentPoint, eps, minPts):
    
        clusters[point] = currentPoint
        
        i = 0
        while i < len(neighbors):
            
            nextPoint = neighbors[i]
            
            if clusters[nextPoint] == -1:
                clusters[nextPoint] = currentPoint
            
            elif clusters[nextPoint] == 0:
                clusters[nextPoint] = currentPoint
                
                nextNeighbors = self.get_neighbors(X, nextPoint, eps)
                
                if len(nextNeighbors) >= minPts:
                    neighbors = neighbors + nextNeighbors
            
            i += 1

    def simple_DBSCAN(self, X, clusters, eps, minPts):
        currentPoint = 0
        for i in range(0, X.shape[0]):
            print(i)
            
            if clusters[i] != 0:
                continue
            neighbors = self.get_neighbors(X, i, eps)

            if len(neighbors) < minPts:
                clusters[i] = -1

            else:
                currentPoint += 1
                self.expand(X, clusters, i, neighbors, currentPoint, eps, minPts)
        
        return clusters


    def get_neighbors_optimized(self, X, pt, epsilon, nn):
        indices = nn.radius_neighbors([X[pt]], radius=epsilon, return_distance=False)
        return indices[0]

    def expand_optimized(self, X, clusters, point, neighbors, currentPoint, eps, minPts, nn):
        clusters[point] = currentPoint
        i = 0
        while i < len(neighbors):
            nextPoint = neighbors[i]
            if clusters[nextPoint] == -1:
                clusters[nextPoint] = currentPoint
            elif clusters[nextPoint] == 0:
                clusters[nextPoint] = currentPoint
                nextNeighbors = self.get_neighbors_optimized(X, nextPoint, eps, nn)
                
                if len(nextNeighbors) >= minPts:
                    neighbors = np.concatenate((neighbors, nextNeighbors))
            i += 1
    
    def optimized_fit(self, X, clusters, eps, minPts):
        currentPoint = 0
        nn = NearestNeighbors(radius=eps)
        nn.fit(X)
        
        for i in range(0, X.shape[0]):
            print(i)
            if clusters[i] != 0:
                continue
        
            neighbors = self.get_neighbors_optimized(X, i, eps, nn)

            if len(neighbors) < minPts:
                clusters[i] = -1

            else:
                currentPoint += 1
                self.expand_optimized(X, clusters, i, neighbors, currentPoint, eps, minPts, nn)
                
        return clusters
    
    def visualize_clusters(self, c, inlier_cloud, outlier_cloud):
        max_label = c.max()
        colors = plt.get_cmap("tab20")(c/(max_label if max_label >0 else 1))
        colors[c<0] = 0

        outlier_cloud.colors = open3d.utility.Vector3dVector(colors[:,:3])
        Min_points=20
        obbs=[]
        indexes = pd.Series(range(len(c))).groupby(c,sort = False).apply(list).tolist()
        for i in range(0,len(indexes)):
            nb_pts = len(outlier_cloud.select_by_index(indexes[i]).points)
            if nb_pts>Min_points:
                
                sub_cloud = outlier_cloud.select_by_index(indexes[i])
                obb = sub_cloud.get_axis_aligned_bounding_box()
                obb.color = (0,0,1)
                obbs.append(obb)
        list_of_visuals = []
        list_of_visuals.append(outlier_cloud)
        list_of_visuals.extend(obbs)
        list_of_visuals.append(inlier_cloud)
        open3d.visualization.draw_geometries(list_of_visuals)
    
if __name__ == '__main__':
    dbscan = MyDBScan()

    # Defining the path for the file
    path = "Task2\\assignment\\pcd_10.pcd"

    # Reads the pointcloud data , performs downsampling and segmentation
    # Returns the pcd data (for outliers to further perform object detection), inliner and outlier clouds.  
    data, inlier_cloud, outlier_cloud = dbscan.preprocess_lidar_data(path)
    cluster = [0]*data.shape[0]

    # Performs optimized dbscan
    #c = dbscan.optimized_fit(data,cluster,eps=0.55,minPts=7)

    # Performs unoptimized dbscan uncomment below code and comment line 159 to perform unoptimized code.
    c = dbscan.simple_DBSCAN(data,cluster,eps=0.55,minPts=7)
    
    c = np.array(c)
    dbscan.visualize_clusters(c, inlier_cloud, outlier_cloud)

