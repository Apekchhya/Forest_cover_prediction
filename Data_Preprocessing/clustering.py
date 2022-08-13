import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sympy import Curve
from kneed import KneeLocator
from File_Operations import file_methods

class Clustering:
    def __init__(self,file_obj, log_obj):
        self.file_obj = file_obj
        self.log_obj = log_obj

    def elbow_plot(self, data):
        self.log_obj.log(self.file_obj, 'Entered the elbow_plot method of class Clustering')
        wcss = []
        try: 
            for i in range(1,11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss)
            plt.title('The Elbow Method')
            plt.xlabel('No. of clusters')
            plt.ylabel('wcss')
            plt.savefig('Preprocessing_Data/Elbow.PNG')
            self.kn = KneeLocator(range(1,11),wcss, curve = 'convex', direction = 'decreasing')

            self.log_obj.log(self.file_obj,'The optimum number of cluster is : ' + str(self.kn.knee)+ '. Exited the elbow_plot method of class Clustering')

            return self.kn.knee

        except Exception as e:
            self.log_obj.log(self.file_obj, 'Exception occured while making elbow plot. Exception message:  '+str(e))
            self.log_obj.log(self.file_obj, 'Elbow plot is not created. Exited elbow plot method of class Clustering')
            raise Exception()

    def create_cluster(self, data, number_of_clusters):
        self.log_obj.log(self.file_obj, 'Entered the create_cluster method of class Clustering')
        self.data = data
        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init = 'k-means++', random_state=42)
            self.y_means = self.kmeans.fit_predict(data)
            self.file_opr = file_methods.File_Operation(self.file_obj,self.log_obj)
            self.save_model = self.file_opr.model_saving(self.kmeans, 'KMeans_clustering')
            self.data['cluster'] = self.y_means
            self.log_obj.log(self.file_obj, 'Sucessfully created cluster. Exited create_cluster method of class Clustering')
            return self.data

        except Exception as e:
            self.log_obj.log(self.file_obj, 'Exception occured while creating cluster. Exception message: '+str(e))
            self.log_obj.log(self.file_obj, 'Fitting data to KMeans failed. Exited the create_cluster method of class Clustering')
            
            raise Exception()
            


