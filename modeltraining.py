from pyexpat import model
from sklearn.model_selection import train_test_split
from Data_Loading import training_data_loader
from Data_Preprocessing import preprocessing, clustering
from Finding_model import find_model
from File_Operations import file_methods
from App_Logging import logger

class Trainmodel:

    def __init__(self):
        self.write_log = logger.Logging_App()
        self.file_obj = open('Training_Logs/Training_model_logs.txt', 'a+')

    def model_training(self):
        self.write_log.log(self.file_obj,'Training started')

        try:
            data_getter = training_data_loader.Data_Getter(self.file_obj,self.write_log)
            data = data_getter.get_data()
            preprocessor = preprocessing.Preprocessing(self.file_obj, self.write_log)
            data = preprocessor.encodeCategoricalvalues(data)
            X = data.drop(['class'], axis=1)
            Y = data['class']
            X, Y = preprocessor.imbal_data_handling(X,Y)
            kmeans = clustering.Clustering(self.file_obj, self.write_log)
            num_of_clusters = kmeans.elbow_plot(X)
            X = kmeans.create_cluster(X, num_of_clusters)
            X['Labels']=Y
            list_of_clusters = X['cluster'].unique()

            for i in list_of_clusters:
                cluster_data = X[X['cluster']==i]
                cluster_features = cluster_data.drop(['Labels','cluster'], axis = 1)
                cluster_label = cluster_data['Labels']

                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size = 1/3, random_state = 355)
                x_train = preprocessor.scaled_data(x_train)
                x_test = preprocessor.scaled_data(x_test)
                model_finder = find_model.Find_Model(self.file_obj, self.write_log)
                best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)
                file_op = file_methods.File_Operation(self.file_obj, self.write_log)
                save_model = file_op.model_saving(best_model, best_model_name + str(i))

            self.write_log.log(self.file_obj, 'Successful end of Model Training')
            self.file_obj.close()

        except Exception:
            self.write_log.log(self.file_obj,'Model training unsuccessful')
            self.file_obj.close()
            raise Exception



