import pandas as pd
from File_Operations import file_methods
from Data_Preprocessing import preprocessing, clustering
from Data_Loading import prediction_data_loader
from App_Logging import logger
import os

class Prediction:
    def __init__(self,path):
        self.file_obj = open("Prediction_Logs/prediction_log.txt",'a+')
        self.write_log = logger.Logging_App()
        print('Prediction onset')

    def predict(self):
        try:
            #write code for deleting existing prediction data
            self.write_log.log(self.file_obj,'Prediction started')
            getting_data = prediction_data_loader.Pred_Data_Getter(self.file_obj, self.write_log)
            data = getting_data.get_data()
           

            preprocessor = preprocessing.Preprocessing(self.file_obj, self.write_log)
            data = preprocessor.scaled_data(data)

            load_file = file_methods.File_Operation(self.file_obj, self.write_log)
            kmeans = load_file.load_model('KMeans_clustering')

            clusters = kmeans.predict(data)
            data['clusters']=clusters
            clusters = data['clusters'].unique()
            result = []
            
            for i in clusters:
                cluster_data = data[data['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'], axis = 1)
                model_name = load_file.find_correct_model(i)
                model = load_file.load_model(model_name)

                for value in (model.predict(cluster_data)):
                    if value == 0:
                        result.append('Lodgepole_Pine')
                    elif value == 1:
                        result.append('Spruce_Fir')
                    elif value == 2:
                        result.append('Douglas_Fir')
                    elif value == 3:
                        result.append('Krummholz')
                    elif value == 4:
                        result.append('Ponderosa_Pine')
                    elif value == 5:
                        result.append('Aspen')
                    elif value == 6:
                        result.append('Cottonwood_Willow')

                result = pd.DataFrame(result, columns = ['Prediction'])
                path = 'prediction_output'
                if len(os.listdir(path)) == 0:
                    result.to_csv(path + '/prediction.csv', header=True, mode='a+')
                    self.write_log.log(self.file_obj,'End of Prediction')

                else:
                    for files in os.listdir(path):
                        os.remove(path+'/'+files)
                        result.to_csv(path + '/prediction.csv', header=True, mode='a+')
                        self.write_log.log(self.file_obj,'End of Prediction')

        except Exception as e:
            self.write_log.log(self.file_obj,'Error occured while running prediction.Error message: '+str(e))
            raise Exception()

        return path