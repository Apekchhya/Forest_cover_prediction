import pandas as pd


class Pred_Data_Getter:

    def __init__(self, file_obj, log_obj):
        self.prediction_file = 'prediction_data/input.csv'
        self.file_obj = file_obj
        self.log_obj = log_obj
        print('Prediction_data_getter')

    def get_data(self):
        self.log_obj.log(self.file_obj,'Entered the get data method of Pred Data Getter class')
        print('get_data_entered')

        try:
            self.data = pd.read_csv(self.prediction_file)
            self.log_obj.log(self.file_obj,'Data successfully loaded. Exited the get data method of Pred Data Getter class')
            print('data sucessfully loaded')
            # print(self.data)
            return self.data
        except Exception as e: 
            self.log_obj.log(self.file_obj, 'Exception occurred in get data method of Pred Data Getter class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj, 'Data loading unsuccessful. Exited the get data method of Pred Data Getter Class')

            raise Exception()
