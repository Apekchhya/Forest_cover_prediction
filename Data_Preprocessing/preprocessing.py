from email.errors import StartBoundaryNotFoundDefect
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessing:
    def __init__(self, file_obj, log_obj):
        self.file_obj = file_obj
        self.log_obj = log_obj

    def column_removal(self, data, columns):
        self.log_obj.log(self.file_obj, 'Entered into column_removal method of Preprocessing class')
        self.data = data
        self.columns = columns
        try:
            self.useful_data = self.data.drop(labels = self.columns, axis = 1)
            self.log_obj.log(self.file_obj, 'Successfully removed columns. Exited the column_removal method of Preprocessing class')
            return self.useful_data

        except Exception as e:
            self.log_obj.log(self.file_obj, 'Exception occured in column_removal method of Preprocessing class. Exception message: ' + str(e))
            self.log_obj.log(self.file_obj, 'Column removal failed. Exited the column_removal method of class Preprocessing')

            raise Exception()

    def separate_feature_target(self, data, target_name):
        self.log_obj.log(self.file_obj, 'Entered separate_feature_target method of Preprocessing class')
        self.data = data
        try:
            self.X = data.drop(labels = target_name, axis = 1)
            self.Y = data[target_name]
            self.log_obj.log(self.file_obj, 'Feature target separation sucessful. Exited the separate_feature_target method of class Preprocessing ')
            return self.X, self.Y

        except Exception as e:
            self.log_obj.log(self.file_obj, ' Exception occured while separating feature-target. Exception message: ' + str(e))
            self.log_obj.log(self.file_obj, 'Separation of feature target unsucessful. Exited the separate_feature_target method of Preprocessing class')

            raise Exception()

    def null_value_presence(self,data):
        self.log_obj.log(self.file_obj, 'Entered null_value_presence method of Preprocessing class')
        try:
            self.null_count = data.isna().sum()
            for i in self.null_count:
                if i>0:
                    self.null_present = True
                    break
                if self.null_present:
                    df_with_null = pd.DataFrame()
                    df_with_null['columns'] = data.columns
                    df_with_null['null value count'] = np.asarray(data.isna().sum())
                    df_with_null.to_csv('Preprocessing_Data/null_values.csv')

            self.log_obj.log(self.file_obj, 'Sucessfully found missing values. Created separete dataframe for null values count. Exited null_value_presence method of class Preprocessing ')

        except Exception as e:
            self.log_obj.log(self.file_obj, 'Exception occurred in finding missing values. Exception message: '+str(e))
            self.log_obj.log(self.file_obj, 'Finding missing values unsuccessful. Exited null_value_presence method of class Preprocessing')

            raise Exception()

    def fill_null_values(self, data):
        self.log_obj.log(self.file_obj, 'Entered fill_null_values method of Preprocessing class')
        self.data = data
        try:
            imputer = KNNImputer(n_neighbors=3, weights = 'uniform', missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(data = self.new_array, columns=self.data.columns)
            self.log_obj.log(self.file_obj, 'Successfully imputed missing values. Exited fill_null_values method of class Preprocessing')

        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured while filling na. Exception message: '+str(e))
            self.log_obj.log(self.file_obj, 'Filling missing values failed. Exited fill_null_values method of class Preprocessing')

            raise Exception()

    def get_columns_with_zero_std_dev(self,data):
        self.log_obj.log(self.file_obj, 'Entered get_columns_with_zero_std_dev method of class Preprocessing')
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []
        try:
            for x in self.columns:
                if self.data_n[x]['std']==0:
                    self.col_to_drop.append(x)
            self.log_obj.log(self.file_obj,'Successsfully figured out column with zero std dev. Exited get_columns_with_zero_std_dev of class Preprocessing')
            return self.col_to_drop

        except Exception as e:
            self.log_obj.log(self.file_obj, 'Exception occured at get_columns_with_zero_std_dev method. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Getting columns with zero std dev failed. Exited get_columns_with_zero_std_dev. Exited the remove_columns_with_zero_std_dev method of class Preprocessing')
            raise Exception()

    def scaled_data(self,data):
        scaler = StandardScaler()
        num_data = data[['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology','horizontal_distance_to_roadways', 'horizontal_distance_to_fire_points' ]]
        cat_data = data.drop(['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology','horizontal_distance_to_roadways', 'horizontal_distance_to_fire_points' ], axis = 1)
        scaled_data = scaler.fit_transform(num_data)
        num_data = pd.DataFrame(scaled_data, columns = num_data.columns, index=num_data.index)
        final_data = pd.concat([num_data, cat_data], axis = 1)
        return final_data

    def encodeCategoricalvalues(self,data):
        data['class'] = data['class'].map({'Lodgepole_Pine':0,'Spruce_Fir':1,'Douglas_fir':2,'Krummholz':3,'Ponderosa_Pine':4,'Aspen':5,'Cottonwood_Willow':6})
        print(data)
        return data

    def imbal_data_handling(self, X, y):
        sample = SMOTE()
        X,y = sample.fit_resample(X,y)
        return X,y


    


        


                    





