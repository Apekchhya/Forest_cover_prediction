from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

class Find_Model:
    def __init__(self,file_obj,log_obj):
        self.file_obj = file_obj
        self.log_obj = log_obj
        self.ran_clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective = 'binary:logistic')

    def get_best_params_for_ran_forest(self, train_X, train_y):
        self.log_obj.log(self.file_obj,'Entered get_best_params_for_ran_forest method of Find_Model class')
        try:
            self.param_grid = {
                'n_estimators':[10,50,100,130],
                'criterion': ['gini', 'entropy'],
                # # 'max_depth': range(2,4,1),
                'max_features':['auto','kig2']       
            }

            self.grid = GridSearchCV(estimator = self.ran_clf, param_grid= self.param_grid, cv=5, verbose=3,n_jobs=-1)
            self.grid.fit(train_X, train_y)

            self.criterion = self.grid.best_params_['criterion']
            # self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            self.ran_clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion= self.criterion, max_features=self.max_features)
            
            #Training the new model

            self.ran_clf.fit(train_X, train_y)
            self.log_obj.log(self.file_obj, 'Random Forest best params:'+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Find_Model class')

            return self.ran_clf

        except Exception as e:
            self.log_obj.log(self.file_obj, 'Exception occured in get_best_params_for_random_forest method of Model_Finder class. Exception message:'+str(e))
            self.log_obj.log(self.file_obj, 'Random Forest Classification Tuning Failed. Exited the get_best_params_for_random_forest method of the Find_Model class')
            raise Exception()

    def get_best_params_for_xgboost(self, train_X, train_y):
        self.log_obj.log(self.file_obj,'Entered the get_best_params_for_xgboost of Find_Model class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='multi:softprob'),self.param_grid_xgboost, verbose=3,cv=5,n_jobs=-1)
            # finding the best parameters
            self.grid.fit(train_X, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_X, train_y)
            self.log_obj.log(self.file_obj,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.log_obj.log(self.file_obj,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_obj.log(self.file_obj,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()



    def get_best_model(self,train_x,train_y,test_x,test_y):
        
        self.log_obj.log(self.file_obj,
                               'Entered the get_best_model method of the Find_Model class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)

            self.prediction_xgboost = self.xgboost.predict_proba(test_x) 

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.log_obj.log(self.file_obj, 'Accuracy for XGBoost:' + str(self.xgboost_score)) 
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost, multi_class='ovr') 
                self.log_obj.log(self.file_obj, 'AUC for XGBoost:' + str(self.xgboost_score))

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_ran_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict_proba(test_x) 

            if len(test_y.unique()) == 1:
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                self.log_obj.log(self.file_obj, 'Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest,multi_class='ovr') 
                self.log_obj.log(self.file_obj, 'AUC for RF:' + str(self.random_forest_score))

            #comparing the two models
            if(self.random_forest_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            else:
                return 'RandomForest',self.random_forest

        except Exception as e:
            self.log_obj.log(self.file_obj,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_obj.log(self.file_obj,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
