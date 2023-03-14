import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from source.exception import CoustomException
from source.logger import logging
from source.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting train and test data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random_forest": RandomForestRegressor(),
                "Decison_tree": DecisionTreeRegressor(),
                "Gradient_Boosting": GradientBoostingRegressor(),
                "Linear_regrassor": LinearRegression(),
                "K-Neighbours_regressor": KNeighborsRegressor(),
                "XGBoost_regressor": XGBRegressor(),
                "Adaboost_regressor": AdaBoostRegressor(),
            }

            logging.info("Model report intiated.")
            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(
                "Best model is {} with score of {} before tuning.".format(
                    best_model_name, best_model_score
                )
            )

            if best_model_score < 0.6:
                raise CoustomException("No best model found.")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            y_pred = best_model.predict(X_test)
            best_score = r2_score(y_test, y_pred)

            ###Model Tuning
            logging.info('Enter model tuning.')
            param = dict()
            param['fit_intercept'] = [True,False]
            param['n_jobs'] = [-1,2,3,1]
            param['positive']=[True,False]

            X = np.concatenate((X_train,X_test),axis=0)
            y = np.concatenate((y_train,y_test),axis=0)
            search = GridSearchCV(best_model, param_grid=param, n_jobs=-1)
            result = search.fit(X,y)
            print('Score after Tuning: %s' % result.best_score_)
            print('Best Hyperparameters: %s' % result.best_params_)
            logging.info('Modle tuned with score {}'.format(result.best_score_))

            if (best_score>=result.best_score_):
                return best_score
            else:
                return result.best_score_

        except Exception as e:
            raise CoustomException(e, sys)
