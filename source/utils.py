import os
import sys
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from source.exception import CoustomException

def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):

    try:
        report = {}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred_train = model.predict(X_train)
            train_r2_score = r2_score(y_train,y_pred_train)

            y_pred_test = model.predict(X_test)
            test_r2_score = r2_score(y_test,y_pred_test)

            report[list(models.keys())[i]] = test_r2_score

        return report
    
    except Exception as e:
        raise CoustomException(e,sys)
