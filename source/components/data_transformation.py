import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from source.exception import CoustomException
from source.logger import logging
from source.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Transforms numerical and categorical data.
        """
        try:
            numerical_columns = ["reading score", "writing score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            logging.info("Num_columns: ",numerical_columns)
            logging.info("Cat_columns: ",categorical_columns)

            # for numerical columns, imputer will fill missing values with median values
            # and StandardScaler will scale numerical columns.
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical columns encoding completed.")

            # for categorical columns, imputer will replace missing values with most frequent value,
            # as number of unique values are less in all cat columns, using one hot encoding to encode columns,
            # standard scaler will scale all the columns.
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Categorical columns encoding completed.")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipelines", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CoustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train-test data done.")

            preprocessor = self.get_data_transformer_object()
            logging.info("Obtained preprocessor.")

            target_column = "math score"

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            input_feature_train_pre = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_pre = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_pre, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_pre, np.array(target_feature_test_df)]

            logging.info(
                "Application of preprocessing on train and test data completed."
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                object=preprocessor,
            )
            logging.info("Preprocessor object saved.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CoustomException(e, sys)
