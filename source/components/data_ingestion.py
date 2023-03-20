import os
import sys
from source.exception import CustomException
from source.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from source.components.data_transformation import DataTransformation
from source.components.data_transformation import DataTransformationConfig

from source.components.model_trainer import ModelTrainer
from source.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingistion method or component')
        try:
            df = pd.read_csv('D:/MLProject/notebook/data/StudentsPerformance.csv')
            df.columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course','math_score','reading_score','writing_score']
            logging.info('Read data and load as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train test split start.')

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data completed.')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array,test_array,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    score = modeltrainer.initiate_model_trainer(train_array,test_array)
    print('Best score: ',score)


