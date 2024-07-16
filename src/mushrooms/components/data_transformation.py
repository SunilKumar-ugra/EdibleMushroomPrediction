from mushrooms.entity import DataTransformationConfig
import os
import pandas as pd
from mushrooms import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def scaling(self,dataset):
        '''Scaling Feature'''
        
        scaling_feature=[feature for feature in dataset.columns if feature not in ['class'] ]
        scaler=MinMaxScaler()
        scaler.fit(dataset[scaling_feature])
        data = pd.concat([dataset[['class']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[scaling_feature]), columns=scaling_feature)],
                    axis=1)
        logger.info("Completed scaling dataset")
        return(data)
        
    def train_test_spliting(self,data):
        #data = pd.read_csv(self.config.data_path)
        
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        
    
    def transformation(self):
        data = pd.read_csv(self.config.data_path)
        logger.info("Converted CSV data to DataFrame")
        
        #  scaling the dependent variable
        data=self.scaling(data)
        self.train_test_spliting(data)
        
        
        
