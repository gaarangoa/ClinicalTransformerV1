from sklearn.model_selection import train_test_split
import numpy as np 


class DataLoader():
    '''
    This dataset generates a training dataframe. 
    '''

    def __init__(self, dataset, target_class, features=[], normalize=True, test_size=0.2):
        self.normalize = normalize
        self.test_size = test_size
        self.dataset = dataset
        self.target_class = target_class
        self.features = features
        
        self.__call__()
    
    def __call__(self, ): 
        
        if not self.features:
            self.features = self.dataset.columns[self.dataset.columns != self.target_class]
        
        # Shuffle the dataset keeping integrity of the data
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        self.normalized_dataset = self.dataset[ self.features ]
        self.dataset_min = self.normalized_dataset.min(axis=0)
        self.dataset_max = self.normalized_dataset.max(axis=0)
        
        if self.normalize: 
            self.normalized_dataset = ((self.normalized_dataset - self.dataset_min) / (self.dataset_max - self.dataset_min)).fillna(0)
        
        # Split dataset into training / testing
        X_train_0, X_test, y_train_0, y_test = train_test_split(
            self.normalized_dataset, 
            np.array(self.dataset[self.target_class]), 
            test_size=self.test_size
        )

        # Split the training dataset into training and validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_0,
            y_train_0,
            test_size=self.test_size
        )

        self.X_train, self.X_valid, self.X_test = X_train.reset_index(drop=True), X_valid.reset_index(drop=True), X_test.reset_index(drop=True)
        self.y_train, self.y_valid, self.y_test = y_train, y_valid, y_test
 