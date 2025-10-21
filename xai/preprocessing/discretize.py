import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from collections import Counter
# from xai.preprocessing.utils import LabelEncoder1Plus

import logging
logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def is_unique(df):
    binaries = []
    for i in df.columns:
        val = df[i].dropna()
        items = list(Counter(val).keys())
        
        if len(items) == 1: 
            binaries.append(i)
            
    return binaries

def is_binary(df):
    binaries = []
    for i in df.columns:
        val = df[i].dropna()
        items = list(Counter(val).keys())
        
        if len(items) == 2: 
            binaries.append(i)
            
    return binaries

class CategoricalConverter():
    '''
    Encode all categorical variables in a dataframe into numbers. 

    Example
    ---------
    categorical_features.categorical_encoders['CANCER_TYPE'].inverse_transform([2])
    

    '''
    def __init__(self, ): 
        pass
        
    def encode(self, df):
        
        self.df = df.copy()
        self.categorical_encoders = {}
        self.categorical_features = self.df.columns[~self.df.columns.isin(self.df.select_dtypes(include=np.number).columns.tolist())].tolist()

        for categorical_feature in self.categorical_features: 
            
            logger.info('Categorical Feature: {}'.format(categorical_feature))

            encoder = preprocessing.LabelEncoder()
            # encoder = LabelEncoder1Plus()
            encoder.fit(self.df[categorical_feature])
            
            self.categorical_encoders.update(
                {categorical_feature: encoder}
            )

        self.df = []

    def transform(self, data):
        df = data.copy()
        for categorical_feature in self.categorical_features: 
            try:
                df[categorical_feature] = self.categorical_encoders[categorical_feature].transform(df[categorical_feature])
            except:
                logger.warning( 'Categorical feature not available in your input data: {}'.format(categorical_feature) )
   
        return df

class BinaryConverter():
    def __init__(self, ):
        pass
    
    def encode_feature(self, dataset, feature):
        encoder = preprocessing.OneHotEncoder(sparse=False)
        encoder.fit(dataset[[feature]])

        values = encoder.transform(dataset[[feature]])
        categs = encoder.categories_[0]

        bindf = pd.DataFrame(values, columns=["{}_{}".format(feature, i) for i in categs])

        return encoder, bindf
    
    def encode(self, df, ignore=[]):
        self.df = df.copy()
        self.categorical_encoders = {}
        self.categorical_features = self.df.columns[~self.df.columns.isin(self.df.select_dtypes(include=np.number).columns.tolist())].tolist()
        self.binary_features = is_binary(self.df)
        
        self.categorical_features = [i for i in set(self.categorical_features + self.binary_features) if not i in ignore]

        for categorical_feature in self.categorical_features: 
            
            logger.info('Categorical Feature: {}'.format(categorical_feature))

            encoder = preprocessing.OneHotEncoder(sparse=False)
            encoder.fit(self.df[[categorical_feature]])
            
            self.categorical_encoders.update(
                {categorical_feature: encoder}
            )
            
    def transform(self, data):
        df = data.copy()
        
        df_cat = []
        for categorical_feature in self.categorical_features: 
            
            encoder = self.categorical_encoders[categorical_feature]
            
            values = encoder.transform(df[[categorical_feature]])
            categs = encoder.categories_[0]
            
            bindf = pd.DataFrame(values, columns=["Categorical_{}__{}".format(categorical_feature, i) for i in categs])
            
            df_cat.append(bindf)
            
        df_cat = pd.concat(df_cat, axis=1)
        df_num = df[df.columns[~(df.columns.isin(self.categorical_features))]]
        return pd.concat([df_cat, df_num], axis=1)