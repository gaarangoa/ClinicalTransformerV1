import tensorflow as tf 
import numpy as np 
import logging
import sys
logger = logging.getLogger('Evaluator')

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, discrete_features, continuous_features, priors, tokenizer, max_features=0, max_features_percentile=95, time='OS_MONTHS', event='OS_STATUS', target=None, batch_size=32, shuffle=True, return_index=False, add_mask=False, augment_copies=None, training=None):
        
        '''
        This data loader ignores values equals to NaN's
        This data loader adds 1 to the discrete values (starts the categories from 1). 
        '''

        features = discrete_features + continuous_features
        self.discrete_features = {i:True for i in discrete_features}
        self.continuous_features = {i:True for i in continuous_features}
        
        self.target = target
        self.add_mask = add_mask
        self.data = data
        self.X_dict = data[features].T.to_dict()
        self.y_dict = data[self.target].T.to_dict()

        self.features = features    
        
        logger.info('START: Processing data ...')
        self.fix_data()
        logger.info('END: Processing data ...')

        self.batch_size=batch_size
        self.shuffle=shuffle

        self.priors = priors
        self.with_priors = True if len(priors) > 0 else False
        self.return_index = return_index
        
        if max_features == 0:
            self.max_features = int(np.floor(np.percentile(np.sum(data[self.continuous_features].fillna(0) > 0, axis=1) + len(self.discrete_features), max_features_percentile)))
        else:
            self.max_features = max_features
        
        self.tokenizer = tokenizer
        
        self.on_epoch_end()
        
    def __len__(self, ):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_dict) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_dict))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def get_prior(self, x):
        try:
            return np.array(self.priors[x.replace('molecular_', '')])
        except:
            return False
    
    def fix_data(self, ):
        ''' Alternative function to decrease bootleneck of data loading
            This function gets the features per patient first and then can 
            be used in the main loop. This decreases significantly the processing
            time and makes the best use of GPU. 
        '''

        # logger.info('Loading data in memory and extracting non NaNs')

        # In here we first ask if the feature is a discrete feature. If it is, then we return
        # its value + 1. If it is not discrete, then we ask whether the variable value is NaN. If it is NaN 
        # we ignore this feature. We are not ignoring NaN's on discrete variables. 

        # The name is_discrete_feature is probably missleading and should be called (Ignore_Feature_NaNs). 
        self.X_dict_fixed = {}
        for ox in self.X_dict.keys():
            self.X_dict_fixed[ox] = [self.adjust_discrete_zero_value(i) for i in self.X_dict[ox].items() if self.is_discrete_feature(i) == True]

    def is_discrete_feature(self, f):
        # Returns True if it is a discrete feature, 
        # Returns False if it is a discrete feature with NaN value
        # Returns False if it is a continous feature with NaN value
        # Returns True if it is a continuous feature with real value
        try: 
            if self.discrete_features[f[0]]:
                if np.isnan(f[1]):
                    return False
                else:
                    return True
        except:
            # This means it is a continuous feature
            # If the feature value is NaN, then ignore the feature (returns False)
            if np.isnan(f[1]) == True:
                return False
            else:
                return True

    def adjust_discrete_zero_value(self, f):
        # Discrete variables are added a 1 to their value, so the 0's are not added. 
        # Having a 0 value has a negative effect on the embeddings, so, we added from 
        # 1 to N+1 values on the dicrete variable. 

        try: 
            if self.discrete_features[f[0]]:
                return f[0], f[1] + 1, 1, f[0]
        except:
            return f[0], f[1], 1, f[0]
    
    def __data_generation(self, indexes):
        '''Genrates data containing batch size samples'''
        XN = np.zeros((self.batch_size, self.max_features + 1), dtype=np.float32)
        XF = np.zeros((self.batch_size, self.max_features + 1), dtype=np.float32)
    
        if self.with_priors:
            XP = np.zeros((self.batch_size, self.max_features + 1, self.priors.shape[0]))
        
        pad_i = np.zeros((self.batch_size, self.max_features + 1), dtype=np.float32)
        
        y = np.zeros((self.batch_size, 1), dtype=np.float32)
        
        # Generate data
        ## Select non zero features
        kx=0
        for ix, ox in enumerate(indexes):
            # Ignore features with np.nan, categorical or numerical

            vector = self.X_dict_fixed[ox] # [self.adjust_discrete_zero_value(i) for i in self.X_dict[ox].items() if self.is_discrete_feature(i) == True]
            # unseen_features = [i[0] for i in self.X_dict[ox].items() if self.is_discrete_feature(i[1]) == False]
            # np.random.shuffle(unseen_features)
            
            if len(vector) > self.max_features:
                np.random.shuffle(vector)
                vector = vector[:self.max_features]
            
            # Completes the input feature space by adding <pad> tokens and zero values. 
            padding = [['<pad>', 0, 0, '<pad>']]*(self.max_features - len(vector))
            cls_ = [['<cls>', 1, 2, '<cls>']]

            np.random.shuffle(vector)
            vector = cls_ + vector + padding
            
            XN[ix, :] = [i[1] for i in vector]
            XF[ix, :] = [ self.tokenizer.encoder[i[0]] for i in vector]
            
            if self.with_priors:
                for feat_ix, [feat_id, value, _, _] in enumerate(vector):
                    try:
                        XP[ix, feat_ix, :] = self.priors[feat_id]
                    except:
                        pass
            
            y[ix] = self.y_dict[ox][self.target[0]]
            pad_i[ix, :] = [ 1 if i[2] == 1 else 0 for i in vector]
            

            kx+=1
            
        if self.with_priors:
            return [XN[:kx], XF[:kx], XP[:kx], pad_i[:kx]], y[:kx]
        else:
            return [XN[:kx], XF[:kx], [], pad_i[:kx]], y[:kx]