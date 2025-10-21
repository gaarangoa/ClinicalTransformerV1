import tensorflow as tf 
import numpy as np 

import logging
import sys
logger = logging.getLogger('SurvivalDataLoader')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chandler = logging.StreamHandler()
chandler.setLevel(logging.DEBUG)
chandler.setFormatter(formatter)
logger.addHandler(chandler)
logger.setLevel(logging.DEBUG)

class DataGenerator(tf.keras.utils.Sequence):
    '''

    Add <pad>:0 to continuous zero features. Each patient has to have >0 value to be considered in the data. 

    # data: is a dataframe
    tokenizer = tokenizer(data)

    discrete_features = []
    continuous_features = []

    target = ['time', 'event', 'lambda', 'treatment']

    dataloader = DataGenerator(
        data, discrete_features, continuous_features, tokenizer, target=target
    )

    dataloader.__getitem__(0)

    '''
    def __init__(self, data, discrete_features, continuous_features, tokenizer, priors=[], max_features=0, max_features_percentile=95, time='OS_MONTHS', event='OS_STATUS', target=False, batch_size=32, shuffle=True, return_index=False, add_mask=False, augment_copies=2, training=False):
        
        # target is never used in the survival data loader, the parameters used in here are time and event. 
        # Training parameter is not used 
        # shuffle parameter is used to subsample the population on each epoch.
        # Add mask is not used: This is for self supervised learning.

        self.augment_copies = augment_copies
        self.data = data
        self.features = discrete_features + continuous_features
        self.is_training = training

        self.discrete_features = {i:True for i in discrete_features}
        self.continuous_features = {i:True for i in continuous_features}

        self.X_dict = data[self.features].T.to_dict()
        self.y_dict = data[[time, event]].T.to_dict()
        self.time=time
        self.event=event
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.add_mask = add_mask

        self.fix_data()

        self.priors = priors
        self.with_priors = True if len(priors) > 0 else False
        self.return_index = return_index
        
        if max_features == 0:
            self.max_features = int(np.floor(np.percentile(np.sum(data.fillna(0)[self.features] > 0, axis=1), max_features_percentile)))
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
    
    # This happens on epoch end. This means that this function never gets called when we are working 
    # on testing the model and the order of the patient data is always the same!!
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
        '''Alternative function to decrease bootleneck of data loading. Useful for very large datasets.'''

        # logger.info('Loading data in memory and extracting non NaNs')
        self.X_dict_fixed = {}
        for ox in self.X_dict.keys():
            self.X_dict_fixed[ox] = [self.adjust_discrete_zero_value(i) for i in self.X_dict[ox].items() if self.is_discrete_feature(i) == True]

    def is_discrete_feature(self, f):
        try: 
            if self.discrete_features[f[0]]:
                if np.isnan(f[1]) == True:
                    # TODO: Implement NaNs on the discrete variables. Right now those are encoded with a number.
                    # If the discrete variable has a NaN. You need to make sure the nans are categories in your data. 
                    return False
                else:
                    return True
        except:
            # This means it is a continuous feature.
            # We need to make sure the not available data is masked with NaN in the data. e.g., 
            # mutations are added as numerical features, we need to account for it. 

            if np.isnan(f[1]) == True:
                return False
            else:
                return True
    
    def adjust_discrete_zero_value(self, f):
        # Discrete variables are added a 1 to their value, so the 0's are not added. 
        # Having a 0 value has a negative effect on the embeddings, so, we added from 
        # 1 to N+1 values on the dicrete variable
        try: 
            if self.discrete_features[f[0]]:
                return f[0], f[1] + 1
        except:
            return f[0], f[1]

    def __data_generation(self, indexes):
        '''Genrates data containing batch size samples'''
        
        # Creates 5 versions of the same patient
        XNs = []
        XFs = []
        XPs = []
        ys = []
        Is = []
        for _ in range(self.augment_copies):

            # add 1 token for <cls>
            XN = np.zeros((self.batch_size, self.max_features + 1), dtype=np.float32)
            XF = np.zeros((self.batch_size, self.max_features + 1), dtype=np.float32)
            
            if self.with_priors:
                XP = np.zeros((self.batch_size, self.max_features + 1, self.priors.shape[0]))
            
            y = np.zeros((self.batch_size, 2), dtype=np.float32)
            I = np.zeros((self.batch_size, 1), dtype=np.int32)

            # Generate data
            ## Select non zero features
            kx=0
            for ix, ox in enumerate(indexes):
                # vector = [i for i in self.X_dict[ox].items() if i[1] > 0]
                vector = self.X_dict_fixed[ox] #[self.adjust_discrete_zero_value(i) for i in self.X_dict[ox].items() if self.is_discrete_feature(i) == True]
                
                if len(vector) > self.max_features:
                    np.random.shuffle(vector)
                    vector = vector[:self.max_features]
                
                padding = [['<pad>', 0]]*(self.max_features - len(vector))
                surv = [['<cls>', 0]]
                
                np.random.shuffle(vector)
                vector = surv + vector + padding
                
                XN[kx, :] = [i[1] for i in vector] 
                XF[kx, :] = [ self.tokenizer.encoder[i[0]] for i in vector]
                
                if self.with_priors:
                    for feat_ix, [feat_id, value] in enumerate(vector):
                        try:
                            XP[kx, feat_ix, :] = self.priors[feat_id]
                        except:
                            pass
                
                y[kx] = [self.y_dict[ox][self.time], self.y_dict[ox][self.event]]
                I[kx] = kx
                kx+=1
        
            XNs.append(XN[:kx]) 
            XFs.append(XF[:kx]) 
            if self.with_priors:
                XPs.append(XP[:kx])
            Is.append(I[:kx])
            ys.append(y[:kx])

        XN = np.concatenate(XNs)
        XF = np.concatenate(XFs)
        if self.with_priors:
            XP = np.concatenate(XPs)
        y = np.concatenate(ys)
        I = np.concatenate(Is)

        if self.with_priors:
            return [XN, XF, XP, I, np.array([len(indexes)])], y[:kx]
        else:
            return [XN, XF, [], I, np.array([len(indexes)])], y[:kx]
        