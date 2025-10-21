import tensorflow as tf 
import numpy as np 

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, discrete_features, continuous_features, priors, tokenizer, max_features=0, max_features_percentile=95, time='OS_MONTHS', event='OS_STATUS', target=None, batch_size=32, shuffle=True, return_index=False, add_mask=False, augment_copies=None, training=None):
        '''Padding data generator, Features with 0 values (or missing) are added the token <pad> to be ignored in the transformer'''
        features = discrete_features + continuous_features
        self.discrete_features = {i:True for i in discrete_features}
        self.continuous_features = {i:True for i in continuous_features}
        
        self.add_mask = add_mask
        self.data = data
        self.X_dict = data[features].T.to_dict()
        self.features = features    
        
        self.batch_size=batch_size
        self.shuffle=shuffle

        self.priors = priors
        self.with_priors = True if len(priors) > 0 else False
        self.return_index = return_index
        
        if max_features == 0:
            self.max_features = int(np.floor(np.percentile(np.sum(data[features] > 0, axis=1), max_features_percentile)))
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
    
    def is_discrete_feature(self, f):
        try: 
            return self.discrete_features[f[0]]
        except:
            # This means it is a continuous feature
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
                return f[0], f[1] + 1, 1, f[0]
        except:
            return f[0], f[1], 1, f[0] # token_key, token_value, 1:not-pad-token, token_key


    def __data_generation(self, indexes):
        '''Genrates data containing batch size samples'''
        XN = np.zeros((self.batch_size, self.max_features + 1), dtype=np.float32)
        XF = np.zeros((self.batch_size, self.max_features + 1), dtype=np.float32)
    
        if self.with_priors:
            XP = np.zeros((self.batch_size, self.max_features + 1, self.priors.shape[0]))
        
        pad_i = np.ones((self.batch_size, self.max_features + 1), dtype=np.float32)
        
        y = np.ones((self.batch_size, self.max_features + 1, 4), dtype=np.float32)
        
        # Generate data
        ## Select non zero features
        kx=0
        for ix, ox in enumerate(indexes):
            # If it is a discrete feature returns True, if it is a continuous feature with a value > 0 then returns true. In any other case
            # returns False, it means, featuers that are not present in the sample. This model assumes that 0 values are missing in the data.

            vector = [self.adjust_discrete_zero_value(i) for i in self.X_dict[ox].items() if self.is_discrete_feature(i) == True]
            # unseen_features = [i[0] for i in self.X_dict[ox].items() if self.is_discrete_feature(i) == False]
            # np.random.shuffle(unseen_features)
            
            if len(vector) > self.max_features:
                np.random.shuffle(vector)
                vector = vector[:self.max_features]
            
            # Completes the input feature space by adding <pad> tokens and zero values. 
            padding = [['<pad>', 0, 0, '<pad>']]*(self.max_features - len(vector)) # key, value, 
            cls_token = [['<cls>', 1, 2, '<cls>']]
            
            np.random.shuffle(vector)
            # 10% random masked input vectors. 
            lim = int(len(vector) * 0.2)
            if lim == 0:
                lim = 1
            
            vector_unmasked = vector[lim: ]
            
            if self.add_mask:
                vector_masked = [ ['<mask>', i[1], 2, i[0]] for i in vector[ :lim]]
            else:
                vector_masked = vector[ :lim]
        
            vector = vector_unmasked + vector_masked
            np.random.shuffle(vector)

            vector = cls_token + vector + padding
            
            XN[ix, :] = [i[1] for i in vector]
            XF[ix, :] = [ self.tokenizer.encoder[i[0]] for i in vector]
            
            if self.with_priors:
                for feat_ix, [feat_id, value, _, _] in enumerate(vector):
                    try:
                        XP[ix, feat_ix, :] = self.priors[feat_id]
                    except:
                        pass
            
            y[ix, :, 0] = [ 1 if i[2] == 1 else 0 for i in vector] # <pad> token value 1
            y[ix, :, 1] = [ self.tokenizer.encoder[i[3]] for i in vector] # tokens
            y[ix, :, 2] = [ 1 if i[2] == 2 else 0 for i in vector] # <mask> token value 2
            y[ix, :, 3] = [ i[1] for i in vector] # token values

            pad_i[ix, :] = [ 1 if i[2] == 1 else 0 for i in vector]

            kx+=1
            
        if self.with_priors:
            return [XN[:kx], XF[:kx], XP[:kx], pad_i[:kx]], y[:kx]
        else:
            return [XN[:kx], XF[:kx], [], pad_i[:kx]], y[:kx]