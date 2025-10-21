import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind_from_stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf 
import os
from xai.models import load_transformer

import logging
import sys
logger = logging.getLogger('SurvivalEvaluator')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chandler = logging.StreamHandler()
chandler.setLevel(logging.DEBUG)
chandler.setFormatter(formatter)
logger.addHandler(chandler)
logger.setLevel(logging.DEBUG)

def split_vector(N, k):
    # Calculate the number of partitions required
    partitions = N // k
    remaining = N % k
    
    result = []
    start = 0
    
    # Split the vector into smaller vectors
    for i in range(partitions):
        # Calculate the end index of the current partition
        end = start + k
        
        # Add the (start, end) index pair to the result
        result.append((start, end))
        
        # Update the start index for the next partition
        start = end
    
    # If there are remaining elements, add an extra partition
    if remaining > 0:
        end = start + remaining
        result.append((start, end))
    
    return result

class Evaluator():
    
    def __init__(self, model, path=None, run=None, **kwargs):
        '''
        
        '''
        
        self.path = path
        self.run = run
        self.trainer = model


        
    def predict(self, data, normalize=True, batch_size=10000, iterations=10, **kwargs):
        '''
        Returns:
        --------
        β: [Iterations, Patients, 1]
        Ŵ: [Iterations, Layers, Patients, Heads, Features, Features] [deprecated not returned]
        Ô: [Iterations, Patients, Features, EmbeddingSize]
        
        --------
        
        '''
        dataloader = self.trainer.data_generator(
            data, 
            batch_size=data.shape[0], 
            normalize=normalize, 
            max_features=self.trainer.max_features, 
            shuffle=False,
            augment_copies=1
        )
        
        predictions = []
        attentions = []
        outputs = []
        tokens = []
        risks = []

        for iteration in range(iterations):
        
            X, y = dataloader.__getitem__(0)
            # print(X)
            Ŷ = []
            Ŵ = []
            Ô = []
            R = []
            

            indexes = split_vector(X[0].shape[0], k=batch_size)

            for start, end in indexes:
                x = [ r[start:end] for r in X[:3] ]
                ỹ, ŵ, ô, r, _ = self.trainer.model(x, training=False)

                Ŷ.append(ỹ.numpy())
                # Ŵ.append(ŵ)
                Ô.append(ô.numpy())
            

            Ŷ = np.concatenate(Ŷ)
            # Ŵ = np.concatenate(Ŵ)
            Ô = np.concatenate(Ô)
            
            features = np.array([[self.trainer.tokenizer.decoder[int(i)] for i in p] for p in X[1]])
            
            tokens.append(features)
            predictions.append(Ŷ)
            # attentions.append(Ŵ)
            outputs.append(Ô)

        return np.array(predictions), np.array(attentions), np.array(outputs), np.array(tokens)

    def performance(self, metric, split='validation'):
        
        try:
            events = EventAccumulator("{}/{}/{}/".format(self.path, self.run, split), size_guidance={'tensors': 0})
            events.Reload()

            r = pd.DataFrame([(w, s, float(np.array(tf.make_ndarray(t))) ) for w, s, t in events.Tensors(metric)],
                        columns=['wall_time', 'epoch', metric])
            r['run'] = self.run
            return r
        
        except:
            print("Metric not available, please select one of: {}".format(events.Tags()['tensors']))
            assert(1==0)





