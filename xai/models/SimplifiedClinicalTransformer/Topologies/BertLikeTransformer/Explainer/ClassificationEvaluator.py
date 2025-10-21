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
logger = logging.getLogger('ClassifierEvaluator')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chandler = logging.StreamHandler()
chandler.setLevel(logging.DEBUG)
chandler.setFormatter(formatter)
logger.addHandler(chandler)
logger.setLevel(logging.DEBUG)


class Evaluator():
    def __init__(self, model, epoch=None, **kwargs):
        '''
        
        '''
        self.path = kwargs.get('path', None)
        self.run = kwargs.get('run', None)

        self.epoch = epoch
        self.trainer = model


        
    def predict(self, data, normalize=True, batch_size=10000, iterations=10):

        '''
        Returns:
        --------
        β: [Iterations, Patients, 1]
        Ŵ: [Iterations, Layers, Patients, Heads, Features, Features]
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

            Ŷ = []
            Ŵ = []
            Ô = []
            R = []
            
            output = []
            counts = 0
            total = X[0].shape[0]
            while counts <= total:
                x = [ r[counts:counts+batch_size] for r in X[:3] ]
                ỹ, ŵ, ô = self.trainer.model(x, training=False)

                # mask = create_padding_mask(x[1])
                
                # try:
                #     ỹ, ŵ, ô = self.trainer.model(x, training=False, mask=mask)
                # except:
                #     ỹ, ŵ, ô = self.trainer.model(x, training=False)


                Ŷ.append(ỹ.numpy())
                Ŵ.append(ŵ)
                Ô.append(ô.numpy())
                
                counts += batch_size

            Ŷ = np.concatenate(Ŷ)
            Ŵ = np.concatenate(Ŵ)
            Ô = np.concatenate(Ô)
            
            features = np.array([[self.trainer.tokenizer.decoder[int(i)] for i in p] for p in X[1]])
            
            tokens.append(features)
            predictions.append(Ŷ)
            attentions.append(Ŵ)
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

def compute_performance_folds(path, label='Classifier', metric='epoch_auc', split='validation'):
    r"""This funciton retrieves the performance of the trainin or testing dataset on a given model.
    It looks at the tensorboard logs and save them in a dataframe. 

    Args:
        path (str): Directory where the model is saved [required].
        label (str): Label or name for this experiment.
        metric (str): Metric to extract from the model.
        split (str): obtain metric for train / test / validation split. Default [validation]

    Return types:
        * **dataframe** *(pandas dataframe)* - A dataframe with the metrics.
    """
    
    runs = [i for i in os.listdir(path) if 'fold-' in i]
    
    epoch_auc = []
    for fold in range(10):
        run = [i for i in runs if int(i.split('_')[0].split('-')[1]) == fold][0]

        if fold == 0:
            trainer = load_transformer(path, run, epoch=1)
        else:
            trainer = []

        evaluator = Evaluator(model=trainer, path=path, run=run, epoch=None)
        test_auc = evaluator.performance(metric=metric, split=split)
        epoch_auc.append(test_auc)

        if fold == 0:
            logger.info(
            '''{}: Heads: {}, Layers: {}, Embeddings: {}, Mode: {} '''.format(
                label,
                evaluator.trainer.num_heads,
                evaluator.trainer.num_layers,
                evaluator.trainer.embedding_size,
                evaluator.trainer.mode,
            ))

    epoch_auc = pd.concat(epoch_auc)
    epoch_auc['Model'] = label
    
    logger.info('Best Epoch (mean value): {}'.format(np.argmax(epoch_auc.groupby('epoch').mean().reset_index()[metric]) ))

    return epoch_auc