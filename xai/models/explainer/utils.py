import logging
import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index as lfcindex

logger = logging.getLogger('Evaluator')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chandler = logging.StreamHandler()
chandler.setLevel(logging.INFO)
chandler.setFormatter(formatter)
logger.addHandler(chandler)
logger.setLevel(logging.INFO)

from xai.models import load_transformer

def compute_epoch_performance(dataset, path, target, Evaluator, metrics, stratify, index_metric=1, epoch=10, data_iterations=1, test_size=0.2):
    r"""This funciton retrieves the performance of the testing set. It takes the raw data input 
    and splits into train / test. 

    Args:
        dataset (dataframe): dataframe with the dataset.
        path (str): directory of the experiment.
        target (str): column name in the dataset to be used as label. [time, event] for survival metrics.
        Evaluator (class): Evaluator to use
        metric (str): Function to compute the performance [auc, cindex]
        index_metric (int): From the predicted values of the model, what index to use to compute the score performance. For instance, in a binary classifier use the index 1 that corresponds to the class = 1.
        epoch (int): Epoch to analyze
        data_iterations (str): Number of times a random features are selected by patient (e.g, 10 indicates that each patient is predicte 10 times where each time is different feature order - useful for datasets where there are many features)
        test_size (float): Fraction from the full dataset used for testing (1 if you use in the entire data. Put the same value that was used for training the model to get the same partitions)
    Return types:
        * **performance** *(list)* - A list with the performance over the folds.
    """
    

    runs = [i for i in os.listdir(path) if 'fold-' in i]
    aucs = []
    metrics_val = np.zeros([len(runs), len(metrics)])

    for fold in range(len(runs)):
        run = [i for i in runs if int(i.split('_')[0].split('-')[1]) == fold][0]

        if test_size == 1:
            test_data = dataset
        else:
            train_data, test_data = train_test_split(
                dataset, 
                test_size=test_size, 
                random_state=fold,
                stratify=dataset[stratify]
            )
        
        data = test_data.reset_index(drop=True)
        
        # Load trained model - trainer
        trainer = load_transformer(path, run, epoch=epoch)
        
        # Evaluate trainer on the dataset provided
        transformed_data = trainer.data_converter.transform(data)
        evaluator = Evaluator(model=trainer, path=path, run=run)
        
        out = evaluator.predict(
            transformed_data, iterations=data_iterations, normalize=True, batch_size=10000
        )
        
        β = out[0]

        # Compute risk score
        β0 = β[:, :, index_metric].mean(axis=0)
        data['β'] = β0

        
        for mx, metric in enumerate(metrics): 
            if metric == 'auc':
                val = roc_auc_score(data[target[mx][0]], data['β'])
                metrics_val[fold, mx] = val

            if metric == 'cindex':
                val = lfcindex(data[target[mx][0]], data['β'] , data[target[mx][1]])
                metrics_val[fold, mx] = val
        
        str_perf = ["{}:{:.2f}".format(k, v) for k, v in zip(metrics, metrics_val[fold, :])]

        logger.debug("{}, {}".format(run, str_perf))
    
    res = pd.DataFrame(metrics_val, columns=metrics)
    res['run'] = runs

    return res


def compute_epoch_performance_random_snapshot_from_data(dataset, path, target, Evaluator, metrics, stratify, index_metric=1, epoch=10, data_iterations=1, nan_rand_fraction=0.1, train_fraction=0.2, feat_rand_fraction=0.8, test_size=0.2, split='train', seed=0, verbose=0):
    r"""This funciton retrieves the performance of the testing set. It takes the raw data input 
    and splits into train / test. 

    Args:
        dataset (dataframe): dataframe with the dataset.
        path (str): directory of the experiment.
        target (str): column name in the dataset to be used as label. [time, event] for survival metrics.
        Evaluator (class): Evaluator to use
        metric (str): Function to compute the performance [auc, cindex]
        index_metric (int): From the predicted values of the model, what index to use to compute the score performance. For instance, in a binary classifier use the index 1 that corresponds to the class = 1.
        epoch (int): Epoch to analyze
        data_iterations (str): Number of times a random features are selected by patient (e.g, 10 indicates that each patient is predicte 10 times where each time is different feature order - useful for datasets where there are many features)
        train_fraction (float): Fraction of the training data to use for evaluating the model
        nan_rand_fraction (float): select randomly this fraction of patients from the train_fraction dataset
        feat_rand_fraction (float): Fraction of features to randomize
        split (str): Use train or test split to generate the samples
        test_size (float): Fraction from the full dataset used for testing (1 if you use in the entire data. Put the same value that was used for training the model to get the same partitions)
        seed (int): Random seed
        verbose (int): verbose 0 none, 1 print track
    Return types:
        * **performance** *(list)* - A list with the performance over the folds.
    """
    
    np.random.seed(seed)

    runs = [i for i in os.listdir(path) if 'fold-' in i]
    aucs = []
    metrics_val = np.zeros([len(runs), len(metrics)])

    for fold in range(len(runs)):
        run = [i for i in runs if int(i.split('_')[0].split('-')[1]) == fold][0]

        if test_size == 1:
            test_data = dataset
        else:
            train_data, test_data = train_test_split(
                dataset, 
                test_size=test_size, 
                random_state=fold,
                stratify=dataset[stratify]
            )
        
        if split == 'train':
            data = train_data.sample(frac=train_fraction).reset_index(drop=True)
        else: 
            data = test_data.sample(frac=train_fraction).reset_index(drop=True)
            data = pd.concat([data, data, data]).reset_index(drop=True) # make 3 copies of the same data
        
        if verbose == 1:
            logger.info('split: {}\tepoch: {}\tfold: {}\tvalidation data: {} of {} testing data'.format(split, epoch, fold, data.shape[0] / 3, test_data.shape[0]))
        # randomly shuffle values in all columns to generate fake samples
        indexi = np.int(nan_rand_fraction * data.shape[0])
        datar = data[ :indexi].copy()
        datac = data[indexi: ].copy()

        non_rand_feat = []
        for i in list(datar):
            val = np.array(datar[i].fillna('-'))
            if not len(val) == 1:
                non_rand_feat.append(i)

        np.random.shuffle(non_rand_feat)
                
        for i in non_rand_feat[:np.int(feat_rand_fraction*len(non_rand_feat))]:
            if i in [target]:
                continue
            val = np.array(datar[i].fillna('-'))
            val2 = np.array(datar[i])
            
            if not len(val) == 1:
                np.random.shuffle(val2)
                datar[i] = val2
        
        data = pd.concat([datar, datac]).reset_index(drop=True)

        # Load trained model - trainer
        trainer = load_transformer(path, run, epoch=epoch)
        
        # Evaluate trainer on the dataset provided
        transformed_data = trainer.data_converter.transform(data)
        evaluator = Evaluator(model=trainer, path=path, run=run)
        
        out = evaluator.predict(
            transformed_data, iterations=data_iterations, normalize=True, batch_size=10000
        )
        
        β = out[0]

        # Compute risk score
        β0 = β[:, :, index_metric].mean(axis=0)
        data['β'] = β0

        
        for mx, metric in enumerate(metrics): 
            if metric == 'auc':
                val = roc_auc_score(data[target[mx][0]], data['β'])
                metrics_val[fold, mx] = val

            if metric == 'cindex':
                val = lfcindex(data[target[mx][0]], data['β'] , data[target[mx][1]])
                metrics_val[fold, mx] = val
        
        str_perf = ["{}:{:.2f}".format(k, v) for k, v in zip(metrics, metrics_val[fold, :])]

        logger.debug("{}, {}".format(run, str_perf))
    
    res = pd.DataFrame(metrics_val, columns=metrics)
    res['run'] = runs

    return res

def compute_performance_folds(path, Evaluator, label='Classifier', metric='epoch_auc', split='validation', **kwargs):
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
    for fold in range(len(runs)):
        logger.debug('epoc {}'.format(fold))
        run = [i for i in runs if int(i.split('_')[0].split('-')[1]) == fold][0]

        if fold == 0:
            # this is a place holder, we are just looking into the results logged to tensorflow
            trainer = load_transformer(path, run, epoch=kwargs.get('epoch', 1))
        else:
            trainer = []

        evaluator = Evaluator(model=trainer, path=path, run=run)
        test_auc = evaluator.performance(metric=metric, split=split)
        epoch_auc.append(test_auc)

        if fold == 0:
            info = '''{}, Runs: {}, Heads: {}, Layers: {}, Embeddings: {}, Mode: {}, LR: {}, TestSplit: {}, Features: {}({}%tile)'''.format(
                label,
                len(runs),
                evaluator.trainer.num_heads,
                evaluator.trainer.num_layers,
                evaluator.trainer.embedding_size,
                evaluator.trainer.mode,
                evaluator.trainer.learning_rate,
                evaluator.trainer.test_size,
                evaluator.trainer.max_features,
                evaluator.trainer.max_features_percentile,
            )

    epoch_auc = pd.concat(epoch_auc)
    epoch_auc['Model'] = label
    
    logger.info('{}, Best Epoch (mean value): {}'.format(info, np.argmax(epoch_auc[['epoch', metric]].groupby('epoch').mean().reset_index()[metric]) ))

    return epoch_auc