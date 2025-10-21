from xai.models import load_transformer as load_model
from .SurvivalEvaluator import Evaluator as TransformerSurvivalEvaluator
from .SurvivalExplainer import survival_attention_scores

from tqdm.auto import tqdm
import umap
import pandas as pd
import numpy as np
from .utils import Output
import os

class Extractor():
    def __init__(self, **kwargs):
        self.time = kwargs['time']
        self.event = kwargs['event']
        self.sample_id = kwargs['sample_id']
        # self.runs = kwargs['runs']
        self.path = kwargs['path']
        self.epoch = kwargs['epoch']
        self.fold = kwargs['fold']

        self.runs = [i for i in os.listdir(self.path) if 'fold-' in i]
        self.run = [i for i in self.runs if int(i.split('_')[0].split('-')[1]) == self.fold][0]
        
        # Load trained model - trainer
        self.trainer = load_model(self.path, self.run, epoch=self.epoch)
        self.evaluator = TransformerSurvivalEvaluator(model=self.trainer)

        self.embeddings__ = ["E{}".format(i) for i in range(self.trainer.embedding_size)]
    
    def perturb_one_feature(self, data, var, pre):
        '''
        Perturb one feature in the dataset and return the perturbed versions for each sample id
        '''
        datas = []
        iterations = len(pre[var])
        for ptid in data[self.sample_id].values:

            ptdata = data[data[self.sample_id] == ptid].copy()

            data_ = pd.concat([ptdata for _ in range(iterations)]).reset_index(drop=True)
            data_[self.sample_id] = ["{}_{}".format(i, ix) for ix,i in enumerate(data_[self.sample_id])]

            data_[var] = pre[var]

            data_['id__'] = ptid
            datas.append(data_)

        return pd.concat(datas).reset_index(drop=True).copy()       
    
    def get_random_value_from_data(self, train_data, feature):
        value = train_data[feature].values
        np.random.shuffle(value)

        return value[0]
    
    def perturb_feature_by_random_sampling(self, data, data_to_sample, variables, sample_size=100):
        '''
        Perturb features by using a random sampling and by looking at the values in the data_to_sample dataset. Basically, it will create up to sample_size
        where each feature will be replaced it by a random selected value from the data_to_sample data. 
        '''
        
        datas = []
        for ptid in data[self.sample_id].values:

            ptdata = data[data[self.sample_id] == ptid].copy()
            
            data_ = pd.concat([ptdata for _ in range(sample_size)]).reset_index(drop=True)
            data_[self.sample_id] = ["{}_{}".format(i, ix) for ix,i in enumerate(data_[self.sample_id])]
            
            for var in variables:
                data_[var] = [self.get_random_value_from_data(data_to_sample, var) for i in range(sample_size)]

            data_['id__'] = ptid
            datas.append(data_)

        return pd.concat(datas).reset_index(drop=True).copy()  
    
    def scores(self, data, iterations, batch_size = 10000000):
        '''
        Compute survival scores.
        '''

        transformed_data = self.trainer.data_converter.transform(data)
        β, w, o, t = self.evaluator.predict(
            transformed_data, iterations=iterations, normalize=True, batch_size=batch_size
        )

        # Compute risk score
        β = β.mean(axis=0)
        data['β'] = β[:, 0]

        return data

    def embeddings(self, data, iterations, batch_size=10000000):
        '''
        Extract the embeddings from the <cls> token, averaged using the iterations and returns all ouptuts including attention scores for all tokens in the results object. 
        
        results: 
        results.data: Input data
        results.outputs: Output embeddings for all patients and tokens
        results.features: FEatures for all patients and tokens
        results.patient_ids: patient ids for all patients and tokens
        results.iters: vector with id for iterations
        results.attention_scores: matrix with patients tokens and attentions
        
        results.embs: dataframe with the original data and the embeddings for the <cls> feature token
        
        '''
        transformed_data = self.trainer.data_converter.transform(data)
        results = Output()
        results.data, results.outputs, results.features, results.patient_ids, results.iters, results.attention_scores = survival_attention_scores(transformed_data, self.evaluator, iterations=iterations, batch_size=batch_size, sample_id=self.sample_id)
        
        
        betas = []
        embs = []
        for ptid in results.data[self.sample_id]:
            emb = results.outputs[(results.patient_ids == ptid) & (results.features == '<cls>')]

            emb = np.mean(emb, axis=0).reshape(1, self.trainer.embedding_size)
            
            # Wo = self.trainer.model.layers[1].weights[0].numpy()
            # beta = np.sum((emb.T*Wo))

            emb = pd.DataFrame(emb, columns=self.embeddings__)
            emb[self.sample_id] = ptid
            # emb['beta'] = beta
            emb['β'] = results.data[results.data[self.sample_id] == ptid].β.values[0]

            embs.append(emb)

        embs = pd.concat(embs).reset_index(drop=True)
        results.edata = pd.merge(data, embs, on=self.sample_id)
        results.embeddings__ = self.embeddings__

        return results
    
    def beta_weights(self):
        Wo = self.trainer.model.layers[1].weights[0].numpy()
        return Wo
        