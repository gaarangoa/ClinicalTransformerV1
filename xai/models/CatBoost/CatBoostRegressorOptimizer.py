import sklearn.metrics
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import nevergrad as ng
from sklearn import metrics
import shap
import numpy as np 
import pandas as pd
import pickle
   
class CatBoostRegressorOptimizer():
    def __init__(self, dataset, fast=False):
        self.model = CatBoostRegressor()
        self.dataset = dataset 
        
        max_iterations = 1000
        if fast:
            max_iterations = 100
        
        self.parametrization = ng.p.Instrumentation(
            iterations=ng.p.Scalar(lower=50, upper=max_iterations).set_integer_casting(),
            depth=ng.p.Scalar(lower=6, upper=10).set_integer_casting(),
            learning_rate=ng.p.Log(lower=0.0001, upper=1.0),
            random_strength=ng.p.Scalar(lower=1, upper=20).set_integer_casting(),
            l2_leaf_reg=ng.p.Scalar(lower=1, upper=20).set_integer_casting(),
            bagging_temperature=ng.p.Scalar(lower=0, upper=1),
            leaf_estimation_iterations=ng.p.Scalar(lower=1, upper=10).set_integer_casting(),
        )
    
    def tunning(self, iterations: int, depth: int, learning_rate: float, random_strength: int, l2_leaf_reg: int, bagging_temperature: float, leaf_estimation_iterations: int):
        self.model = CatBoostRegressor(
            iterations = iterations,
            depth = depth,
            random_strength = random_strength,
            l2_leaf_reg = l2_leaf_reg,
            bagging_temperature = bagging_temperature,
            leaf_estimation_iterations = leaf_estimation_iterations,
            verbose = self.verbose,
            learning_rate = learning_rate
        )
        
        self.model.fit(self.dataset.X_train, self.dataset.y_train)
        
        mse_ = metrics.mean_squared_error(self.dataset.y_valid, self.model.predict(self.dataset.X_valid))
        
        return mse_
    
    def fit(self, budget=100, verbose=0):
        self.verbose = verbose
        self.optimizer = ng.optimizers.PSO(
            parametrization = self.parametrization,
            budget = budget
        )
        
        self.recommendation = self.optimizer.minimize(self.tunning).kwargs
    
        self.best_model = CatBoostRegressor(
            iterations=self.recommendation['iterations'],
            depth=self.recommendation['depth'],
            random_strength = self.recommendation['random_strength'],
            l2_leaf_reg = self.recommendation['l2_leaf_reg'],
            bagging_temperature = self.recommendation['bagging_temperature'],
            leaf_estimation_iterations = self.recommendation['leaf_estimation_iterations'],
            learning_rate = self.recommendation['learning_rate'],
            verbose=1
        )
        
        self.best_model.fit(
            pd.concat([self.dataset.X_train, self.dataset.X_valid]), 
            np.concatenate([self.dataset.y_train, self.dataset.y_valid])
        )
        mse_ = metrics.mean_squared_error(self.dataset.y_test, self.best_model.predict(self.dataset.X_test))
        
        print('best model: {}'.format(mse_))

    
    def explain(self, ): 
        self.explainer = shap.TreeExplainer(self.best_model)
        self.shap_values = self.explainer(self.dataset.X_test)
        
    def save(self, outfile): 
        pickle.dump(
            self, 
            open('{}.pk'.format(outfile), 'wb')
        )
    