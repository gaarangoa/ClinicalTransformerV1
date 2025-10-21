import tensorflow as tf  #(version 2.30)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from collections import Counter
# import kerastuner as kt
from tensorflow import keras

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test

from scipy import stats


def survival_loss_tf2(y_true, y_pred):
    """
    :param y_true: n x 2 tensor: n, (time, event)
    :param y_pred: n x 1 tensor: n, beta*X 
    :return: survival loss: -sum_{i:U=1}{beta*X_i - log{sum_{j in Omega_i}_{exp(beta*X_j)}}
    """
    sum_loss = 0.0
    for i in tf.range( len( y_pred[:,0]) ):
        omega_i = tf.math.greater_equal(y_true[:, 0], y_true[i, 0])
        sum_loss += y_true[i,1] * (y_pred[i,0] - tf.math.log(1e-20 + tf.math.reduce_sum(tf.boolean_mask(K.exp(y_pred[:,0]), omega_i))))
    norm_factor = tf.math.reduce_sum(tf.cast(tf.math.greater(y_true[:, 1], 0), K.floatx()))
    
    return sum_loss*(-1)/norm_factor  #layer norm can be adjusted. 

def cIndex_SigmoidApprox(Target, y_pred, sigma=1.0):
    """
    :param Target: n x 2 tensor: n, (time, event)
    :param y_pred: n x 1 tensor: n, beta*X (final layer output)
    :return: c-index loss:1.0-sum_{j,k} { w_{jk} * 1/(1+exp( (eta_k-eta_j)/sigma ))}
    
                     w_{jk}=delta_j*I(T_j<T_k) / sum_{j,k} {delta_j*I(T_j<T_k)}
                     delta_j = 0,1 [censored, deceased]
                     sigma =1.0, a smoothing parameter for the exponetial functions
    """
    # y_pred = y_pred

    y_true=tf.cast(Target[:,0], tf.float32)
    event=tf.cast(Target[:,1], tf.float32)
    
    n=tf.cast(len(y_pred[:,0] ) , tf.float32)
  
    #Eta_k, etaj part of the equation. 
    etaj=tf.repeat([y_pred[:,0]] ,repeats=[n], axis=0) #repeate rows (We have an issue here during training)

    etak=tf.transpose(etaj) #repeate columns

    etaMat=etak-etaj 
    
    sigmoid_eta=1.0/(1.0+tf.math.exp(etaMat/sigma))# standard exponetial sigmoid loss function  

    
    #Event indicator matrix (pairwise)
    eventI=tf.transpose( tf.repeat([event] ,repeats=[n], axis=0) )

    #Ti < Tk part of the equation 
    weightsj=tf.transpose(tf.repeat([y_true] ,repeats=[n], axis=0)) #repeate columns
    weightsk=tf.repeat([y_true] ,repeats=[n], axis=0) #repeate rows 

    rank_mat=tf.where(weightsj==weightsk, 1e-8, tf.cast(weightsj<weightsk, tf.float32) )
    rank_mat=rank_mat-tf.linalg.tensor_diag(tf.zeros([n] )+1e-8) #set diagonals to zero. 

    # matrix of comparable pairs 
    rank_mat=rank_mat*eventI

    rank_mat=rank_mat/tf.math.reduce_sum(rank_mat)

    cindex=tf.math.reduce_sum(sigmoid_eta*rank_mat) #cost/total loss 
    
    if tf.math.is_nan(cindex):
        return 1.0
    


    if cindex > 0.5:
        return 1 - cindex
    else:
        return cindex


