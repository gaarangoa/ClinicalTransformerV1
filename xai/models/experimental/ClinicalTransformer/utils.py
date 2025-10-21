#!/usr/bin/env python
# coding: utf-8

# ### A Tranformer for numerical data (UniFormer).
# 
# The model is implemented with custom loss functions:
# - Concordance index 
# - Cox partial likelihood 
# 
# Performance metric: concordance index (approx)
# 
# 
# 


# In[ ]:


import tensorflow as tf  #(version 2.30)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from collections import Counter
import kerastuner as kt
from tensorflow import keras

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import train_test_split
from scipy import stats

import seaborn as sns
import networkx as nx

    
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# # Model

# ## Scaled dot product attention

# In[225]:


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


# Let's assume that we have a numerical input X that will be projected on k, q, and v

# ## Embedding a numerical vector

# In[226]:


class NumericalEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size=64):
        super(NumericalEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.y = tf.keras.layers.Dense(embedding_size)
        
    def call(self, x):
        return self.y(x)


# ## Multihead attention

# In[227]:


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


# ## Encoder

# ### Encoder Layer

# In[228]:


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


# In[229]:


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.3):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    #self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    #self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):
    
    attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x+attn_output)  # (batch_size, input_seq_len, d_model), 
    #out1 = self.layernorm1(attn_output) 
    
    #ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    #fn_output = self.dropout2(ffn_output, training=training)
    #out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out1, attn_weights   #return out2, attn_weights


# ### Encoder

# In[230]:


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.3):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = NumericalEmbeddingLayer(embedding_size = d_model)
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    x = tf.transpose(x, perm=[0, 2, 1])
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x = self.dropout(x, training=training)
    
    ws = []
    for i in range(self.num_layers):
        x, w = self.enc_layers[i](x, training, mask)
        ws.append(w)
    
    return x, ws  # (batch_size, input_seq_len, d_model)


# ## Transformer

# In[231]:


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, rate=0.3):

        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.final_layer= tf.keras.layers.Dense(num_classes)
        
        self.flat = tf.keras.layers.Flatten() #flatten encoder output before feeding to fcnn? 
    
    def call(self, inp, training):

        enc_output, attention_weights = self.encoder(inp, training=training, mask=None)  # (batch_size, inp_seq_len, d_model)
        final_output= self.final_layer(self.flat(enc_output) )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights, enc_output
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data  

        with tf.GradientTape() as tape:
            y_pred, _, _ = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# - Partial likelihood cox loss function
    
# In[233]:

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


# - Concordance Index Loss

# In[234]:
#@tf.function
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

    loss=tf.math.reduce_sum(sigmoid_eta*rank_mat) #cost/total loss 
    return 1 - loss


# In[235]:


#define concordance index for sigmoid approximation 

def sigmoid_concordance(Target, y_pred):
    
    """
    :param Target: n x 2 tensor: n, (time, event)
    :param y_pred: n x 1 tensor: n, beta*X (final layer output)
    :return: c-index : sum_{j,k}{delta_j I(T_j<T_k)I(eta_j>eta_k)} / sum_{j,k} {delta_j*I(T_j<T_k)} 
    """
        
    y_true=tf.cast(Target[:,0], tf.float32)
    event=tf.cast(Target[:,1], tf.float32)
    
    n=tf.cast(len(y_pred[:,0] ) , tf.float32)
   
    etaj=tf.repeat([y_pred[:,0]] ,repeats=[n], axis=0)

    etak=tf.transpose(etaj) #repeate columns
    etaMat=tf.cast(etaj>etak, tf.float32)
    
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
    metric=tf.math.reduce_sum(etaMat*rank_mat) #metric
    return metric



#Define metric function (C-index) for cox model

def cox_concordance(Target, y_pred):
    
    y_true=tf.cast(Target[:,0], tf.float32)
    event=tf.cast(Target[:,1], tf.float32)
    
    n=tf.cast(y_pred.shape[0] , tf.float32)
   
    etaj=tf.repeat([y_pred[:,0]] ,repeats=[n], axis=0)

    etak=tf.transpose(etaj) #repeate columns
    etaMat=tf.cast(etak>etaj, tf.float32)
    
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
    metric=tf.math.reduce_sum(etaMat*rank_mat) #metric
    return metric


#CHECK FOR OVERFITTING BY PLOTING TRAINING AND VALIDATION LOSS
def loss_function_plots(trained_model, title="add your title"):
    plt.plot(trained_model.history.history["loss"], label="training loss")
    
    plt.plot(trained_model.history.history["val_loss"], color="purple", label="validation loss")
    
    epochs=np.arange(len(trained_model.history.history["loss"]))
    best_epoch=epochs[trained_model.history.history["val_loss"]==np.min(trained_model.history.history["val_loss"]) ]
    plt.axvline(best_epoch, label="Optimal epoch = " + str(best_epoch), color="green" ) 
    
    plt.xlabel("Epochs")
    plt.ylim(0,1)
    title=plt.title(title)
    plt.legend()
    
def top_variables(w, r,k, features):
    
    """
    : w is the list of weights from the uniformer 
    : r is the number of top variables per patient
    : k is the number of global top variables  among all patients 
    : features, a list of features
   """
    agg_att=0
    for i in range(len(w)):
        agg_att+=w[i][:, :, :, : ]/len(w)  #average across layers 


    att_vec=[]

    for i in range( agg_att.shape[0] ):
        att_vec.append( np.mean( np.mean( agg_att[i, :, :, : ], axis=0), axis=0))  #average across heads then per patient 

    atten_avg_l0=pd.DataFrame(att_vec, columns=features)


    top10list=[]
    for i in range( agg_att.shape[0] ):
        top10list +=list( atten_avg_l0.iloc[i, :].sort_values(ascending=False)[:r].index)  #top r variables per patient 


    top10freq=pd.DataFrame( list( zip( list(Counter(top10list).keys()), list(Counter(top10list).values()) )), columns=["Variables", "Freq"])

    top10freq=top10freq.sort_values(by="Freq", ascending=False).reset_index(drop=True)
    
    return( top10freq.iloc[:k] , atten_avg_l0) #global top k variables


# calculate logrank test 
def best_quantile_cutoff(time, event, y_pred, fixed_quant=True):
    """
    :param time: n x 1 array
    :param event: n x 1 array of ones and zeros
    :param y_pred: n x 1 array: n, beta*X (final layer output)
    """
    
    p_values=[]
    risk_value=[]
    hazard_ratios=[]
    
    if fixed_quant:
        quantiles=[0.50] #list of quantiles to search from 
        for ix, c in enumerate(quantiles):

            risk_level = (y_pred[:,0]<np.quantile(y_pred[:,0], c)) 

            group = [1 if i==True else 0 for i in risk_level]    #define group according to risk scores 

            df=pd.DataFrame({ #put the data into a form ready for logrank test
                'durations':time.tolist(), 
                'groups':group,
                'events':event.tolist()
            })


            results = multivariate_logrank_test(df['durations'], df['groups'], df['events']) #log rank test based on quantile cut-off. 
            p_values.append( np.round(results.p_value, 5) )
            risk_value.append(np.quantile(y_pred[:,0], c))                                                                           


            #record hazard ratio at each quantile. 
            mydata=pd.DataFrame(list( zip(time, event, np.multiply(y_pred[:,0]>np.quantile(y_pred[:,0],c), 1) )), columns=["time", "event", "Variable_From_Risk_Score"]
                               )

            cph = CoxPHFitter()
            cph.fit(mydata, duration_col='time', event_col='event')
            hazard_ratios.append( [ cph.hazard_ratios_[0], c])     
    else:
        quantiles=np.round( np.arange(0.25, 0.80, 0.05), 2) #list of quantiles to search from 
        for ix, c in enumerate(quantiles):

            risk_level = (y_pred[:,0]<np.quantile(y_pred[:,0], c)) 

            group = [1 if i==True else 0 for i in risk_level]    #define group according to risk scores 

            df=pd.DataFrame({ #put the data into a form ready for logrank test
                'durations':time.tolist(), 
                'groups':group,
                'events':event.tolist()
            })


            results = multivariate_logrank_test(df['durations'], df['groups'], df['events']) #log rank test based on quantile cut-off. 
            p_values.append( np.round(results.p_value, 5) )
            risk_value.append(np.quantile(y_pred[:,0], c))                                                                           


            #record hazard ratio at each quantile. 
            mydata=pd.DataFrame(list( zip(time, event, np.multiply(y_pred[:,0]>np.quantile(y_pred[:,0],c), 1) )), columns=["time", "event", "Variable_From_Risk_Score"]
                               )

            cph = CoxPHFitter()
            cph.fit(mydata, duration_col='time', event_col='event')
            hazard_ratios.append( [ cph.hazard_ratios_[0], c])
       
    logrank_results=pd.DataFrame(  list( zip(list(quantiles),  p_values, risk_value)), columns=["Quantiles", "Pvalues", "Risk_Value"]
                                ).sort_values(by="Pvalues", ascending=True)
    return( logrank_results.iloc[0,:],hazard_ratios)


#Kaplan Meier Plot based on best quantile cut-off

def KMPlot(time, event, y_pred, title="add your title", fixed_quant=True):
    """
    :param time: n x 1 array
    :param event: n x 1 array of ones and zeros
    :param y_pred: n x 1 array: n, beta*X (final layer output)
    """
    
    kmf= KaplanMeierFitter()
    
    f, axs = plt.subplots(1, 1, figsize=(7, 5))
    
    best_quant, _=best_quantile_cutoff(time, event, y_pred, fixed_quant=fixed_quant)

    risk_level = (y_pred[:,0]<=np.quantile(y_pred[:,0], best_quant[0] ))
    kmf.fit(time[~risk_level], event_observed=event[~risk_level], label= "(" +str( len(risk_level)-np.sum(risk_level) )+ ") " +"Low Risk")
    kmf.plot(ax=axs, ci_show=False, show_censors=True)

    kmf.fit(time[risk_level], event_observed=event[risk_level], label="(" + str(np.sum(risk_level)) +") " +"High Risk")
    kmf.plot(ax=axs, ci_show=False, show_censors=True)
    #axs.set_title('Quantile {}'.format( best_quant[0]))
    axs.set_xlabel("Time[Months]", fontweight="bold")
    axs.set_ylabel("Survival Probabilities", fontweight="bold")
    axs.set_title(title, fontweight="bold")
    plt.ylim(0, 1.001)
    right_side = axs.spines["right"]
    right_side.set_visible(False)

#     left_side = axs.spines["left"]
#     left_side.set_visible(False)

    top_side = axs.spines["top"]
    top_side.set_visible(False)
    
    #add_at_risk_counts(kmf_nonres, kmf_res, ax=axs)
    f.text(0.31, 0.20, "Quantile = "+ str(best_quant[0] ),  
             fontsize = 12, color ='black', 
             ha ='right', va ='bottom',  
             alpha = 1.0)

    f.text(0.35, 0.15, "p-value = "+ str(best_quant[1] ),  
             fontsize = 12, color ='black', 
             ha ='right', va ='bottom',  
             alpha = 1.0)

#aggregating attention matrices. 
def aggregate_attention(w, y_pred, best_quant, features):
    
    """
    : w, a list of attention weight matrices of dimension PxP from UniFormer. P is the number of features 
    : y_pred : n x 1 tensor, the final output of the UniFormer 
    : best_quant, "optimal" quantile cutoff and corresponding p-value. (quantile, p_value)
    : features, a list of names of the variables used as input. 
    
    Output
        :Naive weighting of attention matrices in the whole test sample. 
    
        :2 Weighted attention Matrices of PxP subset on grouping by y_pred and KM curves.
 
    """

    agg_att=0
    for i in range(len(w)):
        agg_att+=w[i][:, :, :, : ]/len(w)  #average across layers 
        
    Weighted_attent_layer_heads=np.mean( agg_att[:,:, :, :], axis=1) #average across heads 
    
    
    #weighted attention among all patients
    Weighted_Attent_full= pd.DataFrame(
        np.mean( np.mean( agg_att[:,:, :, :], axis=1) , axis=0),
        columns=features, index=features ) #average across heads and then across patients 

    risk_level = (y_pred[:,0]<np.quantile(y_pred[:,0], best_quant[0]))

    #weighted attentions for patients with low risk
    Weighted_Attent_weights_low_risk=pd.DataFrame(
        np.mean( np.mean( agg_att[~risk_level,:, :, :], axis=1) , axis=0),
        columns=features, index=features )


    #weighted attentions for patients with high risk 
    Weighted_Attent_weights_high_risk=pd.DataFrame(
        np.mean( np.mean( agg_att[risk_level,:, :, :], axis=1) , axis=0),
        columns=features, index=features)
    
    return (Weighted_attent_layer_heads, Weighted_Attent_full, Weighted_Attent_weights_low_risk,Weighted_Attent_weights_high_risk )



def network_viz_att(weights,features, raw_median_scores, title="Add your title",figsize=(10,10), thr=0.01, bokeh_option=False, node_scaler=1, text_font_size='10px', plot_width=500, plot_height=500):

    if bokeh_option==False:
        
        """
        : weights - attention weight matrices from aggregation function. 
        : features - list of features used in training model. 
        : thr  - attention weight threshold. A numerical value set such that if attention weight > thr, the node is added to the network.  
        """

        cm = sns.light_palette("#4a258a", as_cmap=True)

        f, ax= plt.subplots(1, 1, figsize=figsize)

        attention_weights = pd.DataFrame(np.array(weights, dtype=float), columns=features, index=features)

        G = nx.Graph()

        for ix,i in attention_weights.iterrows():
            edges = [ [ix,w[1],w[0], cm(1*w[0]) ] for w in zip(list(i[i>0]), list(i[i>0].index))] #defining edges and nodes
            for i,j,k,l in edges:
                if k >= thr:
                    G.add_edge(i, j, color=l, weight=k) #the length of the edge refers to the magnitude of the attention weight 
                    #between node i and node j. 


        width =np.array(list(nx.get_edge_attributes(G,'weight').values() )) #weights as width. 

        #scale edge width for better visualizetion. 
        if np.max(width)*100 <10: 
            edge_scaler=50
        else:
            edge_scaler=5
        width=width*edge_scaler 

        #good graph choices: kamada_kawai, maybe spring     
        #add node size 
        mydegree=dict(G.degree())
        node_s=np.array( list(mydegree.values()))#scaled by a factor of K 

        #scale nodesize for better visualization. 
        if np.max(node_s)<=10:
            node_scaler=100
        elif (np.max(node_s)>10 ) & (np.max(node_s)<=25):
            node_scaler=50
        elif (np.max(node_s)>25 ) & (np.max(node_s)<=55):
            node_scaler=30
        elif (np.max(node_s)>55 ) & (np.max(node_s)<=100):
            node_scaler=15
        elif (np.max(node_s)>100 ) & (np.max(node_s)<=500):
            node_scaler=3
        else:
            node_scaler=1.0
        node_s=node_s*node_scaler

        nx.draw_kamada_kawai(G, with_labels=True, ax=ax, width=np.array(width), node_size=node_s, alpha=1.0,
                     font_size=9, font_weight="normal", 
                             node_color="skyblue", 
                            edge_color="lightgrey")
        ax.set_title(title)
    else:

        # BOKEH OPTION 
        try:
            from bokeh.io import output_notebook, show, save
            from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine
            from bokeh.plotting import figure
            from bokeh.plotting import from_networkx
            from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8, Viridis256, Category20
            from bokeh.transform import linear_cmap
            from networkx.algorithms import community
            from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet
            
            from colour import Color
            
        except ImportError:
            print ("No module named bokeh")

        output_notebook() #to fascilitate interactive session with the network. 

        cm = sns.light_palette("#4a258a", as_cmap=True)

        attention_weights = pd.DataFrame(np.array(weights, dtype=float), columns=features, index=features)

        G = nx.Graph()

        for ix,i in attention_weights.iterrows():
            edges = [ [ix,w[1],w[0], cm(1*w[0]) ] for w in zip(list(i[i>0]), list(i[i>0].index))] #defining edges and nodes
            for i,j,k,l in edges:
                if k >= thr:
                    G.add_edge(i, j, color=l, weight=k) #the length of the edge refers to the magnitude of the attention weight 
                    #between node i and node j. 

        #Add node size

        degrees = dict(nx.degree(G))
        nx.set_node_attributes(G, name='degree', values=degrees)
      
        node_s=np.array( list(degrees.values()))#scaled by a factor of K 

        node_s=node_s*node_scaler
        
        nodes=np.array( list(degrees.keys()))

        adjusted_node_size = dict(zip(nodes, node_s))
        
        nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)

        #build communities
        communities = community.greedy_modularity_communities(G)

        #Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
        if len(communities)<8:
            color_palette = Blues8
        else:
            color_palette = Viridis256

        # Create empty dictionaries
        modularity_class = {}
        modularity_color = {}
        #Loop through each community in the network
        for community_number, community in enumerate(communities):
            #For each member of the community, add their community number and a distinct color
            for name in community: 
                modularity_class[name] = community_number
                modularity_color[name] = color_palette[4]
        nx.set_node_attributes(G, modularity_class, 'modularity_class')
        nx.set_node_attributes(G, modularity_color, 'modularity_color')

        # Add color to nodes
        red = Color("blue")
        colors = list(red.range_to(Color("red"),10))
        
        node_colors_by_rawvalue = {}
        node_median_score = {}
        for ix,i in pd.DataFrame(raw_median_scores).iterrows():
            node_colors_by_rawvalue.update({i.name: str(colors[int(10*i[0])]) })
            node_median_score.update( {i.name: i[0]} )
        
        nx.set_node_attributes(G, node_colors_by_rawvalue, 'node_color')
        nx.set_node_attributes(G, node_median_score, 'median_score')
        
        #Choose colors for node and edge highlighting
        node_highlight_color = 'white'
        edge_highlight_color = 'black'

        #Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
        size_by_this_attribute = 'adjusted_node_size'
        color_by_this_attribute = 'node_color'



        #Choose a title!
        title = title

        #Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
               ("Character", "@index"),
                ("Degree", "@degree"),
            ("median_score", "@median_score"),
                 #("Modularity Class", "@modularity_class"),
                #("Modularity Color", "$color[swatch]:modularity_color"),
        ]

        #Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips = HOVER_TOOLTIPS,
                      tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                      x_range=Range1d(-30.1, 30.1), y_range=Range1d(-30.1, 30.1),
                      plot_width=plot_width, plot_height=plot_height, title=title)

        #Create a network graph object
        # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
        network_graph = from_networkx(G, nx.spring_layout, scale=20, center=(0, 0))

        #Set node sizes and colors according to node degree (color as category from attribute)
        network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)
        #Set node highlight colors
        network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
        network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)

        #Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.3, line_width=1)
        #Set edge highlight colors
        network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
        network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)

            #Highlight nodes and edges
        network_graph.selection_policy = NodesAndLinkedEdges()
        network_graph.inspection_policy = NodesAndLinkedEdges()

        plot.renderers.append(network_graph)

        #Add Labels
        x, y = zip(*network_graph.layout_provider.graph_layout.values())
        node_labels = list(G.nodes())
        source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
        labels = LabelSet(x='x', y='y', text='name', source=source, text_font_size=text_font_size, text_font_style="bold")
        plot.renderers.append(labels)
        plot.xgrid.visible = False
        plot.ygrid.visible = False

        show(plot)


# Examine top variables by risk scores grouping. 
def top_variables_plot(w, y_pred, best_quant, features, p): 
    """
    :w - list of attention weight matrices from UniFomer model
    :p - The number of top variable to display 
    
    Output
    : barplots showing variables ordered by magnitude of attention weight. 
    """
    
    _, W_full,W_low,W_high=aggregate_attention(w, y_pred, best_quant, features) #three matrices of attention weights
    
    #using full sample
    top_markers_full=W_full.mean(axis=0).sort_values(ascending=False)[:p] 
    top_markers_full=pd.DataFrame(top_markers_full).reset_index()
    
    #using group=low risk (high survival)
    top_markers_L=W_low.mean(axis=0).sort_values(ascending=False)[:p]
    top_markers_L=pd.DataFrame(top_markers_L).reset_index()
    

    #using group=high risk (low survival)
    top_markers_H=W_high.mean(axis=0).sort_values(ascending=False)[:p]
    top_markers_H=pd.DataFrame(top_markers_H).reset_index()

    data=[top_markers_full, top_markers_L, top_markers_H ]
    level=["All", "Higher Survival", "Lower Survival"]
    
    f, axs=plt.subplots(len(data),1, figsize=(7,18))
    
    for i in np.arange(3):
        sns.barplot(y="index",x=0, data=data[i], ax=axs[i])
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
        
        axs[i].set_title("Top "+str(p)+" variables for "+ level[i] +" Patients")
        
        axs[i].set_xlabel("Attention Scores")
        
## Heatmaps for each layer by head
def attention_heatmaps(w, features):
    """
    w: List of attention weights from UniFormer Model 
     
    Output
     
    : Heatmaps for each head per layer
    """


    if (len(w)>1) & (w[0].shape[1]>1) :
        f, ax = plt.subplots(len(w),w[0].shape[1], figsize=(6*len(w), 7*w[0].shape[1]))
        for l in np.arange(len(w)):
            for h in np.arange( w[0].shape[1] ): 
                temp=pd.DataFrame( np.mean( w[l][:, h, :,:], axis=0), columns=features, index=features)
                _=sns.heatmap(temp, cmap="YlGnBu", ax=ax[l, h])
                ax[l, h].set_title('Layer {} Head {}'.format(l,h))

                ax[l, h].set(xticklabels=[])
                ax[l, h].set(xlabel=None)
                if h>0:
                    ax[l, h].set(yticklabels=[])
                    ax[l,h].set(ylabel=None)
    elif (len(w)>1) & (w[0].shape[1]==1):
        f, ax = plt.subplots(len(w),w[0].shape[1], figsize=(7*w[0].shape[1], 3*len(w) ))
        for l in np.arange(len(w)):
            for h in np.arange( w[0].shape[1] ):
                temp=pd.DataFrame( np.mean( w[l][:, h, :,:], axis=0), columns=features, index=features)
                _=sns.heatmap(temp, cmap="YlGnBu", ax=ax[l])
                ax[l].set_title('Layer {} Head {}'.format(l,h))
                ax[l].set(xticklabels=[])
                ax[l].set(xlabel=None)
                
    elif (len(w)==1) & (w[0].shape[1]>1):
        f, ax = plt.subplots(w[0].shape[1],len(w), figsize=(7*len(w), 3*w[0].shape[1] ))
        for l in np.arange(len(w)):
            for h in np.arange( w[0].shape[1] ):
                temp=pd.DataFrame( np.mean( w[l][:, h, :,:], axis=0), columns=features, index=features)
                _=sns.heatmap(temp, cmap="YlGnBu", ax=ax[h])
                ax[h].set_title('Layer {} Head {}'.format(l,h))
                ax[h].set(xticklabels=[])
                ax[h].set(xlabel=None)
                
    else:
        f, ax = plt.subplots(len(w),w[0].shape[1], figsize=(7*len(w), 7*w[0].shape[1]))
        for l in np.arange(len(w)):
            for h in np.arange( w[0].shape[1] ):
                temp=pd.DataFrame( np.mean( w[l][:, h, :,:], axis=0), columns=features, index=features)
                _=sns.heatmap(temp, cmap="YlGnBu", ax=ax)
                ax.set_title('Layer {} Head {}'.format(l,h))

                ax.set(xticklabels=[])
                ax.set(xlabel=None)
                
                
def t_test_for_directionality(input_x, test_index, y_pred, best_quant, sig_level=1, show_plot=False):
    
    """
    : input_x : n x p - input variables as panda dataframe, all continuous 
    : test_index : n , test index as generated during train/test split
    : y_pred : n x 1 array, risk scores from the model 
    : best_quant, "optimal" quantile and accompanying p-value. (quantile, p-value)
    
    : sig_level - significance level of p%. Default =1 
    """
    
    #t test for low vs high risk patients
    test_predictors=input_x.loc[test_index]  #the feature matrix is not standardized. 


    t_test_p_values=[]
    risk_level=(y_pred[:,0]<np.quantile(y_pred[:,0], best_quant[0]))

    for ix, v in enumerate(test_predictors.columns):

        t1=test_predictors[~risk_level].iloc[:,ix]  #patients with low risk/higher survival

        t2=test_predictors[risk_level].iloc[:,ix]   #patients with high risk/lower survival

        t_test_p_values.append( round( stats.ttest_ind(t1, t2, equal_var = False).pvalue, 5) ) #compute p-value t-test.

    t_test_p_values=pd.DataFrame( t_test_p_values,  index=test_predictors.columns)
    
    #concatenate the means and p-values
    high_low_means=pd.concat( [test_predictors[~risk_level].mean(axis=0), test_predictors[risk_level].mean(axis=0)], axis=1)
    high_low_means=high_low_means.rename(columns={0:"Higher Survival", 1:"Lower Survival"})

    high_low_means=pd.concat([high_low_means, t_test_p_values], axis=1)

    high_low_means=high_low_means.rename(columns={0:"Pvalue"})
    level=["Low", "High"]

    # Extract significantly different variables, t-test, two-side, sig-level=sig_level
    high_low_means=high_low_means[ high_low_means["Pvalue"]<sig_level].sort_values('Pvalue',ascending=True)
    high_low_means=high_low_means.reset_index( drop=False)
    
    if show_plot:
        f, axs=plt.subplots(1,1, figsize=(7,7))
        sns.barplot(y="index", x="Pvalue", data=high_low_means, ax=axs)
        axs.set_ylabel("Features",fontsize=12)

    return (high_low_means)


### SHAP-LIKE BEESWARM PLOT 
def custom_beeswarmplot(w, y_pred, y_test, panda_x_test, e_test, fig_size=(7,7), max_display=None, title="add your title", fixed_quant=True, fontsize=10): 
    
    try:
        import shap
        
    except ImportError: 
        print ("No module named shap")
    
    try:
        import matplotlib as mpl
    except ImportError: 
        print("No module named matplotlib")
    
    assert isinstance(panda_x_test, pd.DataFrame)==True
    
    features=list(panda_x_test)
    if max_display==None:
        if len(features)<10:
            max_display=len(features)
        else:
            max_display=10
        
    
  
#     top_vars, pseudo_shap=top_variables(w, 3, 25, features) #abtain aggregated attention matrices 
    best_quant, HR=best_quantile_cutoff(y_test, e_test, y_pred, fixed_quant=fixed_quant) #exctract best_quantile cut-off for grouping
    
    pseudo_shap_3d,_,_,_=aggregate_attention(w, y_pred, best_quant, features) #extract attention matrices
    
    pseudo_shap=np.empty( (pseudo_shap_3d.shape[0], pseudo_shap_3d.shape[1]))
    
    for i in np.arange(pseudo_shap_3d.shape[0]):
        pseudo_shap[i, :]=np.diag(pseudo_shap_3d [i , :, : ])
    
    pseudo_shap=pd.DataFrame(pseudo_shap, columns=features)
    
    feature_ordered=list(np.mean(pseudo_shap, axis=0)[
        np.argsort(-np.mean( np.array(pseudo_shap), axis=0)) ].index) #rank features based on aggregate attention scores
    
    display_features=feature_ordered[:max_display]
    pseudo_shap=pseudo_shap[display_features] #reorder the pseudo_shap panda dataframe. 
    
    
    X_test_data=pd.DataFrame(panda_x_test)[display_features]
    
    
    
    
    grouping= (y_pred>np.quantile(y_pred, best_quant[0]))*1 #patients with high survival 
    grouping=np.array( [1 if x==1 else -1 for x in grouping])
    
    N_low_risk=Counter(grouping)[1]
    N_high_risk=Counter(grouping)[-1]
    
    #group the pseudo_shap into low vs high risk patients
    
    grouped_pseudo_shap= np.array(pseudo_shap) * grouping.reshape(-1, 1) #group them to patients with low vs high risk.
    
    
    # produce beeswarm plot
    f, ax=plt.subplots(1,1, figsize=fig_size)
    
 
    nbins = 100
    N=grouped_pseudo_shap.shape[0]

    #cmap =mpl.cm.RdBu
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cm1= mpl.colors.LinearSegmentedColormap.from_list("MyCmapName",["skyblue","red"])
    cpick = mpl.cm.ScalarMappable(norm=norm,cmap=cm1)

    for i,pos in enumerate(reversed(np.arange(grouped_pseudo_shap.shape[1]))):

        shaps=grouped_pseudo_shap[:,i] #patient i attention vector (for all variables)
        
        #Ensure equal spacing on the y axis of beeswarm plot. 
        row_height = 0.4
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)) 
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        ax.scatter(x=shaps, y=ys+pos, c=np.array(X_test_data)[:,i], s=16, alpha=1.0, cmap=cm1)
    ax.set_xlabel("Attention Scores (impact on model output)")

    ticklabels=list(reversed(display_features))
    _, _=plt.yticks(range(len(display_features)), ticklabels, fontsize=fontsize)
    
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    left_side = ax.spines["left"]
    left_side.set_visible(False)

    top_side = ax.spines["top"]
    top_side.set_visible(False)
    ax.tick_params(axis='y', which=u'both',length=0) #remove tick marks for better visibility 
    vmin=np.min(plt.yticks()[0])
    vmax=np.max(plt.yticks()[0])
    _=ax.vlines(0.0, -0.5, vmax+0.5, color="darkgrey")
    _=ax.set_ylim(-0.5, len(display_features))

    #add new xlables 
    xticklabels=list(np.round( np.abs(plt.xticks()[0]),4)  )
    _, _=plt.xticks( list(plt.xticks()[0]),xticklabels, 
              fontsize=11)
    #add the colorbar!
    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=cm1)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label("Feature Values", size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)

    #add grouping labels - high risk (lower survival), low risk (higher survival)
    ax.text((plt.xticks()[0][0])*0.90, plt.yticks()[0][0], 
            'High Risk Patients '+ " (" +str( N_high_risk )+")",
            style='normal', fontweight='bold',
            bbox={'facecolor': 'red', 'alpha':0.2, 'pad': 5})

    _=ax.text((plt.xticks()[0][len(plt.xticks()[0])-1])*0.25, plt.yticks()[0][0], 
              'Low Risk Patients'+ " (" +str( N_low_risk )+")",
              style='normal', fontweight='bold',
              bbox={'facecolor': 'green', 'alpha': 0.2, 'pad': 5})
    _=ax.set_title( title, color='k', fontsize=15, fontweight='bold')
    
    f.savefig(title+" beeswarmplot.png")
    plt.show()


### STABILITY TEST FOR UNIFORMER 

def uniformer_stability_test(time, Predictors, event, d_model=32, dff=32,  n_layers=2, n_heads=2,batch_size=100,epochs=100, No_rand_splits=100, test_size=0.20, No_top_var=5, normalize=True, directionality=False, fixed_quant=True):
    

    features=list(Predictors)  #names of the variables from a pandas dataframe
    
    y=np.array(time).astype(np.float32)
    x=np.array(Predictors).astype(np.float32)
    
    if normalize:
        x=(x-np.min(x, axis=0))/(np.max(x, axis=0)-np.min(x, axis=0)) #normalize to values between 0 and 1
    
    x = x.reshape([x.shape[0], 1, x.shape[1]] )#reshape data to match what UniFormer is expecting 
    
    event= np.array(event).astype(np.float32)
    
    cv_cindex_test=[]
    top_var_list=[]
    directionality_tables=[]
    best_quant_list=[]
    Hazard_ratios=[]
    

    for i in np.arange(No_rand_splits):

        X_train, X_test, y_train, y_test, e_train, e_test, X_train1, X_test1= train_test_split(
        x, 
        y, 
        event,
        Predictors, 
        test_size=test_size, 
        random_state=i,
        stratify=event)

        #cast input to tensor 
        X_train=tf.cast(X_train, tf.float32)
        X_test=tf.cast( X_test, tf.float32)

        surv_list=[y_train, e_train]
        data_surv = tf.cast( np.array(surv_list).T, tf.float32)

        batch_size=tf.cast(batch_size, tf.int64)
        #tf.random.set_seed(42) #fix random start for gradient descent 
        tf.debugging.set_log_device_placement(False)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = Transformer(
                num_layers=n_layers, 
                d_model=d_model, 
                num_heads=n_heads, 
                dff=dff,
                num_classes=1
            )

            model.compile(optimizer="adam",
                          loss=cIndex_SigmoidApprox,
                          metrics=[sigmoid_concordance])

            model.fit(X_train, data_surv, epochs=epochs, batch_size=batch_size, verbose=0)
        
        y_pred, w, embeddings=model.predict(X_test)

        surv_test=[y_test, e_test]
        data_test = tf.cast( np.array(surv_test).T, tf.float32)

        cv_cindex_test.append(sigmoid_concordance(data_test,  y_pred))
   
        top_var_list.append( top_variables(w, No_top_var, 25, features)) #global top variables 
    
        if directionality:
            best_quant, HR_s=best_quantile_cutoff(y_test, e_test, y_pred, fixed_quant=fixed_quant)  
            directionality_tables.append( t_test_for_directionality(Predictors, X_test1.index, y_pred,best_quant)   ) #record 
        #directionality table 
            best_quant_list.append(best_quant) #record best quantile cut-off
            Hazard_ratios.append(HR_s)
        
    return (cv_cindex_test, top_var_list, best_quant_list,Hazard_ratios, directionality_tables)


# BASELINE COXPH STABILITY TEST. 

def baseline_CoxPH_stability_test(time, Predictors, event,No_rand_splits=100, test_size=0.20, No_top_var=5, normalize=True):
    
    
    features=list(Predictors)
    
    y=np.array(time).astype(np.float32)
    x=np.array(Predictors)
    if normalize:
        x=(x-np.min(x, axis=0))/(np.max(x, axis=0)-np.min(x, axis=0)) #normalize to values between 0 and 1
    #x = x.reshape([x.shape[0], 1, x.shape[1]] )#reshape data to match what UniFormer is expecting 
    
    event= np.array(event).astype(np.float32)

    cox_ph_cindex=[]
    cox_coef=[]
    for i in np.arange(No_rand_splits):
        X_train, X_test, y_train, y_test, e_train, e_test= train_test_split(
        x, 
        y, 
        event,
        test_size=test_size, 
        random_state=i,
        stratify=event
        )

        X_train0=np.array(X_train)
        #X_train0=X_train0.reshape([X_train0.shape[0], X_train0.shape[2]] )

        x_var=pd.DataFrame(X_train0, columns=features).reset_index(drop=True)

        y_var=pd.DataFrame( list(zip(y_train, e_train)), columns=["time", "event"])
        mydata=pd.concat( [ y_var, x_var], axis=1)

        cph = CoxPHFitter()
        cph.fit(mydata, duration_col='time', event_col='event')
        cox_coef.append(cph.summary.coef)

        X_test0=np.array(X_test)
        #X_test0=X_test0.reshape([X_test0.shape[0], X_test0.shape[2]] )

        xtest_var=pd.DataFrame(X_test0, columns=features).reset_index(drop=True)

        cox_pred=cph.predict_partial_hazard(xtest_var)
        cox_pred=np.reshape( np.array(cox_pred), (cox_pred.shape[0], 1 ))

        surv_test=[y_test, e_test]
        data_test = np.array(surv_test).T
        cox_ph_cindex.append( sigmoid_concordance(data_test,  -cox_pred))
        
    return (cox_ph_cindex,cox_coef)



