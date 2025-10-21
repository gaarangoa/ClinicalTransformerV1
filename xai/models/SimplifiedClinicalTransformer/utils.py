import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
import pickle
import os

def create_padding_mask(seq):
    ''' Mask the pad variables of the input. 
        It ensure that the model does not treat padding as the input. 
        The mask indicates where pad value 0 is present: it outpus
        1 at those locations and 0 otherwise.
        
        This is useful for self training
        
    '''
    seq = tf.cast(tf.math.equal(seq, 0.), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

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

    output = tf.matmul(attention_weights, v, transpose_a=False)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class NumericalEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size=64):
        super(NumericalEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.y = tf.keras.layers.Dense(embedding_size)
        
    def call(self, x):
        return self.y(x)
    
class PriorEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size=64):
        super(PriorEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.y = tf.keras.layers.Dense(embedding_size)
        
    def call(self, x):
        return self.y(x)

class TokenEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocabulary_size=128, embedding_size=64, input_size=24):
        super(TokenEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.input_size = input_size
        
        self.y = tf.keras.layers.Embedding(self.vocabulary_size, self.embedding_size, input_length=self.input_size)
        
    def call(self, x):
        return self.y(x)
    
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
        
    return output, attention_weights, q, k

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.3):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):
    
    attn_output, attn_weights, q, k = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x+attn_output)  # (batch_size, input_seq_len, d_model), 
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    

    return out2, attn_weights, q, k   #return out2, attn_weights

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, num_features, embedding_size, max_features, rate=0.3, with_prior=True):
    super(Encoder, self).__init__()

    self.with_prior = with_prior
    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = NumericalEmbeddingLayer(embedding_size = d_model)
    self.embedding_token = TokenEmbeddingLayer(num_features, embedding_size, input_size=max_features)
    self.embedding_prior = PriorEmbeddingLayer(embedding_size = d_model)
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, inp, training, mask):
    
    if self.with_prior:
      # We have prior knowledge embeddings feeding the model
      XN, XT, XP = inp[0], inp[1], inp[2]
      
      # numerical embedding
      x = tf.reshape(XN, (tf.shape(XN)[0], tf.shape(XN)[1], 1))
      x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
      
      # categorical / feature name embedding
      z = self.embedding_token(XT)
      
      # Prior knowledge embeddings
      w = self.embedding_prior(XP)
      x += z
      x += w
    
    else:
      # There is no prior knowledge 
      XN, XT = inp[0], inp[1]
      # numerical embedding
      x = tf.reshape(XN, (tf.shape(XN)[0], tf.shape(XN)[1], 1))
      x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
      
      # categorical / feature name embedding
      z = self.embedding_token(XT)      
      x += z
    
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout(x, training=training)
    
    ws = []
    qs = []
    ks = []
    for i in range(self.num_layers):
        x, w, q, k = self.enc_layers[i](x, training, mask)
        ws.append(w)
        qs.append(q)
        ks.append(k)
    
    return x, ws, qs, ks  # (batch_size, input_seq_len, d_model)

def load_model(sdir, run, epoch=None, **kwargs):
    # Load model
    tf.random.set_seed(0)

    trainer = pickle.load(
        open('{}/{}/traineer.pk'.format(sdir, run), 'rb')
    )
    
    # Load testing data
    trainer.out_dir = sdir
    # trainer.dataset = pickle.load(open('{}/dataset.pk'.format(sdir) , 'rb'))
    # trainer.X_train, trainer.X_test = pickle.load(open("{}/{}/train_test_dataset.pk".format(sdir, run), 'rb'))
    # trainer.training_data_generator, trainer.testing_data_generator = pickle.load(open("{}/{}/train_test_dataset_generator.pk".format(sdir, run), 'rb'))

    # trainer.testing_data_generator.augment_copies = 1
    # trainer.training_data_generator.augment_copies = 1
    
    # trainer.training_data_generator.shuffle = False
    # trainer.testing_data_generator.shuffle = False

    epochs = os.listdir('{}/{}'.format(sdir, run))
    
    # Load model weights
    file='{dir}/{fold}/model.E{epoch:06d}.h5'.format(dir=sdir, fold=run, epoch=epoch)
    trainer.model = trainer.Transformer(
        num_layers=trainer.num_layers, #change number of layers (more layers)
        d_model=trainer.embedding_size, 
        num_heads=trainer.num_heads, # reduce number of heads (one head)
        dff=trainer.embedding_size,
        num_classes=trainer.num_classes, 
        masking=True,
        num_features=trainer.tokenizer.vocabulary_size, 
        embedding_size=trainer.embedding_size, 
        max_features=trainer.max_features, 
        with_prior=trainer.priors
    )

    # X, _ = trainer.testing_data_generator.__getitem__(0)
    # _=trainer.model(X)
    _=trainer.model(trainer.dummy_data)
    trainer.model.load_weights(file)
    
    return trainer

def clean_run(**kwargs):
    '''
    Remove models that accumulate when training. This function will remove all the unwanted model files. 

    directory: path
    keep: A list with the models to keep. 

    Example: 

    clean_run(
        directory='./',
        keep=[50, 60, 70, 80, 90, 100, 110]
    )
    
    '''

    directory = kwargs.get('path', '/scratch/kmvr819/data/mlflow/')
    # project_id = kwargs.get('project_id', None)
    # run_id = kwargs.get('run_id', None)
    # pretrained = kwargs.get('pretrained', '')
    ignore = kwargs.get('keep', None)

    print('Processing: {}'.format(directory))
    
    assert(ignore)
    
    # directory = '{}/{}/{}/artifacts/models/{}/'.format(
    #     root_path, 
    #     project_id,
    #     run_id,
    #     pretrained,
    # )

    try:   
        os.remove("{}/dataset.pk".format(directory))
    except:
        pass
    
    folds = [i for i in os.listdir(directory) if 'fold-' in i]
    ignore = ['model.E{epoch:06d}.h5'.format(epoch=k) for k in ignore]

    for fold in folds:
        fold_dir = "{}/{}/".format(directory, fold)
        
        models = [i for i in os.listdir(fold_dir) if 'model.' in i ]
        for ix, model in enumerate(models):
            if not model in ignore: 
                try:
                    os.remove("{directory}/{model}".format(directory=fold_dir, model=model))
                except Exception as inst:
                    print(inst)
                    pass

        # clean trainer file: BUG: for old runs, we need to remove a copy of the data stored in the tokenizer class. 
        # New runs have this fixed: remove train_test_generatos and dataset.pk, used only during training.

        try:
            os.remove("{}/train_test_dataset.pk".format(fold_dir))
            os.remove("{}/train_test_dataset_generator.pk".format(fold_dir))
        except:
            pass

        try:
            # remove copy of the data in trainer class
            trainer = pickle.load(open("{}/traineer.pk".format(fold_dir), 'rb'))

            trainer.data_converter.df = []
            trainer.base_model.data_converter.df = []

            pickle.dump(trainer, open("{}/traineer.pk".format(fold_dir), 'wb'))
        except:
            pass
        
        


    


def get_q(x, cutoffs, fold):
    if x < cutoffs[fold][0.25]:
        return 'Q1'
    elif x < cutoffs[fold][0.5]:
        return 'Q2'
    elif x < cutoffs[fold][0.75]:
        return 'Q3'
    else:
        return 'Q4'