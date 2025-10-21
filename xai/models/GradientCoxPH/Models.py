import tensorflow as tf
from xai.losses.survival.coxph import cox_ph_loss as CoxPhLoss

def LinearModel(omega=0.01, seed=0, **kwargs):
    tf.random.set_seed(seed)
    
    lr = kwargs.get('learning_rate', 0.01)
    l1 = kwargs.get('l1', 1e-5)
    l2 = kwargs.get('l2', 1e-5)
    
    model = tf.keras.Sequential()
    
    model.add(
        tf.keras.layers.Dense(
            1, use_bias=True, 
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        )
    )
    
    opt=tf.keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(optimizer=opt, loss=CoxPhLoss)
    
    return model


def NonLinearModel(layers=1, embedding_size=32, seed=0, **kwargs):
    tf.random.set_seed(seed)
    
    lr = kwargs.get('learning_rate', 0.01)
    l1 = kwargs.get('l1', 1e-5)
    l2 = kwargs.get('l2', 1e-5)
    
    model = tf.keras.Sequential()
    
    [model.add(tf.keras.layers.Dense(
        embedding_size, activation="relu",
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)
    )) for layer in range(layers)]

    model.add(
        tf.keras.layers.Dense(
            1, use_bias=True, 
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        )
    )
    
    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=CoxPhLoss)
    
    return model