import tensorflow as tf
from tensorflow.keras.losses import Loss
from xai.losses.survival import cIndex_SigmoidApprox as cindex_loss

class CompositeLoss(Loss):
    def __init__(self, a1=1, a2=1, a3=1):
        super().__init__()
        '''
            a1: token loss
            a2: value loss
            a3: survival loss
        '''
        self.token_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.value_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.cindex_loss = cindex_loss
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    
    def call(self, y, pred):
        
        y_pred = pred[:, :, :-2]
        v_pred = pred[:, :, -2:-1]
        s_pred = pred[:, :, -1:]
        
        
        loss1 = self.token_loss(y[:, :, 1], y_pred, sample_weight=y[:, :, 2])
        loss2 = self.value_loss(y[:, :, 3:4], v_pred, sample_weight=y[:, :, 2:3])
        loss3 = self.cindex_loss(y[:, 0, 4:], s_pred[:, 0, :])
        
        return self.a3*loss3 + self.a1*loss1 + self.a2*loss2
    