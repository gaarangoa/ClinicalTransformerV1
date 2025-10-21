import tensorflow as tf
from tensorflow.keras.losses import Loss
class CompositeLoss(Loss):
    '''
    Sample weight in the loss function is used to mute the value of the sample at a given
    output value. Let's say, your input has 10 tokens, you are masking 3 for self supervision
    Then when using the sample weight in the loss, you will be only taking the loss values of 
    those 3 tokens (Your sample weight then needs to be of the same size of your output). 
    If we would have only a classifier output, this wouldn't be recommended, as
    it would completely mask the value of the prediction. USE THIS ONLY WHEN YOU ARE DOING SELF
    SUPERVISION AND YOU HAVE MULTIPLE OUTPUTS AND ONLY FEW OF THEM YOU CARE. 
    '''
    
    def __init__(self, feature_w=1, value_w=0.01):
        super().__init__()
        self.token_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.value_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.feature_w = feature_w
        self.value_w = value_w

    def call(self, y, pred):
        
        y_pred = pred[:, :, :-1]
        v_pred = pred[:, :, -1:]
        
        loss1 = self.token_loss(y[:, :, 1], y_pred, sample_weight=y[:, :, 2])
        loss2 = self.value_loss(y[:, :, 3:], v_pred, sample_weight=y[:, :, 2:3])
        
        return self.feature_w*loss1 + self.value_w*loss2
    
loss = CompositeLoss()