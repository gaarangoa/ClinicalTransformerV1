import tensorflow as tf 

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

