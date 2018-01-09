from keras import backend as K

def emd(y_truth, y_pred):
    y_truth_cdf = K.cumsum(y_truth, axis=-1)
    y_pred_cdf = K.cumsum(y_pred, axis=-1)
    sample_wise_emd = K.sqrt(K.mean(K.square(K.abs(y_truth_cdf - y_pred_cdf)), axis=-1))
    return K.mean(sample_wise_emd)
