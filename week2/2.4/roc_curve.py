from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def roc_curve(probabilities, labels): 
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
#     probset = set(probabilities)
#     print labels
    data = np.vstack((np.sort(probabilities),np.sort(labels)))
#     print data
    tpr = []
    fpr = []
    thresholds = np.sort(probabilities)
#     print thresholds
    for z in thresholds:
#         print z
#         pos = len(np.where(probabilities > z)[0])
        tp = np.sum(np.logical_and(data[1,:] == 1,
                                             data[0,:] > z))
        fn = np.sum(np.logical_and(data[1,:] == 1,
                                             data[0,:] < z))
        fp = np.sum(np.logical_and(data[1,:] == 0,
                                             data[0,:] > z))
        tn = np.sum(np.logical_and(data[1,:] == 0,
                                             data[0,:] < z))
#         print tp, fn, fp, tn
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
    return tpr,fpr, thresholds