"""
Use case: Deep learning for high content imaging
Description: Some helpers for TF based DL
Author: yr897021
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def one_hot(vec, cols = None):
    """ Take a vector of labels and converts it into a one hot 
    encoding
    """

    if cols is None:
        cols = len(np.unique(np.array(vec)))
    one_hot_vec = np.zeros([len(vec), cols])
    one_hot_vec[np.arange(len(vec)),vec] = 1
    return one_hot_vec

def mult_task_label(vec, d=4):
    """ Take a vector of int labels and converts it into a vector
    of binary lists
    """
    ret = []
    for v in vec:
        ret.append([i for i in ('{:0' + str(d) + 'b}').format(v)])
    return np.array(ret).astype('float')

def create_overlay(img, in_size, stride, probs):
    """ Take an image, patch_size, stride_size and associated
    patch probability and produce a similar sized overlay image
    """
    im_size = img.shape
    
    row_range = range(0, im_size[0]-in_size[0]+1,
                  stride[0])
    col_range = range(0, im_size[1]-in_size[1]+1,
                  stride[1])
    
    overlay = np.zeros_like(img)
    it = 0
    for row in row_range:
        for col in col_range: 
            overlay[row:row+in_size[0], col:col+in_size[1],0] = (128*probs[it])
            it = it+1
    
    return overlay

def create_6_plots(batch, labels = None):
    """ Take in a batch of images, overlays and class labels
    and produce a PDF with the results
    """
    
    if labels is None:
        labels = ['Basal', 'Activated']
    fig = plt.figure(figsize=[20,20])
    
    it = 1
    for img, overlay, class_ in batch:
        fig.add_subplot(3,2,it)
        plt.imshow(img*10)
        plt.imshow(overlay, alpha=0.6)
        plt.axis('off')
        plt.text(200, 200, 'True label: %s'%labels[class_], 
             fontdict={'size': 30, 'color':'white'})
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2, hspace=0.01)
        it += 1
    return fig
