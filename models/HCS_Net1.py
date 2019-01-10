"""
Use case: Deep learning for high content imaging
Description: TF_model HCS_Net1 """

from TF_model import *
from TF_data import *
from TF_ops import *
import tensorflow as tf
import numpy as np
import os

class HCS_Net1(TF_model):

    def load(self):
        """Create the network graph."""
        try:
            scenario_name = self.scenario_name
        except:
            scenario_name = 'NoDataName'
        self.net_name = 'HCS_Net1_'+ scenario_name
        print('Creating HCS_Net_1')
        LEARNING_RATE = 0.01
        
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.X, 11, 11, 16, 4, 4, padding='SAME', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')
        
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 1 group
        norm1 = tf.pad(norm1, [[0, 0], [2, 2], [2, 2], [0, 0]]) # Manual padding to make it match caffe model
        conv2 = conv(norm1, 5, 5, 32, 1, 1, padding='VALID', groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        norm2 = tf.pad(norm2,  [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv3 = conv(norm2, 3, 3, 64, 1, 1, padding='VALID', name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into 1 group
        conv3 = tf.pad(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv4 = conv(conv3, 3, 3, 64, 1, 1, padding='VALID', groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        pool5 = max_pool(conv4, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        self.flattened = tf.reshape(pool5, [-1, 7*7*64])
        self.features = fc(self.flattened, 7*7*64, self.NUM_CLASSES, relu=False, name='fc6')
        
        # Loss function
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.features)
            self.loss = tf.reduce_mean(entropy, name='loss')
        
        # Optimizer
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step,
                                           10000, 0.96, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, 
                                        global_step=self.global_step)
        
        # Prediction
        self.prediction = predict(self.features)
        
        # Accuracy
        self.accuracy = acc(self.prediction, self.Y, self.BATCH_SIZE)
        
        with tf.name_scope('Train'):
            tf.summary.scalar("Loss", tensor = self.loss)
            tf.summary.scalar("Accuracy",self.accuracy)
            tf.summary.histogram("HistLoss", self.loss)
            self.train_summary_op = tf.summary.merge_all()
            
        with tf.name_scope('Validate'):
            tf.summary.scalar("Loss", tensor = self.loss)
            tf.summary.scalar("Accuracy",self.accuracy)
            tf.summary.histogram("HistLoss", self.loss)
            self.val_summary_op = tf.summary.merge_all()
       
