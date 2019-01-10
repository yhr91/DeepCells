"""
Use case: Deep learning for high content imaging
Description: Base TF model class from which other models inherit
Baseline definitions are for a tensorflow model, but derived classes
can work with keras models
Author: yhr91

"""

import tensorflow as tf
import numpy as np
import os
import glob
import cv2
from TF_ops import *
from TF_data import *
from Helpers import *
from keras.callbacks import TensorBoard


class TF_model():

    def __init__(self, NUM_CLASSES, BATCH_SIZE = 200, 
                 input_shape = None, model = None,
                 config = tf.ConfigProto(), **kwargs):
        
        self.NUM_CLASSES = NUM_CLASSES
        self.BATCH_SIZE = BATCH_SIZE
        self.input_shape = input_shape
        self.model = model
        self.config = config
        
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
                
        if(self.model_init() == -1):
            return
        
        self.load()
        
    def model_init(self):
       
        in_size = list()
        in_size.append(self.BATCH_SIZE)
        in_size.extend(self.input_shape)
        
        self.X = tf.placeholder(tf.float32, in_size,
                                name = "placeholder")
        self.Y = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.NUM_CLASSES],
                                name = "Y_placeholder")
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name='global_step')
        self.ckpt_base_path = '/data/ifd3/dev/Cell_Imaging/TF/checkpoints/'
   
   
    def load(self, **kwargs):
        # Must be implemented by children
        print('No Network loaded')
        
    def initialize_sess(self):
            # Initialize a variable saver and load checkpoint
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep = 0)
            
            ckpt_path = self.ckpt_base_path + self.net_name +'/'
            if not os.path.exists(ckpt_path):
                print('NO CHECKPOINT FOUND: model weights initialized from scratch')
                os.makedirs(ckpt_path)
            
            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname(ckpt_path))
            
            # if a checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring checkpoint from {}'.format(ckpt.model_checkpoint_path))
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                
            return saver
                    
    def train(self, X_batches = None, Y_batches = None, 
                   lmdb = None, lmdb_keys = None, n_epochs = 5,
                   val_lmdb = None, val_keys = None ):  
        
        with tf.Session(config=self.config) as self.sess:
            saver = self.initialize_sess()
            writer = tf.summary.FileWriter('/data/ifd3/dev/Cell_Imaging/TF/graphs/'+self.net_name,
                                           self.sess.graph)
            
            initial_step = self.global_step.eval()
            SKIP_STEP = 50
            total_loss = 0
            if lmdb is not None: 
                print('Begin training with live reads from LMDB')
                if type(lmdb) is not list:
                    lmdb = [lmdb]
                if lmdb_keys is None:
                    _ = [l.make_key_batches(self.BATCH_SIZE) for l in lmdb]
                    
            elif X_batches is not None and Y_batches is not None: 
                print('Begin training while reading images from memory')
            else:
                raise InputError("No training data")
                
            if val_keys is not None:
                print('Reading validation data from LMDB')
                if val_lmdb is None: val_lmdb = lmdb;
                val_X, val_Y = val_lmdb.read_lmdb_batch(val_keys, self.NUM_CLASSES)
                
            i = 0
            n_batches = np.max([len(l.key_batches) for l in lmdb]) 
            print('Training step:', initial_step)
            for index in range(initial_step, n_batches * n_epochs):
                
                if lmdb is not None:
                    for l in lmdb:
                        try:
                            X_batch, Y_batch = l.read_lmdb_batch(l.key_batches[i], self.NUM_CLASSES)
                        except:
                            continue
                if X_batches is not None:
                    X_batch = X_batches[i]; Y_batch = Y_batches[i]
                
                
                _, loss_batch, summary = self.sess.run([self.optimizer, self.loss,
                                    self.train_summary_op], feed_dict={self.X: X_batch, 
                                    self.Y:Y_batch}) 
                
                i += 1
                if i == n_batches:
                    i = 0
                    
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0:
                    
                    val_acc = None
                    if val_keys is not None:
                        val_acc, _, _,  val_summary = self.infer_batch(val_X, val_Y)
                        writer.add_summary(val_summary, global_step=index)
                    else: val_acc = 0;
                        
                    print('At step {}, Average loss : {:5.3f}, Validation accuracy\
                          : {:5.3f}'.format(index+1, (total_loss/SKIP_STEP), val_acc))
                    total_loss = 0.0
                    saver.save(self.sess, self.ckpt_base_path +self.net_name+'/', index)
                    
                writer.add_summary(summary, global_step=index)               
    
                   
    def infer_batch(self, X_batch, Y_batch = None, test=True):
        if (test):
            acc, pred, feat, summary = self.sess.run([self.accuracy, 
                                      self.prediction, self.flattened, self.val_summary_op], 
                                      feed_dict={self.X: X_batch, self.Y:Y_batch})
            return acc, pred, feat, summary
        else:
            pred, feat, summary = self.sess.run([self.prediction, self.flattened,
                                                 self.val_summary_op], 
                                  feed_dict={self.X: X_batch, self.Y: np.zeros([len(X_batch), self.NUM_CLASSES])})
            return pred, feat, summary

    # TO-DO: Lots of redundant code in this method 
    def infer(self, data, batch_size, test = True):
        print("Starting batchwise inference run")
        total_correct_preds = 0
            
        # Initialize a variable saver and load checkpoint
        with tf.Session(config=self.config) as self.sess:
            saver = self.initialize_sess()

            # Read in images first
            try:
                if (data.images.keys()[0]):
                    print('Images pre-loaded')
            except:
                data.read_images_all()
            print('Finished reading')    
            
            patch_batch = []
            patch_batch_class = []
            patch_batch_names = []
            
            total_items = 0
            accs = []
            names = []
            preds = []
            classes = []
            feats = []
            
            for name, img  in iter(data.images.items()):
                if img is None:
                    print('Skipping an empty file: '+name)
                    continue
                ext, pname = data.pbatch_extend(patch_batch, patch_batch_class,
                                                name, img)
                patch_batch_names.extend(pname)
                
                if len(patch_batch) > batch_size:
                    # Keep the extras for the next batch
                    xtra = patch_batch[batch_size:]
                    xtra_class = patch_batch_class[batch_size:] 
                    xtra_names = patch_batch_names[batch_size:]
                    
                    batch = patch_batch[:batch_size]
                    batch_class = patch_batch_class[:batch_size]
                    batch_names = patch_batch_names[:batch_size]             
                    total_items += len(batch)
                    
                    if test:
                        acc, pred, feat, _ = self.infer_batch(batch, 
                                     one_hot(batch_class,self.NUM_CLASSES), test=True)
                        accs.append(acc)
                        classes.extend(batch_class)
                        print("Processed {} images, Batch \
                                    accuracy: {}".format(total_items, 
                                            acc))
                    else:
                        pred, feat, _ = self.infer_batch(batch, test=False)
                    
                        print("Processed {} images".format(total_items))
                    
                    names.extend(batch_names)
                    preds.extend(pred)
                    feats.extend(feat)
                    
                    patch_batch = list(xtra)
                    patch_batch_class = list(xtra_class)
                    patch_batch_names = list(xtra_names)
          
            if test:
                print("Accuracy {0}".format(np.mean(accs)))
                self.results = [names, preds, classes, feats]
            else:
                self.results = [names, preds, feats]
            
    def calc_full_img_results(self, test=True, op='avg', fld_avg=False,
                              class_interest = None):
        """ Return results on the full image as opposed
        to just crops
        """
        if(fld_avg==True):
                self.results[0] = [name.split('fld ')[0] for name in self.results[0]]
                self.calc_full_img_results(test=test, op=op, fld_avg = False)
        
        if test:
            try:
                [names, preds, truth, feats] = self.results
            except:
                [names, preds, truth] = self.results
                feats = [0] * len(self.results[0])
        else:
            try:
                [names, preds, feats] = self.results
            except:
                [names, preds] = self.results
                feats = [0] * len(self.results[0])
        full_img = {}
    
        # First create a dictionary of full images
        for i,name in enumerate(names):
            if name in full_img.keys():
                full_img[name]['pred'].append(preds[i])
                full_img[name]['feats'].append(feats[i])
            else:
                if test:
                    full_img[name] = {'pred': [preds[i]], 'truth':truth[i], 'feats':[feats[i]]}
                else:
                    full_img[name] = {'pred': [preds[i]], 'feats':[feats[i]]}
            
        # Average over all sub image probabilities
        full_img_avg = {}
        for key in iter(full_img.keys()):
            if(op=='avg'):
                def op(x):
                    return np.mean(x,0)
            elif(op=='max'):
                def op(x):
                    return np.argmax(x,1)[class_interest]
            elif(op=='avg_max'):
                def op(x):
                    return np.mean(np.argmax(x,1)[class_interest])
                 
            if test:
                 full_img_avg[key] = {'pred': op(full_img[key]['pred']),
                                     'truth': full_img[key]['truth'],
                                     'feats': np.mean(full_img[key]['feats'],0)}
            else:
                 full_img_avg[key] = {'pred': op(full_img[key]['pred']),
                                     'feats': np.mean(full_img[key]['feats'],0)}
                    
        self.full_results = full_img, full_img_avg

class keras_model(TF_model):
    """ Similar functionality as TF_model but works with externally defined keras
        models
        
        Also has the capability to work with multiple TF_input files"""
    
    def model_init(self):
        try:
            self.model
        except:
            print("Please pass a keras model for initialization")
            return -1
        
        self.ckpt_base_path = '/data/ifd3/dev/Cell_Imaging/TF/checkpoints/'
    
    def load(self, train_data = None, val_data = None, val_batch_size = None,
            class_weight = None, model_name = None):
        
        self.class_weight = class_weight
        
        try:
            scenario_name = self.scenario_name
        except:
            scenario_name = 'NoDataName'
            
        if model_name is None:
            self.net_name = 'keras_model_'+ scenario_name
        else:
            self.net_name = model_name + '_' + scenario_name
        
        # Load train and validation keys
        if train_data is not None:
            if type(train_data) is not list:
                train_data = [train_data]
                
            self.train_data = train_data
            self.train_keys_batches = []
            
            for t in train_data:
                if not hasattr(t, 'lmdb_keys'):
                    t.get_lmdb_keys()
                train_keys = list(t.lmdb_keys)
                shuffle(train_keys)
                self.train_keys_batches.append(batchize(train_keys, self.BATCH_SIZE))
        
        if val_data is not None:
            self.val_data = val_data
            if val_batch_size is None:
                val_batch_size = self.BATCH_SIZE
            if not hasattr(val_data, 'lmdb_keys'):
                val_data.get_lmdb_keys()
            val_keys = list(val_data.lmdb_keys)
            shuffle(val_keys)
            self.val_keys_batches = batchize(val_keys, val_batch_size)
            
            
        # Load weights if they already exist
        self.ckpt_path = self.ckpt_base_path + self.net_name +'/'
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
            self.model.save(self.ckpt_path + 'base_model')
            
        weights = [f for f in glob.glob(self.ckpt_path + '*.h5') if '.h5' in f]
        if not weights:
            print('NO CHECKPOINT FOUND: model weights initialized from scratch')      
        else:
            self.weights_file = weights[-1]
            print('Restoring checkpoint from {}'.format(self.weights_file))
            self.model.load_weights(self.weights_file)
        
    def train(self, num_epochs = 100, min_delta = 0.00001, 
              patience =100, save_step = 100):
        
        self.train_outs = []; self.val_outs = [];
        self.tboard_setup()
        self.tensorboard.set_model(self.model)
        
        # Trying to flip through datasets per iteration while using data maximally
        max_batch_len = np.max([len(batches) for batches in self.train_keys_batches])
        
        def named_logs(logs, prefix=None):
            result = {}
            for l in zip(self.model.metrics_names, logs):
                if prefix is not None:
                    result[str(prefix) + '_' + l[0]] = l[1]
                else:
                    result[l[0]] = l[1]
            return result
                
        try:
            self.itr = int(self.weights_file.split('_')[-2]) + 1
        except:
            pass
        
        for epoch in range(num_epochs):
            output = []; 

            # Iterate over each batch within ...
            for b in range(max_batch_len):
                # ... each list of batches (dataset specific)
                for i, batches in enumerate(self.train_keys_batches):
                    try:
                    # Check if the index exists for a particular batch list
                        logs = self.train_iteration(batches[b], i)
                        output.append(logs)
                        self.tensorboard.on_epoch_end(self.itr, named_logs(logs, 'train'))
  
                    except:
                        # If not, move on
                        continue
                    self.itr += 1
                self.train_outs.append(np.mean(output,0))

                # Validation after an epoch
                logs = self.val_iteration()
                self.val_outs.append(logs)
                self.tensorboard.on_epoch_end(self.itr-1, named_logs(logs, 'val'))
                
                # Save the trained weights every n iterations
                if (self.itr > self.last_step + save_step):
                    self.model.save_weights(self.ckpt_path + '_' + str(self.itr) +'_weights.h5')
                    self.last_step = self.itr

                print('Training metrics:', np.mean(output,0))
                print('\t Validation metrics:', logs)
                
                
                # Early stopping
                if len(self.val_outs)>2:
                    if int(self.val_outs[-1][1] - self.val_outs[-2][1] < min_delta):
                        self.count +=1
                    else:
                        self.count =0
                    if self.count == patience:
                        self.model.save_weights(self.ckpt_path + '_' + str(self.itr) +'_weights_ESTOP.h5')
                        print("Early Stopping ! < -------   ")
                        
        self.tensorboard.on_train_end(None)
            
    def tboard_setup(self):
        try:
            if (self.tboard_flag):
                return
        except:
            self.tensorboard = TensorBoard(
              log_dir='/data/ifd3/dev/Cell_Imaging/TF/graphs/'+self.net_name,
              histogram_freq=0,
              batch_size=self.BATCH_SIZE,
              write_graph=True,
              write_grads=True
            )
            
            # Set iteration counters        
            self.count = 0; self.itr = 0; self.last_step = 0;
            self.tboard_flag = 1
    
    def train_iteration(self, batch, i=0):
        # Use that dataset for training for that iteration
        x_batch, y_batch = self.train_data[i].read_lmdb_batch(batch, 
                                multitask_binary=self.multitask_binary)
        return self.model.train_on_batch(x_batch, y_batch,
                                        class_weight = self.class_weight)
        
    def val_iteration(self):
        x_batch, y_batch = self.val_data.read_lmdb_batch(self.val_keys_batches[0], 
                                multitask_binary = self.multitask_binary)
        return self.model.test_on_batch(x_batch, y_batch)
    
    def infer_iteration(self, lmdb, keys, test=False):
        x_batch, y_batch = lmdb.read_lmdb_batch(keys, 
                                multitask_binary = self.binary)
        preds = self.model.predict(x_batch, batch_size = len(x_batch))
        return (preds, y_batch)
        
    def infer(self, data, lmdb = None, test = True, infer_batch_size = None):
        # Add batching capability for optimization
          
        names = []
        preds = []
        classes = []
              
        if lmdb is not None:
            if not hasattr(lmdb, 'lmdb_keys'):
                lmdb.get_lmdb_keys()
                if infer_batch_size is None:
                    infer_batch_size = self.batch_size
                lmdb.make_key_batches(batch_size = infer_batch_size)
                
            for i, keys_batch in enumerate(lmdb.key_batches):
                print(i,' of ',len(lmdb.key_batches))
                (pred, truth) = self.infer_iteration(lmdb, keys_batch)
                names.extend(keys_batch)
                preds.extend(pred)
                classes.extend(truth)
            
            # Since these patches have been pre-made their names have to cleaned
            names =[n.split(':patch')[0] for n in names]
            self.results = [names, preds, classes]
            return self.results
          
        ### Everything under here needs to be re-done because it's just been ported over    
        try:
            if (data.images.keys()[0]):
                print('Images pre-loaded')
        except:
                data.read_images_all()
                print('Finished reading')    
            
        patch_batch = []
        patch_batch_class = []
        patch_batch_names = []

        total_items = 0

        for name, img  in iter(data.images.items()):
            if img is None:
                print('Skipping an empty file: '+name)
                continue
            _, pname = data.pbatch_extend(patch_batch, patch_batch_class,
                                            name, img)
            patch_batch_names.extend(pname)
            batch = patch_batch
            batch_class = patch_batch_class
            batch_names = patch_batch_names           
            total_items += len(batch)
            pred = self.model.predict(batch, batch_size = self.BATCH_SIZE)

            print("Processed {} images".format(total_items))

            names.extend(batch_names)
            preds.extend(pred)
            classes.extend(batch_class)

        if test:
            self.results = [names, preds, classes, feats]
        else:
            self.results = [names, preds, feats]
            
        
class InputError(Exception):
    """Exception raised for errors in the input.
    """
    
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
