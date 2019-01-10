"""
Use case: Deep learning for high content imaging
Description: Data reading, writing and manipulation class for TF models
Author: yr897021, unless otherwise stated
TO-DO: TF_data_set class needs to be finished, likely it will be
merged with parent
clearly position mean and normalization
Move out pbatch_extend

"""

import tensorflow as tf
from TF_ops import *
from Helpers import *
from cv2 import imread
import cv2
from random import shuffle
from image_process import easy_affines
import lmdb

## First some helpers

def batchize(data, batch_size, shuff = True):
    """ Create batches of batch_size from the given data
    """
    splits = np.arange(0,len(data),batch_size)
    if shuff:
        shuffle(data)
    batches = []
    if len(splits)>1:
        for s in range(1,len(splits)):
            batches.append(data[splits[s-1]:splits[s]])
    return batches

class TF_data():
    """ TF_data
    A  class for reading and manipulating data into TF models

    Inputs:
    path: path from which all containing image data is read
    batch_size: size of the batches that will contain image data

    Attributes:
    filename: complete path for an image file without path information
    labels: assigned classification label. Stored as dictionary using 
    filename as key
    image_IDs: string designed as "class label:filename:path_number". 
    Stored as dictionary using filename as key. This is used for writing 
    LMDB by children.
    """
    
    def __init__(self, path = None, batch_size = None, swap = None, 
                       set_mean = None, config = {},  
                       img_config={}, 
                       **kwargs):
        if path is not None: 
            print('Initializing' + path)
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.path = path
        self.config = config
        self.batch_size = batch_size
        self.set_mean = set_mean
        
        if 'input_shape' not in img_config: 
            img_config['input_shape'] = None
        if 'crop_shape' not in img_config: 
            img_config['crop_shape'] = None
        if 'crop_stride' not in img_config: 
            img_config['crop_stride'] = None
        self.img_config = img_config
        
        self.set_model_prop(**kwargs)
        self.read_data()
        self.labels={}
        if swap is not None:
            self.swap = swap
            self.swap_ch()
            
    def read_data(self):
        """ Reads filenames from input path, if TF_lmsb_set object is 
        instantiated then this is overridden 
        """ 
        self.read_names()
        if self.path is not None:
            self.read_images_all()
    
    def read_names(self):
        """ A method to read in all filenames from a given directory 
        and store it as an attribute. Called by non-LMDB objects
        """
        if self.path is None:
            return
        
        print('Reading filenames')
        
        if not self.path.endswith('/'):
            self.path = self.path + '/'

        files = os.listdir(self.path)
        extensions = ['tif','TIF','tiff']
        self.filenames = [self.path + f for f in files if f.split('.')[-1] in extensions]
        shuffle(files)

        if self.batch_size is not None:
            self.name_batches = batchize(files, self.batch_size)
            
    def set_model_prop(self, **kwargs):
        """ A method for assigning database/model specific parameters
        Implemented by children
        """
        pass
   
    def read_images_all(self):
        """ A method to read all images into a dictionary that links 
        filename to image
        Takes in tuples for swap_out, swap in of the form (condition,
        channel number). Indicates when to swap and which channel 
        to swap
        """
        
        print('Reading in all images...')
        self.images = {}
        try:
            self.config['skip_read']
        except: 
            self.config['skip_read'] = False
        if not self.config['skip_read']:            
            for it, filename in enumerate(self.filenames):
                self.images[filename] = imread(filename)
                if it%1000 == 0:
                    print(str(it)+' images read')     
            # If channels need to be swapped do it before any data access        
            try:
                if self.swap is not None: self.swap_ch();
            except:
                pass
        else:
            print('Skipping image reading')
        
        
    def swap_ch(self, new_name = None):
        """ Method to swap out an image channel with another image
        """
        print('Swapping out channels...')
        out_key, out_ch = self.swap[0]
        in_key, in_ch = self.swap[1]
        
        if new_name is None:
            new_name = ''
            
        for i, fname in enumerate(self.filenames):
            if out_key in fname:
                in_fname = fname.replace(out_key,in_key)
                new_fname = fname.replace(out_key,new_name)
                
                try:
                    self.config['skip_read']
                except: 
                    self.config['skip_read'] = False
                if not self.config['skip_read']:        
                    self.images[fname][:,:,out_ch] = self.images.pop(
                        in_fname)[:,:,in_ch]
                    self.images[new_fname] = self.images.pop(fname)
                
                self.filenames[i] = new_fname
                    
        # By now all marked images would have been swapped in/out
        self.filenames[:] = [f for f in self.filenames if in_key not in f]
        _ = [self.images.pop(f) for f in self.filenames if in_key in f]
        
#### ----- Some pre processors
        
    def preprocess(self, pix_scale = 255.):
        self.pix_rescale(pix_scale)
        self.mean_compute()
        self.mean_clean()
        self.max_normalize()
    
    def pix_rescale(self, pix_scale):
        """ Rescale pixels from 0 to 1
        """
        try:
            self.config['skip_pix_rescale']
        except:
            self.config['skip_pix_rescale'] = False
                
        if np.max(list(self.images.values())) <= 1: # Don't redo it...
                self.config['skip_pix_rescale'] = True
                
        if not self.config['skip_pix_rescale']:
            print('Pixel rescaling at ' + str(pix_scale+1) + ' bit')
            for keys in iter(self.images.keys()):
                self.images[keys] = self.images[keys]/pix_scale
    
    def mean_compute(self):
        """ Mean compute
        """
        try:
            self.config['skip_mean']
        except:
            self.config['skip_mean'] = False
        if not self.config['skip_mean']:
            if self.set_mean is None:
                print('Computing mean')
                self.mean = np.mean(np.array(list(self.images.values())),0)
            else:
                print('Assigning pre-defined mean')
                self.mean = self.set_mean
        
    def mean_clean(self):
        """ Mean subtract
        """
        try:
            self.config['skip_mean']
        except:
            self.config['skip_mean'] = False
        if not self.config['skip_mean']:
            print('Subtracting mean')
            for keys in self.images.iterkeys():
                self.images[keys] = (self.images[keys] - 
                                     self.mean.astype(int)).astype('int8')
    
    def max_normalize(self):
        """ Max-normalize to the maximum absolute value
        Not Recommended
        """
        try:
            self.config['skip_norm']
        except:
            self.config['skip_norm'] = False 
        if not self.config['skip_norm']:
            print('Normalizing')
            max_val = np.max(abs(np.array(list(self.images.values()))))
            for keys in self.images.iterkeys():
                self.images[keys] = self.images[keys]/max_val
                
                
    def patch_preprocess(self, patch):
        patch = self.patch_resize(patch)
        return patch
        
    def patch_resize(self, patch):
        return cv2.resize(patch, tuple(self.img_config['input_shape'][:2]), 
                           interpolation = cv2.INTER_CUBIC)
        
## ----- Processing images and attaching labels
    
    def create_image_IDs(self, label_dict=None, label_location = (0,1), 
                         plate_label = '0', label_split = None):
        """ Attach class labels to the image names and leave room for 
        patch numbers
        """
        self.image_IDs = {}
        self.labels = {}
        
        print("Creating Image IDs")
        # If the entire plate is one class
        if label_dict is None:
            print('NOTE: no labels found, assigning same label to all')
            for s in self.filenames:
                self.labels[s] = plate_label
                self.image_IDs[s] = ((str(plate_label) + ':' + 
                      s + ':patch%d'))
            
        elif -1 in label_dict.keys():
            for s in self.filenames:
                self.image_IDs[s] = ((str(label_dict[-1]) +
                                      ':' + s + ':patch%d'))
                self.labels[s] = int(label_dict[-1])

        else:
            for i,s in enumerate(self.filenames):
                if self.path is None:
                    imgname = s
                else:
                    imgname = s.split(self.path)[-1]
                
                
                # Link well to class
                try:
                    if label_split is not None:
                        label = label_dict[imgname.split(label_split)[0] + label_split]
                    else:
                        label = label_dict[imgname[label_location[0]:label_location[1]]]
                except:
                    print('No label for ', imgname.split(label_split)[0] + label_split)
                    continue
                      
                if int(label) == -1: 
                    self.filenames[i] = -1
                    _ = self.images.pop(s)
                else:
                    self.image_IDs[s] = ((str(label) + ':' + 
                                          s + ':patch%d'))
                    self.labels[s] = int(label)
            
            # Clear up the unclassified
            self.filenames[:] = [f for f in self.filenames if f != -1]
    
    def pbatch_extend(self, patch_batch, patch_batch_class, name, img):
        """ Helper method used by TF_model for creating LMDB key 
        batches for use during inference.
        Takes in an image and uses it to extend an array of patches and 
        patch class labels
        """
        patches = [self.patch_preprocess(patch) for 
                   patch in self.yield_patches(img, affines = False)]
        
        patch_labels = []
        # Checking if labels dictionary is empty
        if not not self.labels:
            patch_labels = [self.labels[name] for i in range(len(patches))]
        patch_names = [name]*len(patches)

        patch_batch.extend(patches)
        patch_batch_class.extend(patch_labels)
        
        return len(patches), patch_names
                   
    def yield_patches(self, image, affines, crop_shape=None, crop_stride=None):
        """ Given an image, yield patches based on the image shape and 
        given stride
        author: jumutc (Intel)
        TODO: Add an exists(crop_shape)? check
        """
        
        ## Read from base class variables ONLY if undefined in functional call
        if crop_shape is None:
            if self.img_config['crop_shape'] is not None:
                crop_shape = self.img_config['crop_shape']
            else:
                crop_shape = self.img_config['input_shape']
        
        if crop_stride is None:
            if self.img_config['crop_stride'] is not None:
                crop_stride = self.img_config['crop_stride']
            else:
                crop_stride = crop_shape[:2]
        
        row_range = range(0, image.shape[0]-crop_shape[0]+1,
                          crop_stride[0])
        col_range = range(0, image.shape[1]-crop_shape[1]+1,
                          crop_stride[1])

        for row in row_range:
            for col in col_range: 
                patch = np.empty(tuple(crop_shape),
                                 dtype=image.dtype)
                patch[:,:,:] = image[row:row+crop_shape[0],
                                     col:col+crop_shape[1],:]
                if affines:
                    affines_7 = easy_affines(patch)
                    for affine in affines_7.values():
                        yield affine 
                else:
                    yield patch
    
class TF_lmdb(TF_data):
    """
    TF_data::TF_lmdb
    Data class for reading and writing LMDB,

    Inputs:
    lmdb_file: lmdb file for reading from or writing to

    Attributes:
    """
    
    def set_model_prop(self, lmdb_file = 'untitled_lmdb', **kwargs):
        self.lmdb_file = lmdb_file
        self.dataset_name = self.lmdb_file.strip('/').split('/')[-1]
                 
    # ----- LMDB writing functions --------
           
    def save_lmdb(self, affines = True):
        """ Convert image names into batches of patches that are then written 
        as an lmdb database
        author: yr897021, jumutuc (Intel)
        """
        try:
            if self.images is None:
                self.read_images_all()
        except:
            self.read_images_all()
        map_size = 10000 * len(self.images) * np.prod(self.img_config['input_shape'])
        
        
        # Begin writing data
        print('Saving as LMDB')
        print('Number of original images: ' + str(len(self.images)))
        with lmdb.open(self.lmdb_file, map_size = map_size) as env:
            for it, fname in enumerate(self.filenames):
                image = self.images[fname]
                id = self.image_IDs[fname]
                with env.begin(write=True) as txn:
                    try:
                        for i, patch in enumerate(
                            self.yield_patches(image, affines = affines)):
                            patch = self.patch_preprocess(patch)
                            if txn.get((id % i).encode()) is None:
                                txn.put((id % i).encode(), patch.tobytes())

                    except IOError:
                        print('Cannot write file to disk!')
                    
        
    # ----- LMDB reading functions --------
       
    def get_lmdb_keys(self):
            print("Reading keys from LMDB file")
            with lmdb.open(self.lmdb_file) as env:
                with env.begin() as txn:
                    with txn.cursor() as cursor:
                        keys = [key.decode() for key in 
                                cursor.iternext(values=False)]
            self.lmdb_keys = keys
           
    def make_key_batches(self, batch_size = None):
            
            try:
                if self.lmdb_keys is None:
                    self.get_lmdb_keys()
            except:
                self.get_lmdb_keys()
                
            try:
                if self.key_batches is not None:
                    return 
            except:
                pass
            
            if batch_size is not None:
                self.batch_size = batch_size
            if self.batch_size is None:
                raise InputError("No batch size provided")
            
            print("Making batches of keys")
            shuffle(self.lmdb_keys)
            self.key_batches = batchize(self.lmdb_keys, self.batch_size)

    def read_lmdb(self, selected_keys = None, read_as_dict = False):
            if selected_keys is None:
                self.get_lmdb_keys()
                selected_keys = self.lmdb_keys
        
            with lmdb.open(self.lmdb_file) as env:
                with env.begin() as txn:
                    
                    # Read bitstrings from lmdb
                    if read_as_dict:
                        self.read_images = {}
                        for key in selected_keys:
                            lmdb_image = txn.get(key.encode())
                            self.read_images[key] = np.fromstring(lmdb_image, 
                                      dtype=np.uint8).reshape(self.img_config['input_shape'])
                           
                    else:
                        self.read_images = []
                        for key in selected_keys:
                            lmdb_image = txn.get(key.encode())
                            try:
                                self.read_images.append(np.fromstring(lmdb_image, 
                                dtype=np.uint8).reshape(self.img_config['input_shape']))
                            except:
                                try: 
                                    self.read_images.append(np.fromstring(lmdb_image, 
                                    dtype=np.int8).reshape(self.img_config['input_shape']))
                                except:
                                    self.read_images.append(np.fromstring(lmdb_image, 
                                    dtype=np.float64).reshape(self.img_config['input_shape']))
                                   
    def read_lmdb_batch(self, keys, num_classes = None, multitask_binary = False):
            #if type(keys) is not list:
            #    keys = [keys]    # Prevent iterating through a string

            # Read the images from LMDB
            self.read_lmdb(keys)
            X_batch = np.array(self.read_images)

            # Assign the labels for each image in one-hot format
            labels = [int(k.split(':')[0]) for k in keys]
            
            if multitask_binary == True:
                Y_batch = mult_task_label(labels)
            else:
                Y_batch = one_hot(labels, num_classes)

            return X_batch, Y_batch

                                                        
class TF_lmdb_set(TF_lmdb):
    """
    TF_data::TF_data_set
    Data class for combining many different TF_data objects.
    This class should not directly read image data from disk. That should happen 
    before merging.

    Inputs:
    TF_data_objs: A list of TF_data objects that have been initialized and have
    labels assigned to filenames

    Attributes:
    """
    def read_data(self):
        
        print('Mergining input datasets to create '+self.dataset_name)
        for i, TF_data_obj in enumerate(self.TF_data_objs):
            print('Reading object %d' % i)
            for key, value in TF_data_obj.__dict__.items():
                if value is str: 
                    continue
                if key is 'set_mean':
                    continue
                if i == 0:
                    setattr(self, key, value)
                else:
                    attr = getattr(self, key)
                    if (key is 'img_config'):
                        print(key + '  :No update needed')
                        continue
                    
                    if type(value) is dict:
                        print('Updating %s' % key)
                        attr = dict(attr)
                        attr.update(value)
                    elif type(value) is list:
                        print('Updating %s' % key)
                        attr = list(attr)
                        attr.extend(value)
                    elif type(value) is np.ndarray:  
                        print('Updating %s' % key)
                        attr = np.array(value)
                        try:
                            attr = np.stack([attr, value])
                        except:
                            value = np.stack([value])
                            attr = np.concatenate([attr, value],0)
                    else: continue;
                        
                    setattr(self, key, attr)
          
        try:
            if self.mean is not None:
                if len(self.mean.shape) > 3:
                    self.set_mean = np.mean(self.mean,0)
                else:
                    self.set_mean = self.mean
                np.save(self.dataset_name + '_mean.npy',
                                                   self.set_mean)
        except:
            pass
            
    def create_image_IDs(self, label_dict, cols = False):
        print("Warning: Ignoring function create_image_IDs()")
        print("Not overwriting pre-existing image IDs")
        pass

class InputError(Exception):
    """Exception raised for errors in the input.
    """
    
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
