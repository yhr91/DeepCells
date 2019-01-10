<h1>DeepCells</h1>

- <b>DeepCells</b> is an easy to use Python framework for training deep learning models using data generated from high content cellular microscopy images. It is able to automate metadata parsing and typical data pre-processing steps. It scales easily to very large datasets using LMDB.
- <b>TF_model</b>: Class for defining the tensorflow graph and for all related interactions including training, inference
- <b>TF_data</b> - Class for handling data manipulation, metadata, file I/O 
- <b>TF_ops</b> - Tensorflow function call wrappers
- A sample graph that inherits from TF_model is provided under the models directory

DeepCells was originally created by Yusuf Roohani while working at GlaxoSmithKline LLC

