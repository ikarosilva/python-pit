'''
Created on May 3, 2017

@author: ikaro

'''
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_format_data
from time import time
import tempfile

tf.logging.set_verbosity(tf.logging.ERROR)  
df_train,df_test, LABEL_COLUMN=load_format_data()
CATEGORICAL_COLUMNS = df_train.select_dtypes(['category']).columns
CONTINUOUS_COLUMNS=df_train.select_dtypes(['float64','int64']).columns

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(np.ravel(df[k].values)) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor. #rent_price_4+room_bus
  #categorical_cols = {k: tf.constant(df[k].values) for k in CATEGORICAL_COLUMNS}
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=[np.ravel(df[k].values)],
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)
   
def build_estimator():
  """Build an estimator."""
  print("%s Categorical colums:"%len(CATEGORICAL_COLUMNS))
  print(" ".join([x for x in CATEGORICAL_COLUMNS]))
  print("%s Continous colums:"%len(CONTINUOUS_COLUMNS))
  print(" ".join([x for x in CONTINUOUS_COLUMNS]))
    
  #Categorical variables ( need to be int!)
  cat_var=list()
  for cat in CATEGORICAL_COLUMNS:
      cat_var.append(tf.contrib.layers.sparse_column_with_hash_bucket(cat, hash_bucket_size=df_train[cat].nunique()))
  cat_var=list()
  for cat in CONTINUOUS_COLUMNS:
      cat_var.append(tf.contrib.layers.real_valued_column(column_name=cat))
  
  # Wide columns and deep columns.
  #deep_columns = []# [tf.contrib.layers.embedding_column(workclass, dimension=8),]
  model_dir = tempfile.mkdtemp()
  model = tf.contrib.learn.LinearRegressor(feature_columns=cat_var,model_dir=model_dir)

#   m = tf.contrib.learn.DNNLinearCombinedClassifier(
#     linear_feature_columns=wide_columns,
#     dnn_feature_columns=deep_columns,
#     dnn_hidden_units=[100, 50],
#     fix_global_step_increment_bug=True,
#     config=config)

  return model  

def run():
    
    
    start=time()
    model=build_estimator()
    '''
      Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      * for each `column` in `feature_columns`:
    if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`
    '''
    model.fit(input_fn=train_input_fn, steps=800)
    print("Done training in %s minutes"%((time()-start)/60))
    results = model.evaluate(input_fn=eval_input_fn, steps=1,metrics={'accuracy':tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.accuracy)})
    print("oi")
    for key in sorted(results):
        print("Results %s: %s" % (key, results[key]))
    print("Results")
    

    #Create NN
    #print df_train.dtypes
    #print df_train.apply(lambda x: x.nunique())
       
    #print train_df.shape
    #m = build_estimator(cats)
    #m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    #results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    #for key in sorted(results):
    #    print("%s: %s" % (key, results[key]))
    
    #TODO: Loop to measure capacity as function of hyperparameters
    
    


if __name__ == '__main__':
    run()

