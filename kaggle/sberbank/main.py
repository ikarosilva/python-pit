'''
Created on May 3, 2017

@author: ikaro

'''
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_format_data
train_steps=200

def build_estimator(categories):
  """Build an estimator."""
  #Categorical variables ( need to be int!)
  cat_var=list()
  for cat in categories:
      #cat_var.append(tf.contrib.layers.sparse_column_with_keys(column_name=cat['name'],keys=cat['keys']))
      cat_var.append(tf.contrib.layers.real_valued_column(column_name=cat['name']))
  wide_columns = cat_var
  
  # Wide columns and deep columns.
  #deep_columns = []# [tf.contrib.layers.embedding_column(workclass, dimension=8),]

  m = tf.contrib.learn.LinearClassifier(feature_columns=wide_columns)
#   m = tf.contrib.learn.DNNLinearCombinedClassifier(
#     linear_feature_columns=wide_columns,
#     dnn_feature_columns=deep_columns,
#     dnn_hidden_units=[100, 50],
#     fix_global_step_increment_bug=True,
#     config=config)

  return m  

def run():
    
    df_train,df_test, labels,cats=load_format_data()
    CATEGORICAL_COLUMNS = df_train.select_dtypes(['category']).columns
    print("Categorical colums=%s"%str(CATEGORICAL_COLUMNS))
    CONTINUOUS_COLUMNS=df_train.select_dtypes(['float']).columns
    #Create NN
    print df_train.dtypes
    #print df_train.apply(lambda x: x.nunique())
       
    #print train_df.shape
    print "Total categories=%s of %s"%(len(cats),train_df.shape[1])
    print labels.plot()
    plt.show()
    #m = build_estimator(cats)
    #m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    #results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    #for key in sorted(results):
    #    print("%s: %s" % (key, results[key]))
    
    #TODO: Loop to measure capacity as function of hyperparameters
    
    


if __name__ == '__main__':
    run()

