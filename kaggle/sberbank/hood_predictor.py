'''
Created on May 3, 2017

@author: ikaro

'''
import pandas as pd
import numpy as np
import tensorflow as tf
from preprocess import get_hood_factors,load_hood_vals, data_dir, remove_eco_trend
from time import time
import tempfile
import matplotlib.pyplot as plt

LABEL_COLUMN='price_doc'
train_hood, test_hoods= load_hood_vals()
local=[x for x in get_hood_factors() if x in  train_hood.columns]
train_hood=train_hood[local]
print train_hood.price_doc.mean()
train_hood=remove_eco_trend(train_hood)
print train_hood.price_doc.mean()

Corr=train_hood.corr(min_periods=15)[LABEL_COLUMN]
keep =[x for x in Corr.index]# if abs(Corr[x]) > 0]
#keep.append('timestamp')
print Corr[keep]
df_train=train_hood[keep]
keep.remove('price_doc')
df_test=test_hoods[keep]
tf.logging.set_verbosity(tf.logging.ERROR)  

df_test=df_test.set_index('timestamp',drop=False).sort_index()
df_train=df_train.set_index('timestamp',drop=False).sort_index()

    
def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(np.ravel(df[k].values)) for k in df.columns if k !=LABEL_COLUMN}
  feature_cols = dict(continuous_cols.items())
  if(LABEL_COLUMN in df.columns):
      # Converts the label column into a constant Tensor.
      label = tf.constant(df[LABEL_COLUMN].values)
  else:
      label=[]
  return feature_cols, label

def train_input_fn():
  foo=df_train#.sample(n=47,axis=0,random_state=0)
  print "Training shape=%s "%str(foo.shape)
  return input_fn(foo)

def eval_input_fn():
  return input_fn(df_train)

def test_input_fn():
  print "Training shape=%s "%str(df_test.shape)
  return input_fn(df_test)
  
def build_estimator(CONTINUOUS_COLUMNS):
  """Build an estimator."""
  #Categorical variables ( need to be int!)
  cat_var=list()
  for cat in CONTINUOUS_COLUMNS:
      if cat !=LABEL_COLUMN:
          cat_var.append(tf.contrib.layers.real_valued_column(column_name=cat))
  model_dir = tempfile.mkdtemp()
  model = tf.contrib.learn.LinearRegressor(feature_columns=cat_var,model_dir=model_dir)
  return model  
  
def run():
    
    start=time()
    model=build_estimator(df_train.columns)
    model.fit(input_fn=train_input_fn, steps=2000)
    results = model.evaluate(input_fn=eval_input_fn, steps=1)#,metrics={'accuracy':tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.accuracy)})
    for key in sorted(results):
        print("Results %s: %s" % (key, results[key]))
    yhat=model.predict_scores(input_fn=eval_input_fn,as_iterable=False)
    yhat_test=model.predict_scores(input_fn=test_input_fn,as_iterable=False)
    plt.figure()
    plt.scatter(yhat,df_train['price_doc'].values)
    #plt.plot(df_train['timestamp'].values,df_train['price_doc'].values-yhat,'ok-')
    #plt.plot(df_test['timestamp'].values,yhat_test,'-go')
    
    file_name=data_dir+"predicted_hood_values.csv"
    #tm=np.concatenate((df_train['timestamp'].values,df_test['timestamp'].values),axis=0)
    #price=np.concatenate((yhat,yhat_test),axis=0)
    #price=price-price[0]
    #plt.figure()
    #plt.plot(tm,price,'o-')
    plt.show()
    #pred=pd.DataFrame(data={'hood_price_hat':price},index=tm)
    #pred.to_csv(file_name)
    print("Done training in %s minutes. Saved predictions to %s "%((time()-start)/60, file_name))
    
    


if __name__ == '__main__':
    run()

