'''
Created on May 3, 2017

@author: ikaro

'''

import pandas as pd
import seaborn as sns
data_dir="/mnt/ssd/kaggle/sberbank/"
import numpy as np
import matplotlib.pyplot as plt
color = sns.color_palette()
# We start by loading the training / test data and combining them with minimal preprocessing necessary
# Most of the data preparation is taken from here: 
# https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
train_df = pd.read_csv(data_dir+ 'train.csv')
macro = pd.read_csv(data_dir+ 'macro.csv')
print train_df.dtypes

# 
# #train_df.set_index('timestamp', inplace=True) Reduces numbers?
# train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
# grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()
# median_price=grouped_df.price_doc.values
# sns.barplot(grouped_df.yearmonth.values, median_price, alpha=0.8, color=color[2])
# plt.ylabel('Median Price', fontsize=12)
# plt.xlabel('Year Month', fontsize=12)
# plt.xticks(rotation='vertical')
# plt.show()
# foo=train_df.groupby('yearmonth')
# foo.mean()['life_sq'].plot()
# print len(train_df.columns)
# train_df=train_df.join(macro, lsuffix='_input', rsuffix='_macro')
# print len(train_df.columns)
