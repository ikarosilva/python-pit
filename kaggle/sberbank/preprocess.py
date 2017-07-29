#Define converter methods
from numpy import NaN, ravel
import matplotlib.pyplot as plt
LABEL_COLUMN='price_doc'
CAT_TH=0 #If number of unique values is below this, make variable a category
from math import log
import pandas as pd
from time import time
import numpy as np
import seaborn as sns
data_dir="/mnt/ssd/kaggle/sberbank/"
    

def get_medians():
    
    train_df,_,_ = load_format_data("full_set.csv",full_set=True)
    grouped_df = train_df.groupby('timestamp').agg(np.nanmedian).reset_index()
    file_name=data_dir+"median_vals.csv"
    grouped_df.to_csv(file_name)
    print("Wrote %s rows of median price to %s"%(len(grouped_df),file_name))
    
    test_df,_,_ = load_format_data("test.csv",full_set=True)
    grouped_df =test_df.groupby('timestamp').agg(np.nanmedian).reset_index()
    file_name=data_dir+"test_median_vals.csv"
    grouped_df.to_csv(file_name)
    print("Wrote %s rows of median price to %s"%(len(grouped_df),file_name))

def get_hood():
    
    train_df,_,_ = load_format_data("full_set.csv",full_set=True,load_macro=False)
    train_df['hood']=train_df['timestamp'].apply(lambda x: str(x)) +train_df['sub_area'].apply(lambda x: "_"+ str(x))
    grouped_df = train_df.groupby('hood').agg(np.nanmedian).reset_index()
    file_name=data_dir+"hood_vals.csv"
    grouped_df.to_csv(file_name)
    print("Wrote %s rows of training hood data to %s"%(len(grouped_df),file_name))
    
    test_df,_,_ = load_format_data("test.csv",full_set=True,load_macro=False)
    test_df['hood']=test_df['timestamp'].apply(lambda x: str(x)) +test_df['sub_area'].apply(lambda x: "_"+ str(x))
    grouped_df = test_df.groupby('hood').agg(np.nanmedian).reset_index()
    file_name=data_dir+"test_hood_vals.csv"
    grouped_df.to_csv(file_name)
    print("Wrote %s rows of hood price to %s"%(len(grouped_df),file_name))
        
    
def code(x,bins):
    y=0
    for bn in bins:
        if(x<bn):
            break
        y+=1
    return y

def get_code(df,cname,graph):
    rg=[0.01,0.2,0.4,0.6,0.8,0.99]
    bins=[]
    for r in rg:
        bins.append(df[cname].quantile(r))
    if(graph):
        print sorted(df[cname].unique(),reverse=True)
        df[cname].hist(bins='auto')
        plt.show()
    return bins

def culture_objects_top_25(x):
    return 1 if x=='yes' else 0

def codex(x,bins):
    if(x=='NA'):
        x=NaN
    x=float(x)
    return code(x,bins)

def full_sq(x):
    return codex(x,[27,38,44,54,67,135])

def life_sq(x):
    return codex(x,[1.0,19.0,27.0,32.0,44.0,100.0])

def floor(x):
    return codex(x,[1.0, 3.0, 5.0, 8.0, 12.0, 23.0])

def max_floor(x):
    return codex(x,[0.0, 5.0, 10.0, 16.0, 17.0, 25.0])

def load_median_vals():
    input_converters={'timestamp':month2int}
    medians = pd.read_csv(data_dir+ 'median_vals.csv',converters=input_converters)
    z=medians.apply(lambda x: x.nunique())
    keep=z[z>15].index
    medians=medians[keep] #Min number of unique values in order to include
    
    test_medians = pd.read_csv(data_dir+ 'test_median_vals.csv',converters=input_converters)
    test_medians=test_medians[keep.drop('price_doc')] #Min number of unique values in order to include
    
    return medians,test_medians

def load_hood_vals():
    input_converters={'timestamp':month2int}
    hood = pd.read_csv(data_dir+ 'hood_vals.csv',converters=input_converters)
    z=hood.apply(lambda x: x.nunique())
    keep=z[z>15].index
    hood=hood[keep] #Min number of unique values in order to include
    
    test_hood = pd.read_csv(data_dir+ 'test_hood_vals.csv',converters=input_converters)
    test_hood=test_hood[keep.drop('price_doc')] #Min number of unique values in order to include
    
    return hood,test_hood

def get_economic_factors():
    eco_feat = pd.read_csv(data_dir+'eco_feat.txt',header=None)
    eco_feat=[x.replace("+","") for x in ravel(eco_feat.values)]
    return eco_feat

def get_hood_factors():
    eco_feat = pd.read_csv(data_dir+'neighboor_feat.txt',header=None)
    eco_feat=[x.replace("+","") for x in ravel(eco_feat.values)]
    return eco_feat

def load_data(file_name,load_macro):
    input_converters={'timestamp':timestamp}
    train_df = pd.read_csv(data_dir+ file_name,converters=input_converters)
    if(load_macro):
        macro_converts={'timestamp':timestamp}
        macro = pd.read_csv(data_dir+ 'macro.csv',converters=macro_converts)   
        macro=macro.rename(columns=lambda x: x.replace("+",""))
    else:
        macro=[]
    train_df=train_df.rename(columns=lambda x: x.replace("+",""))
    
    return train_df, macro

def load_format_data(file_name='sample.csv', full_set=False,use_surrogate=False,load_macro=True):
    start=time()
    train_df, macro= load_data(file_name,load_macro)
    print("Lodaded data in %s seconds"%(time()-start))
    
    #Fix data issues
    train_df=train_df.replace(to_replace={'state': {33: NaN}})
    train_df=train_df.replace(to_replace={'build_year': {20052009: NaN}})
    train_df=train_df.replace(to_replace={'full_sq': {2000: NaN}})
    #TODO: Figure this out ... mask=train_df[train_df['life_sq'] > train_df['full_sq']]
    #train_df.drop(mask)
    #mask=train_df[train_df['build_year'] > 2017 ] + train_df[train_df['build_year'] < 1800]
    #train_df.drop(mask)
    start=time()
    #Convert string sets to categories
    train_cat_cols=train_df.select_dtypes(['object']).columns
    for ct in train_cat_cols:
        train_df[ct]=train_df[ct].astype('category')
    train_df[train_cat_cols] = train_df[train_cat_cols].apply(lambda x: x.cat.codes)
    
    #Impute missing values
    train_df=train_df.apply(lambda x: x.fillna(x.mean()),axis=0)
    
    if(load_macro):
        macro_cat_cols=macro.select_dtypes(['object']).columns
        for ct in macro_cat_cols:
            macro[ct]=macro[ct].astype('category')
        macro[macro_cat_cols] = macro[macro_cat_cols].apply(lambda x: x.cat.codes)
        macro=macro.apply(lambda x: x.fillna(x.mean()),axis=0)
        train_df=train_df.join(macro.set_index('timestamp'),on='timestamp', how='left', lsuffix='input', rsuffix='macro',sort=False)
        print("Joined training and macro data in %s seconds"%(time()-start))
    
    #Use surrogate labels for now
    if(use_surrogate):
        target=train_df['price_doc'].sample(frac=1)
        train_df['price_doc']=target.values
    
    start=time()
    #Categorize any variable below threshold count of unique values
    ucount= train_df.apply(lambda x: len(x.unique()))
    colnames=ucount[ucount<CAT_TH].axes[0]
    for col in colnames:
        train_df[col]=train_df[col].astype('category')
        code_map={v: k for k, v in dict(enumerate(train_df[col].cat.categories)).iteritems()}
        train_df[col] = train_df[col].apply(lambda x: str(code_map[x]))
        #train_df[col]=train_df[col].astype(int)
        train_df[col]=train_df[col].astype(str)
    
    print("Categorized and imputed missing values in %s seconds"%(time()-start))
    start=time()
    if(full_set):
        Nt=len(train_df)
    else:
        Nt=int(len(train_df)*0.8)
    #print("Dropping rows with nans")
    #train_df=train_df.dropna(axis=1, how='all')
    #train_df=train_df.dropna(axis=0, how='any')
    df_train=train_df[:Nt]
    df_test=train_df[Nt:]
    print("Partitioned data in %s seconds train shape=%s"%(time()-start,str(df_train.shape)))
    return df_train,df_test, LABEL_COLUMN

def remove_eco_trend(df_train):
    #Remove economic factor
    eco_fact=pd.read_csv(data_dir+"predicted_median_values.csv")
    for index, row in df_train.iterrows():
        corrected=row['price_doc']-eco_fact.eco_price_hat[row['timestamp']]
        df_train.set_value(index,'price_doc', corrected)
    return df_train     

#timestamp=lambda x: int(x[:4]+x[5:7]+x[8:10])
def timestamp(x):
    tm=int(x[:4]+x[5:7])
    return tm   

def month2int(x):
    tm=str(x)
    offset=2011*12
    year=int(tm[:4])
    month=int(tm[4:])
    dt=year*12+month - offset
    return dt

if __name__ == '__main__':
    get_hood()