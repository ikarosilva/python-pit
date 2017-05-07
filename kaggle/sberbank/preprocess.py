#Define converter methods
from numpy import NaN
import matplotlib.pyplot as plt
LABEL_COLUMN='work_all'
CAT_TH=20 #If number of unique values is below this, make variable a category
from math import log
import pandas as pd
from time import time
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

def load_data():
    data_dir="/mnt/ssd/kaggle/sberbank/"
    input_converters={'timestamp':timestamp}
    macro_converts={'timestamp':timestamp}
    train_df = pd.read_csv(data_dir+ 'train.csv',converters=input_converters)
    macro = pd.read_csv(data_dir+ 'macro.csv',converters=macro_converts)   
    return train_df, macro

def load_format_data():
    start=time()
    train_df, macro= load_data()
    print("Lodaded data in %s seconds"%(time()-start))
    
    start=time()
    #Convert string sets to categories
    train_cat_cols=train_df.select_dtypes(['object']).columns
    for ct in train_cat_cols:
        train_df[ct]=train_df[ct].astype('category')
    train_df[train_cat_cols] = train_df[train_cat_cols].apply(lambda x: x.cat.codes)
    macro_cat_cols=macro.select_dtypes(['object']).columns
    for ct in macro_cat_cols:
        macro[ct]=macro[ct].astype('category')
    macro[macro_cat_cols] = macro[macro_cat_cols].apply(lambda x: x.cat.codes)
    train_df=train_df.join(macro.set_index('timestamp'),on='timestamp', how='left', lsuffix='input', rsuffix='macro',sort=False)
    print("Coded and Joined training and macro data in %s seconds"%(time()-start))
    
    #Use surrogate labels for now
    target=train_df['price_doc'].sample(frac=1)
    train_df['price_doc']=target.values
    
    #Impute missing values
    #Categorize any variable below threshold count of unique values
    start=time()
    ucount= train_df.apply(lambda x: len(x.unique()))
    colnames=ucount[ucount<CAT_TH].axes[0]
    for col in colnames:
        train_df[col]=train_df[col].astype('category')
        code_map={v: k for k, v in dict(enumerate(train_df[col].cat.categories)).iteritems()}
        train_df[col] = train_df[col].apply(lambda x: code_map[x])
        #train_df[col]=train_df[col].astype(int)
        train_df[col]=train_df[col].astype('category')
    
    print("Imputed missing values in %s seconds"%(time()-start))
    start=time()
    labels=train_df[['timestamp',LABEL_COLUMN]]
    #labels=train_df.groupby('timestamp')[LABEL_COLUMN].aggregate(np.mean).reset_index()
    train_df=train_df.drop(LABEL_COLUMN, 1)

    macro=list(macro.columns)
    for c in train_df.columns:
        if(c not in macro):
            train_df=train_df.drop(c, 1)
    # remove NaN elements
    Nt=int(24376*0.8)
    df_train=train_df[:Nt]
    df_test=train_df[Nt:]
    
    cats=[]
    for cl  in df_train.columns:
        if cl in macro:
            num=df_train[cl].nunique()
            if(num<20 and str(df_train[cl].dtype).startswith('int')):
                pdf=df_train[cl].value_counts(normalize=True)
                ent=- sum([ p * log(p) / log(2.0) for p in pdf ])
                #print ("Adding category %s : %s ent=%0.1f %s"%(cl,num,ent,df_train[cl].dtype))
                tmp={'name':cl,'keys':df_train[cl].unique()}
                cats.append(tmp)
    
    train_df=train_df[[x['name'] for x in cats]]
    print("Formatted data in %s seconds"%(time()-start))
    return df_train,df_test, labels,cats



#timestamp=lambda x: int(x[:4]+x[5:7]+x[8:10])
timestamp=lambda x: int(x[:4]+x[5:7])