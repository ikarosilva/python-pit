'''
Created on Jul 29, 2017

@author: ikaro
'''
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.metrics import log_loss
N_CLASS=9

def get_training():

    training_variants_df = pd.read_csv("training_variants")
    training_text_df = pd.read_csv("training_text",sep="\|\|", engine='python', 
                                   header=None, skiprows=1, names=["ID","Text"])
    training_merge_df = training_variants_df.merge(training_text_df,left_on="ID",right_on="ID")
    
    
    #TODO: Consider using Gene and variations later...
    gene_encoder= LabelBinarizer(sparse_output=True)
    gene_hot= gene_encoder.fit_transform(training_merge_df['Gene'])
    variation_encoder= LabelBinarizer(sparse_output=True)
    variation_hot= variation_encoder.fit_transform(training_merge_df['Variation'])
    
    train ,test = train_test_split(training_merge_df,test_size=0.2,random_state=42) 
    np.random.seed(0)
    
    
    X_train = train['Text'].values
    X_test = test['Text'].values
    y_train = train['Class'].values
    y_test = test['Class'].values
    return X_train,X_test,y_train,y_test

def random_guess(X_test):
    np.random.seed(42)
    y_hat=np.random.randint(1,N_CLASS+1, size=(X_test.shape[0], 1))
    encoder= OneHotEncoder()
    return encoder.fit_transform(y_hat).toarray()

def mode_guess(X_test):
    y_hat=np.zeros((X_test.shape[0], N_CLASS))
    y_hat[:,6]=1
    return y_hat


def run():
    X_train,X_test,y_train,y_test=get_training()
    
    y_hat=random_guess(X_train)
    print log_loss(y_train,y_hat)
    
    y_hat=mode_guess(X_train)
    print log_loss(y_train,y_hat)
    
if __name__ == "__main__":
    run()