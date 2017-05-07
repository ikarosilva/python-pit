'''
Created on May 4, 2017

@author: ikaro
'''
from numpy import delete
from numpy.random import permutation
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def kmeans_sampling(X_train,N_balance, n_clusters):
    N_normals = len(X_train)
    # Define number of points per cluster
    K = N_balance / n_clusters
    clf = KMeans(n_clusters=n_clusters, n_jobs=3, max_iter=50000)
    clf.fit(X_train)
    paired_ind = [x for x in enumerate(clf.labels_)]
    rand_ind = permutation(N_normals)
    pick_class = range(n_clusters) * K
    keep_ind = []
    for ind in rand_ind:
        if(paired_ind[ind][1] in pick_class):
            keep_ind.append(paired_ind[ind][0])
            pick_class.remove(paired_ind[ind][1])
        if(len(pick_class) == 0):
            break
    remove_ind = [ind for ind,x in enumerate(X_train) if ind not in keep_ind]
    x_bal=delete(X_train, remove_ind, axis=0)
    return x_bal

def xgbboost_sample_check(xdat):
    '''
        From Konrad Banachewicz in 
        https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms
    '''
    y = xdat['istrain']; xdat.drop('istrain', axis = 1, inplace = True)
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44)
    xgb_params = {'learning_rate': 0.05, 'max_depth': 4,'subsample': 0.9,
            'colsample_bytree': 0.9,'objective': 'binary:logistic',
            'silent': 1, 'n_estimators':100, 'gamma':1,
            'min_child_weight':4
            }
    
    clf = xgb.XGBClassifier(**xgb_params)#, seed = 10)
    
    for train_index, test_index in skf.split(xdat, y):
            x0, x1 = xdat.iloc[train_index], xdat.iloc[test_index]
            y0, y1 = y.iloc[train_index], y.iloc[test_index]        
            clf.fit(x0, y0, eval_set=[(x1, y1)],
                   eval_metric='logloss', verbose=False,early_stopping_rounds=10)
                    
            prval = clf.predict_proba(x1)[:,1]
    
    return roc_auc_score(y1,prval)