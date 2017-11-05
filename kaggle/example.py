'''
Created on Nov 5, 2017

@author: ikaro
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

HOUSING_PATH="./"
rooms_ix, bedrooms_ix, population_ix, household_ix =3,4,5,6

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
        
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:,rooms_ix] / X[:,household_ix]
        populations_per_household = X[:,population_ix] / X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,populations_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,populations_per_household]
        
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())
    
def save_model(model,model_name='foo.pkl'):
    joblib.dump(model,model_name)

def load_model(model_name):
    return joblib.load(model_name)


housing = load_housing_data()
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

housing.hist(bins=50,figsize=(20,15))
plt.show()

train_set,test_set = train_test_split(housing,test_size=0.2, random_state=42)
housing["income_cat"]= np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["median_income"]<5,5.0,inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set= housing.loc[test_index]
        
housing["income_cat"].value_counts()/len(housing)

for mset in (strat_train_set, strat_test_set):
        mset.drop(["income_cat"],axis=1,inplace=True)
        
housing = strat_train_set.copy()

housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4, 
             s=housing["population"]/100,label='population',
             c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)

imputer=Imputer(strategy="median")
housing_num=housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

X=imputer.transform(housing_num)
housing_tr= pd.DataFrame(X,columns=housing_num.columns)

encoder =LabelEncoder()
housing_cat= housing["ocean_proximity"]
housing_cat_encoded =encoder.fit_transform(housing_cat)
encoder = OneHotEncoder()
housing_cat_1hot= encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
encoder=LabelBinarizer()
housing_cat_1hot= encoder.fit_transform(housing_cat)

attr_adder =CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

num_pipeline= Pipeline([
                        ('imputer',Imputer(strategy='median')),
                        ('attribs_adder',CombinedAttributesAdder()),
                        ('std_scaler',StandardScaler()),])

housing_num_tr= num_pipeline.fit_transform(housing_num)

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

num_pipeline= Pipeline([('selector',DataFrameSelector(num_attribs)),
                        ('imputer',Imputer(strategy='median')),
                        ('attribs_adder',CombinedAttributesAdder()),
                        ('std_scaler',StandardScaler()),])

cat_pipeline= Pipeline([('selector',DataFrameSelector(cat_attribs)),
                        ('label_binarizer',LabelBinarizer()),])

full_pipeline = FeatureUnion(transformer_list=[
                            ("num_pipeline",num_pipeline),
                            ("cat_pipeline",cat_pipeline),
                            ])

housing_prepared = full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
some_data = housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t ",lin_reg.predict(some_data_prepared))

tree_reg=DecisionTreeRegressor()
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
display_scores(scores)
lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
display_scores(lin_scores)
forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
forest_scores = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
display_scores(forest_scores)

param_grid=[
            {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
            {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
            ]
forest_reg=RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring="neg_mean_squared_error")
print("Doing crossfold validation...")
grid_search.fit(housing_prepared,housing_labels)
print("Done crossfold validation")
grid_search.best_params_
grid_search.best_estimator_

cvres=grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
        print(np.sqrt(-mean_score),params)
        
final_model=grid_search.best_estimator_
X_test= strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared=full_pipeline.transform(X_test)
final_predictions=final_model.predict(X_test_prepared)
final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print("Final rmse=%s"%final_rmse)
print("Done!!")