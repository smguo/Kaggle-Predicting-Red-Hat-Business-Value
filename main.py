
#get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import datetime
import pylab
import matplotlib.pyplot as plt
import math

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.cross_validation import (LeaveOneOut,KFold,StratifiedKFold,LabelKFold,train_test_split)
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import (mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier # Used for imputing rare / missing values
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn2 import XGBClassifier
# Regressors considered:
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge # only model used for final submission

import xgboost as xgb
import random
from operator import itemgetter
import time
import copy
import group_1_date_trick as gdt

#%% define xgboost object for passing to sci-kit learn function, not used in the end 

class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
#        self.early_stopping_rounds = early_stopping_rounds
        self.params = params
#        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
#        early_stopping_rounds = early_stopping_rounds or self.early_stopping_rounds
#        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, 
            num_boost_round=num_boost_round,
            verbose_eval=True)
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
        
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)
#%% impute the categories that only appear once with a general "rare" category
def reduce_dimen(dataset,column,toreplace):
    for index,i in dataset[column].duplicated(keep=False).iteritems():
        if i==False:
            dataset.set_value(index,column,toreplace)
    return dataset
#%% read train and test data
def read_test_train():

    print("Read people.csv...")
    people = pd.read_csv("../input/people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv("../input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test = pd.read_csv("../input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])
    return train, test, people                   

#%%
def process_table(train, test, people):
    print("Process tables...")
    for table in [train, test]:
#        table.drop('char_10',axis=1, inplace = True) # drop char_10
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        table['people_id'] = table['people_id'].str.lstrip('ppl_').astype(np.float).astype(np.int32)
        for i in range(1, 11): # include char_10
            table['char_' + str(i)].fillna('type 999', inplace=True)
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    people['people_id'] = people['people_id'].str.lstrip('ppl_').astype(np.float).astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int8)
    min_max_scaler = MinMaxScaler()
    people['char_38'] = min_max_scaler.fit_transform(people['char_38']).reshape(-1, 1)
#    people['char_38'] = people['char_38'].apply(np.log)
#    for i in range(1, 10):
#        people = people.join(pd.get_dummies(people['char_' + str(i)], prefix='char_' + str(i)))
#        people = people.drop(['char_' + str(i)], axis=1)
#        people = people.drop(['char_' + str(i) +'_1'], axis=1)
    print("Merge...")
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(999, inplace=True)
#    for table in [train, test]:     
#        table['date_x_prob'] = table.groupby('date_x')['outcome'].transform('mean')
#        table['date_y_prob'] = table.groupby('date_y')['outcome'].transform('mean')
#        table['date_x_count'] = table.groupby('date_x')['outcome'].transform('count')
#        table['date_y_count'] = table.groupby('date_y')['outcome'].transform('count')
    
    for table in [train, test]: 
        table['year_x'] = table['date_x'].dt.year
        table['month_x'] = table['date_x'].dt.month
        table['day_x'] = table['date_x'].dt.day
        table['year_y'] = table['date_y'].dt.year
        table['month_y']= table['date_y'].dt.month
        table['day_y'] = table['date_y'].dt.day
        table['isweekend_x'] = (table['date_x'].dt.weekday >= 5).astype(int)
        table['isweekend_y'] = (table['date_y'].dt.weekday >= 5).astype(int)
        table.drop('date_x', axis=1, inplace=True)
        table.drop('date_y', axis=1, inplace=True)
        
    print('Construct features...')
#    train = train.sort_values(['people_id'], ascending=[1])
    train_y = train['outcome'].as_matrix()
    
    train=train.drop('outcome',axis=1)
    
    return train, test, train_y, people 
#%%        
def feature_engineer(train, test):
    train_test=pd.concat([train,test],ignore_index=True)
    categorical=['group_1','activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_10_x','char_1_y','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
        
    for category in categorical:
        train_test=reduce_dimen(train_test,category,9999999)    
    
    train=train_test[:len(train)]
    test=train_test[len(train):]
#    train['outcome'] = train_y.values
#    features = get_features(train, test)
    train.drop('activity_id', axis = 1, inplace =True)
    test.drop('activity_id', axis = 1, inplace =True)
    train.drop('people_id', axis = 1, inplace =True)
    test.drop('people_id', axis = 1, inplace =True)
    categorical=['group_1','activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_10_x','char_1_y','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
    not_categorical= difference(train.columns.values,categorical)    
#    print("One Hot enconding categorical fea1tures...")
#    n_cat_col = len(categorical)*2
#    i = 0
#    for table in [train, test]:
#        for col in categorical:              
#            table = table.join(pd.get_dummies(table[col], prefix=col,sparse=True))
#            table = table.drop(col, axis=1)
#            table = table.drop([col +'_1'], axis=1)
#            i = i+1
#            print('progress {}%'.format(i/n_cat_col*100))
#        table = table.to_sparse(fill_value=0)
    
#    enc = OneHotEncoder(handle_unknown='ignore')
#    enc=enc.fit(pd.concat([train[categorical],test[categorical]]))
#    train_sparse=enc.transform(train[categorical])
#    test_sparse=enc.transform(test[categorical])
#    
#    from scipy.sparse import hstack
#    train_sparse=hstack((train[not_categorical].astype(float), train_sparse))
##    not_categorical= difference(categorical, ['outcome'])    
#    test_sparse=hstack((test[not_categorical].astype(float), test_sparse))
#    train_sparse = train_sparse.tocsr()
#    test_sparse = test_sparse.tocsr()


#    print("Training data: " + format(train_sparse.shape))
#    print("Test data: " + format(test_sparse.shape))
    
    
#    features = not_categorical+categorical
    train_sparse = train
    test_sparse = test
    features = get_features(train, test)
    return train_sparse, test_sparse, features
#%% 
def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
#    output.remove('people_id')
#    output.remove('activity_id')
    return sorted(output)

#%%
random.seed(10)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))
    
def difference(a, b):
    return list(set(a) - set(b))
    
def run_ridge(train, test, features, target):
    X = train[features]
    y = train[target]
      
    model_grid = [{'normalize': [True, False], 'alpha': np.logspace(0,10, num = 10)}]
    model = Ridge()
    
    # Use a grid search and leave-one-out CV on the train set to find the best regularization parameter to use.
    # (might take a minute or two)
#    grid = GridSearchCV(model, model_grid, cv=LeaveOneOut(len(y)), scoring='mean_squared_error')
    grid = GridSearchCV(model, model_grid, cv=5, scoring='mean_squared_error', verbose = 5, n_jobs = 10)
    grid.fit(X, y)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    
    # Re-train on full training set using the best parameters found in the last step.
    model.set_params(**grid.best_params_)
    model.fit(X, y)
    check = model.predict(X)
    test_prediction = model.predict(test)
    score = roc_auc_score(y.values, check)
    print('Check error value: {:.6f}'.format(score))
    return test_prediction.tolist(), score

def run_xbg_cv(dtrain, target, features,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    alg = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=1,
     scale_pos_weight=1,
     seed=27)
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[features].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,seed = 0)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[features], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[features])
    dtrain_predprob = alg.predict_proba(dtrain[features])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
#%%    
def tune_xgb_para(train, features, target):
    param_test1 = {
     'max_depth':np.arange(3,10,2),
     'min_child_weight':np.arange(1,6,2)
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.2, n_estimators=150, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=10,), 
     param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(train[features],train[target])
    return gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#%%
def tune_xgb_linear_para(train, train_y):
    param_test1 = {
     'reg_lambda': np.logspace(-1,0, num=2),
     'reg_alpha':np.logspace(-1,0, num=2)
    }
#    clf = XGBoostClassifier(num_boost_round=200, eta= 0.02,
#     booster = "gblinear", eval_metric = 'auc',
#     objective= 'binary:logistic', num_class = 2, silent = 1, seed=10,
#     verbose_eval=True)
    clf = XGBClassifier(booster = 'gblinear',learning_rate =0.2, n_estimators=250, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=10,) 
     
    gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1,
        scoring='roc_auc',iid=False, cv=2,n_jobs = 1,verbose=100)
    gsearch1.fit(train,train_y)
    return gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#%%
def run_single_xgb_tree(train, test, features, train_y, random_state=0):
    eta = 0.2
    eta_list  = [1]*80 +[0.2]*100 
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
   
    num_boost_round = 180
    early_stopping_rounds = 10
    test_size = 0.1

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train, y_valid = train_test_split(train_y, test_size=test_size, random_state=random_state)
#    print('Length train:', len(X_train.index))
#    print('Length valid:', len(X_valid.index))
#    y_train = X_train[target]
#    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, learning_rates = eta_list, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    print("Best interation: ", str(gbm.best_iteration))
    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_valid, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score
#%%
    
def run_single_xgb_linear(train, test, features, train_y, random_state=0):
    params = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic' }
    #params['nthread'] = 4
    params['eval_metric'] = 'auc'
    params['subsample'] = 0.7
    params['colsample_bytree']= 0.7
    params['min_child_weight'] = 0
    params['booster'] = "gblinear"    
    params['seed'] = 0
    
    eta_list  = [0.1]*30 +[0.02]*60 + [0.001]*80
    num_boost_round = 170
    early_stopping_rounds = 20
    test_size = 0.1
    start_time = time.time()

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train, y_valid = train_test_split(train_y, test_size=test_size, random_state=random_state)
#    print('Length train:', len(X_train.index))
#    print('Length valid:', len(X_valid.index))
#    y_train = X_train[target]
#    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, learning_rates = eta_list,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    print("Best interation: ", str(gbm.best_iteration))
    print("Validating...")
    check = gbm.predict(dvalid)
    score = roc_auc_score(y_valid, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test))

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score
#%%
def run_kfold(nfolds, train, test, test_raw,features, train_y,labels, random_state=0,
              reg_lambda=0, reg_alpha=0):
    eta = 0.02
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gblinear",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "silent": 1,
        "seed": random_state
    }
    eta_list  = [0.1]*30 +[0.02]*60 + [0.001]*80
#    eta_list  = [1]*80 +[0.2]*100 + [0.05]*100
    num_boost_round = 170
    early_stopping_rounds = 10

    yfull_train = dict()
    yfull_test = copy.deepcopy(test_raw[['activity_id']].astype(object))
#    kf = KFold(train.shape[0], n_folds=nfolds, shuffle=True, random_state=random_state)
#    kf = StratifiedKFold(train_y, n_folds=nfolds, shuffle=True, random_state=random_state)    
    kf = LabelKFold(labels, n_folds=nfolds)    
    num_fold = 0
    for train_index, test_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        X_train, X_valid = train[train_index], train[test_index]
        y_train, y_valid = train_y[train_index], train_y[test_index]
        X_test = test

        print('Length train:', X_train.shape[0])
        print('Length valid:', X_valid.shape[0])

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,learning_rates = eta_list,
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        print("Best interation: ", str(gbm.best_iteration))
        print("Validating...")
        yhat = gbm.predict(xgb.DMatrix(X_valid))
        score = roc_auc_score(y_valid, yhat)
        print('Check error value: {:.6f}'.format(score))

        # Each time store portion of precicted data in train predicted values
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = yhat[i]

        imp = get_importance(gbm, features)
        print('Importance array: ', imp)

        print("Predict test set...")
        test_prediction = gbm.predict(xgb.DMatrix(X_test))
        yfull_test['kfold_' + str(num_fold)] = test_prediction

    # Copy dict to list
    train_res = []
    for i in sorted(yfull_train.keys()):
        train_res.append(yfull_train[i])

    score = roc_auc_score(train_y, np.array(train_res))
    print('Check error value: {:.6f}'.format(score))

    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds+1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return yfull_test['mean'].values, score
#%%
def run_kfold_tree(nfolds, train, test, test_raw,features,train_y, labels, random_state=0):
    eta = 1
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()
#    eta_list  = [0.8]*80 +[0.2]*100 + [0.05]*100
    eta_list  = [0.2]*300 
    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round = 300
    early_stopping_rounds = 20
    
    yfull_train = dict()
    yfull_test = copy.deepcopy(test_raw[['activity_id']].astype(object))
#    kf = KFold(len(train.index), n_folds=nfolds, shuffle=True, random_state=random_state)
    kf = LabelKFold(labels, n_folds=nfolds) 
    num_fold = 0
    for train_index, test_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]
        y_train, y_valid = train_y[train_index], train_y[test_index]
        X_test = test[features].as_matrix()

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, learning_rates = eta_list, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        
        print("Validating...")
        yhat = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.tolist(), yhat)
        print('Check error value: {:.6f}'.format(score))

        # Each time store portion of precicted data in train predicted values
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = yhat[i]

        imp = get_importance(gbm, features)
        print('Importance array: ', imp)

        print("Predict test set...")
        test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
        yfull_test['kfold_' + str(num_fold)] = test_prediction

    # Copy dict to list
    train_res = []
    for i in sorted(yfull_train.keys()):
        train_res.append(yfull_train[i])

    score = roc_auc_score(train_y, np.array(train_res))
    print('Check error value: {:.6f}'.format(score))

    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds+1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return yfull_test['mean'].values, score
#%%
def grid_search_kfold(nfolds, train, test, test_raw,features, train_y, labels, random_state=0):
     reg_lambda = np.logspace(-2,2, num=5)
     reg_alpha = np.logspace(-2,2, num=5)
     test_prediction = np.zeros(shape = (len(reg_lambda),len(reg_alpha),len(test_raw)))
     score = np.zeros(shape = (len(reg_lambda),len(reg_alpha)))
     for i in np.arange(0, len(reg_lambda)):
         for j in np.arange(0, len(reg_alpha)):             
             test_prediction[i,j,:], score[i,j] = run_kfold(3, train, test, test_raw, 
                features, train_y,labels, random_state = random_state,
             reg_lambda=reg_lambda[i], reg_alpha=reg_alpha[j])
             print(score)
     return test_prediction, score 
#%%
def create_submission(score, test, prediction):
    now = datetime.datetime.now()
    sub_file = '../submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('activity_id,outcome\n')
    total = 0
    for id in test['activity_id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()
#%% main script 
train_raw, test_raw, people_raw = read_test_train()
#people_raw.isnull().values.ravel().sum() # check count of missing values
#train_raw['char_10'].value_counts().plot(title="City Group Distribution in the Train Set", kind='bar')
#plt.show()
train_raw, test_raw, train_y, people_raw = process_table(train_raw, test_raw, people_raw)
train = train_raw.copy()
test =  test_raw.copy()
people = people_raw.copy()
train_sparse, test_sparse, features = feature_engineer(train, test)

print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
#train = train.iloc[1:500,:]
#test_prediction, score = run_ridge(train, test, features, 'outcome')
#test_prediction, score = run_single_xgb_linear(train_sparse, test_sparse, features, train_y, 0)
#test_prediction, score = run_single_xgb_tree(train_sparse, test_sparse, features, train_y, 0)
test_prediction, score = run_kfold_tree(5, train_sparse, test_sparse, test_raw, features, train_y,train_raw['people_id'].as_matrix(), random_state=0)
#test_prediction, score = grid_search_kfold(3, train_sparse, test_sparse, test_raw,features, train_y,train_raw['people_id'].as_matrix(),random_state=0)
#test_prediction, score = run_kfold(3, train_sparse, test_sparse, test_raw, features, train_y, train_raw['people_id'].as_matrix())
#grid_scores, best_params, best_score_ = tune_xgb_para(train, features, 'outcome')
#grid_scores, best_params, best_score =tune_xgb_linear_para(train_sparse, train_y)
#run_xbg_cv(train, 'outcome', features,useTrainCV=True, cv_folds=5, early_stopping_rounds=50)
#%% prediction of the leak model 
test_leak = gdt.group1_date_predict()
test_leak['group_1'] = test_leak['group_1'].str.lstrip('group ').astype(np.int32)
#%%
test_no_leak = test_raw.copy().loc[:,['group_1', 'activity_id']]
test_no_leak['outcome'] = test_prediction
#test_leak_keep = test_leak[test_leak['group_1'].isin(train_raw['group_1'])]
test_leak_keep_ind = test_leak['group_1'].isin(train_raw['group_1'])
test_leak_weight = test_leak_keep_ind.astype('float').as_matrix()*0.5 
#test_leak_disca = test_leak[~test_leak['group_1'].isin(train_raw['group_1'])]
#test_no_leak_keep = test_no_leak[~test_no_leak['group_1'].isin(np.unique(train_raw['group_1'].values))]
test_no_leak_keep_ind = ~test_no_leak['group_1'].isin(np.unique(train_raw['group_1'].values))
test_no_leak_disca_ind = test_no_leak['group_1'].isin(np.unique(train_raw['group_1'].values))
test_no_leak_weight = test_no_leak_keep_ind.astype('float').as_matrix()+ test_no_leak_disca_ind.astype('float').as_matrix()*0.5
#test_no_leak_disca = test_no_leak[test_no_leak['group_1'].isin(np.unique(train_raw['group_1'].values))]
#test_all=pd.concat([test_no_leak_keep, test_leak_keep],ignore_index=True)
#%% wieghted averaging of prediction from leak and no-leak models
test_all = test_no_leak.copy()
test_all['outcome'] = test_leak['outcome'].as_matrix()*test_leak_weight + test_no_leak['outcome'].as_matrix()*test_no_leak_weight
#%% Creat submission files
#create_submission(score, test_raw, test_prediction)
create_submission(score, test_all, test_all['outcome'].values)
