
#get_ipython().magic('matplotlib inline')
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.cross_validation import (LeaveOneOut,KFold,StratifiedKFold,LabelKFold,train_test_split)
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
import seaborn as sns

# Regressors considered:
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from operator import itemgetter
import time
import copy
from itertools import product
proj_dir = 'E:\\Google Drive\\Python\\RedHat\\scripts'
os.chdir(proj_dir)



#% impute the categories that only appear once with a general "rare" category
def reduce_dimen(dataset,column,toreplace):
    for index,i in dataset[column].duplicated(keep=False).iteritems():
        if i==False:
            dataset.set_value(index,column,toreplace)
    return dataset
#% read train and test data
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

#%
def process_table(train, test, people):
    print("Process tables...")
    for table in [train, test]:
#        table.drop('char_10',axis=1, inplace = True) # drop char_10 for now
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        table['people_id'] = table['people_id'].str.lstrip('ppl_').astype(np.float).astype(np.int32)
        for i in range(1, 11): 
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
    print("Merge...") #join the table
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(999, inplace=True)
    first_act_date= min([train['date_x'].min(), test['date_x'].min(), people['date'].min()]) 
#    last_act_date= table['date_x'].max()
    for table in [train, test]: #                
        table['act_days'] = table['date_x'] - first_act_date        
        table['act_days'] = table["act_days"].dt.days + 1 #
        table['ppl_days'] = table['date_y'] - first_act_date        
        table['ppl_days'] = table['ppl_days'].dt.days + 1 #
        table['isweekend_x'] = (table['date_x'].dt.weekday >= 5).astype(int)
        table['isweekend_y'] = (table['date_y'].dt.weekday >= 5).astype(int)
        table.drop('date_x', axis=1, inplace=True)
        table.drop('date_y', axis=1, inplace=True)                 
    return train, test, people 
#%        
def feature_engineer(train, test):
    print('Construct features...')
    train_outcome = train['outcome']   
    train=train.drop('outcome',axis=1)
    train_test=pd.concat([train,test],ignore_index=True)
    categorical=['activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x',
                 'char_1_y','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
        
    for category in categorical:
        train_test=reduce_dimen(train_test,category,9999999) #Group all the categories that only shows up once into a big category   
    
    train=train_test[:len(train)]
    test=train_test[len(train):]
    train['outcome'] = train_outcome.values
#    features = get_features(train, test)
    train.drop('activity_id', axis = 1, inplace =True)
    test.drop('activity_id', axis = 1, inplace =True)
    train.drop('people_id', axis = 1, inplace =True)
    test.drop('people_id', axis = 1, inplace =True)
#    categorical=['activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_1_y','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
    
    print("One Hot enconding and aggregating categorical features...")
    n_cat_col = len(categorical)
    i = 0
#    train_agg = train[['group_1','act_days','','outcome']]
    not_categorical= difference(train.columns.values,categorical)    
    train_agg = train[not_categorical]
    train_agg = train_agg.groupby(['group_1','act_days']).mean().add_prefix('mean_').reset_index()
    not_categorical= difference(test.columns.values,categorical)    
    test_agg = test[not_categorical]
    test_agg = test_agg.groupby(['group_1','act_days']).mean().add_prefix('mean_').reset_index()
#    test_agg = test_agg[['group_1','act_days']]
    for col in categorical: #             
        col_dum = pd.concat([train_test[['group_1','act_days']],pd.get_dummies(train_test[col], prefix=col)], axis = 1)
        train_col_dum=col_dum[:len(train)]
        test_col_dum=col_dum[len(train):]
        train_col_agg = train_col_dum.groupby(['group_1','act_days']).mean().add_prefix('mean_').reset_index()
        test_col_agg = test_col_dum.groupby(['group_1','act_days']).mean().add_prefix('mean_').reset_index()          
        train_agg =  pd.concat([train_agg, train_col_agg.iloc[:,2:]], axis = 1)
        test_agg =  pd.concat([test_agg, test_col_agg.iloc[:,2:]], axis = 1)           
        i = i+1
        print('progress {}%'.format(i/n_cat_col*100))
#    train_agg = pd.concat([train_agg,pd.get_dummies(train_agg['group_1'], prefix='group_1', sparse=True)], axis = 1)
#    train_agg = train_agg.drop(['group_1'])       
    return train_agg, test_agg
#% 
def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
#    output.remove('people_id')
#    output.remove('activity_id')
    return sorted(output)

#%
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

#%
def run_single_xgb_tree(train, test, features, target, random_state=0):
    eta = 0.2
    eta_list  = [1]*80 +[0.2]*100 
    max_depth = 100
    subsample = 1
    colsample_bytree = 1
    min_child_w = 0
    start_time = time.time()
    eta_list  = [0.5]*100+[0.1]*2900 
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
        "min_child_weight": min_child_w,
        "silent": 1,
        "seed": random_state,
    }
   
    num_boost_round = 3000
    early_stopping_rounds = 20
    test_size = 0.1

#    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
#    y_train, y_valid = train_test_split(train_y, test_size=test_size, random_state=random_state)
#    print('Length train:', len(X_train.index))
#    print('Length valid:', len(X_valid.index))
    X_train = train[features]    
    y_train = train[target]
#    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train, y_train)
#    dvalid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(dtrain, 'train')]
#    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, learning_rates = eta_list, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    print("Best interation: ", str(gbm.best_iteration))
    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_train), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_train, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score

#%
def run_kfold_tree(nfolds, train, test, features,target, label_cv=0, labels=0, random_state=0):
    eta = 0.2
    max_depth = 100
    subsample = 1
    colsample_bytree = 1
    min_child_w = 0
    start_time = time.time()
    eta_list  = [0.5]*100 +[0.1]*2900 
#    eta_list  = [0.1]*10000 
    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_w,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round =3000
    early_stopping_rounds = 20
    train_y = train[target]
    yfull_train = dict()
    yfull_test = copy.deepcopy(test[['group_1','act_days']].astype(object))
    if label_cv == 1 : # maker sure the cv folds don't share the same labels        
        kf = LabelKFold(labels, n_folds=nfolds)
    else:  
        kf = StratifiedKFold(train_y,n_folds=nfolds, shuffle=True, random_state=random_state)
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
#%
def tune_cls_para(alg, train, features, target, param_test, n_jobs=8):
    
    gsearch = GridSearchCV(estimator = alg,param_grid = param_test, 
        scoring='roc_auc',n_jobs=n_jobs,iid=False, verbose=3, 
        cv=StratifiedKFold(train[target], n_folds=5, shuffle=True))
    gsearch.fit(train[features],train[target]) 
    alg.set_params(**gsearch.best_params_)
    for score in gsearch.grid_scores_:
        print(score)         
    print('best CV parameters:')
    print(gsearch.best_params_)
    print('best CV score: %f' % gsearch.best_score_)    
    return alg
#%
def xgb_fit(alg, dtrain, predictors, target, cv=True, folds=None, cv_folds=5, early_stopping_rounds=20):       
    if cv:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], 
            folds=folds,nfold=cv_folds, verbose_eval=True,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, stratified=True)
        alg.set_params(n_estimators=cvresult.shape[0])
        test_score = cvresult['test-auc-mean'].iloc[-1]
        print('# of estimators: %d' % cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], verbose=True, eval_metric='auc')
#    train_yhat = alg.predict_proba(dtrain[predictors])[:,1]
    train_yhat = alg.predict(dtrain[predictors])
#    train_score = roc_auc_score(dtrain[target].tolist(), train_yhat)
    train_score = metrics.accuracy_score(dtrain[target].tolist(), train_yhat)
    if cv:
        score=test_score
        print("AUC Score (Test): %f" % score)
    else:
        score=train_score
#        print("AUC Score (Train): %f" % score)
        print("Accuracy : %.4g" % score)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return alg, score

def build_leak_model(train, features):
    group_1 = train['group_1'].unique()
    model = np.empty(len(group_1),dtype=object)
    score = np.zeros(len(group_1))
    ind = 0
    for group in group_1:
        train_one_group = train[train['group_1']==group]
        xgb_leak = XGBClassifier(     
             learning_rate=0.1,
             n_estimators=10,
             max_depth=100,
             min_child_weight=0,
             gamma=0,
             subsample=1,
             colsample_bytree=1,
             objective= 'binary:logistic',
             nthread=8,
             scale_pos_weight=1,
             seed=27)
#        xgb_leak, score = xgb_fit(xgb_leak, train_one_group, features, 'mean_outcome', cv=False)
        xgb_leak.fit(train_one_group[features], train_one_group['mean_outcome'], verbose=True, eval_metric='auc')
#    train_yhat = alg.predict_proba(dtrain[predictors])[:,1]
        train_yhat = xgb_leak.predict(train_one_group[features])
#    train_score = roc_auc_score(dtrain[target].tolist(), train_yhat)
        score[ind] = metrics.accuracy_score(train_one_group['mean_outcome'].tolist(), train_yhat)        
        model[ind] = xgb_leak
        ind = ind + 1
        if ind%1000 == 0:
            print('progress: %d %%' % (ind/len(group_1)*100))
        
    group_1_leak_models = pd.DataFrame(data={'group_1':group_1, 'model':model})
    print("Accuracy : %.4g" % np.mean(score))
    return group_1_leak_models

def leak_model_predict(group_1_leak_models, test, features):
    group_1_train = group_1_leak_models['group_1'].unique()
    group_1_test = test['group_1'].unique()
    for group in group_1_test:
        if group in group_1_train:
            test_one_group = test[test['group_1']==group]
            try:
                xgb_leak = group_1_leak_models[group_1_leak_models['group_1']==group]['model'].values[0]
            except:
                print(group)    
    #        test_yhat = xgb_leak.predict_proba(test_one_group[features])[:,1]
            test_yhat = xgb_leak.predict(test_one_group[features]) #
            test.loc[test['group_1']==group, 'outcome']=test_yhat
        else:
            test.loc[test['group_1']==group, 'outcome']=0.5 # assign 0.5 for outcome when group_1 is out of the model range
    
    return test                
#%
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
train_raw, test_raw, people_raw = process_table(train_raw, test_raw, people_raw)
train = train_raw.copy()
test =  test_raw.copy()
people = people_raw.copy()
train, test = feature_engineer(train, test)
#%% creat a tabel with unique 'group_1','act_days' values
features = ['group_1','act_days']
act_days = np.arange(train['act_days'].min(),train['act_days'].max(),5) #create a activity day series for plotting
group_1 = train['group_1'].unique()
df_group_1_act_days = pd.DataFrame(list(product(group_1, act_days)), columns = features)
#%% train leak model using only group_1 and activity_date info
group_1_leak_models = build_leak_model(train, features)
df_group_1_act_days = leak_model_predict(group_1_leak_models, df_group_1_act_days, features)
test = leak_model_predict(group_1_leak_models, test, features)
train_predict = df_group_1_act_days.copy()
test_leak = pd.merge(test_raw[['group_1','act_days','activity_id']], test[['group_1','act_days','outcome']], on=['group_1','act_days'], how='left')

#%% plot the leak model prediction plus mean outcome v.s. days for each group_1 
sns.set_context("poster")

group_1 = list(train['group_1'].unique())
fig, axes = plt.subplots(nrows=8, ncols=1)

for j in range(0,8):
    k = j+8    
    train_group = train[train['group_1']==group_1[k]][['act_days', 'mean_outcome']].sort('act_days')
    train_group.columns = ['act_days', 'train_outcome']
    train_predict_group = train_predict[train_predict['group_1']==group_1[k]][['act_days', 'outcome']].sort('act_days')
    train_predict_group.columns = ['act_days', 'train_predict_outcome']
    if j == 0:
        train_group.plot(x='act_days', y = 'train_outcome', ax = axes[j],legend=True,marker='o')
        train_predict_group.plot(x='act_days', y = 'train_predict_outcome', ax = axes[j],legend=True)
    else:
        train_group.plot(x='act_days', y = 'train_outcome', ax = axes[j],legend=False,marker='o')
        train_predict_group.plot(x='act_days', y = 'train_predict_outcome', ax = axes[j],legend=False)
    #axes.set_xlim([0,36])
    axes[j].set_ylim([-0.1,1.1])
    
plt.savefig(os.path.join(proj_dir,'figures','group_1_act_day.png'),dpi=300)
#%%  train no-leak model using the non-leak features 
features = train.columns.tolist()
features.remove('group_1')
features.remove('mean_outcome')
xgb_no_leak = XGBClassifier(
 learning_rate=0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=8,
 scale_pos_weight=1,
 seed=27)
kf = LabelKFold(train['group_1'], n_folds=5)
xgb_no_leak, score = xgb_fit(xgb_no_leak, train, features, 'mean_outcome', folds=kf)
test_prediction = xgb_no_leak.predict_proba(test[features])[:,1]
#test_prediction = xgb_no_leak.predict(test[features])
test['outcome'] = test_prediction 
test_no_leak = pd.merge(test_raw[['group_1','act_days','activity_id']], test[['group_1','act_days','outcome']], on=['group_1','act_days'], how='left')

#%% wieghted averaging of prediction from leak and no-leak models
test_leak_keep_ind = test_leak['group_1'].isin(group_1)
test_leak_weight = test_leak_keep_ind.astype('float').as_matrix()*1 

test_no_leak_keep_ind = ~test_no_leak['group_1'].isin(np.unique(train_raw['group_1'].values))
test_no_leak_disca_ind = test_no_leak['group_1'].isin(np.unique(train_raw['group_1'].values))
test_no_leak_weight = test_no_leak_keep_ind.astype('float').as_matrix()+test_no_leak_disca_ind.astype('float').as_matrix()*0

test_all = test_no_leak.copy()
test_all['outcome'] = test_leak['outcome'].as_matrix()*test_leak_weight + test_no_leak['outcome'].as_matrix()*test_no_leak_weight
#%% Creat submission files
#create_submission(score, test_raw, test_prediction)
create_submission(score, test_all, test_all['outcome'].values)
#create_submission(score, test_leak, test_leak['outcome'].values)
