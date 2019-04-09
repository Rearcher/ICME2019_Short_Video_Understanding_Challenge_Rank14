import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# 获取正样本计数
def gen_pos_counts(X_train, X_test, cols, target):
    train, test = pd.DataFrame(), pd.DataFrame()
    
    for col in cols:
        new_col = col + '_' + target + '_cnt'
        count_map = X_train.groupby(col)[target].sum()
        train[new_col] = X_train[col].map(count_map) - X_train[target]
        test[new_col] = X_test[col].map(count_map).fillna(np.mean(count_map))
    return pd.concat([train, test], axis=0).reset_index(drop=True)


# kfold target均值
def get_kfold_mean(X_train, X_test, cols, target, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=918)
    train, test = pd.DataFrame(), pd.DataFrame()
    
    for col in cols:
        new_col = col + '_' + target + '_mean'
        train[new_col] = np.zeros(X_train.shape[0])
        
    for tr_idx, val_idx in skf.split(X_train, X_train[target]):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        for col in cols:
            new_col = col + '_' + target + '_mean'
            tmp_means = X_val[col].map(X_tr.groupby(col)[target].mean())
            train[new_col][val_idx] = tmp_means
            
    prior = X_train[target].mean()
    for col in cols:
        target_map = X_train.groupby(col)[target].mean()
        
        new_col = col + '_' + target + '_mean'
        train[new_col].fillna(prior, inplace=True)
    
        test[new_col] = X_test[col].map(target_map)
        test[new_col].fillna(prior, inplace=True)
    
    return pd.concat([train, test], axis=0).reset_index(drop=True)


# smooth target均值
def get_smooth_mean(X_train, X_test, cols, target, m=300):
    def get_smooth_mean_map(df, by, on, m=300):
        mean = df[on].mean()
        agg = df.groupby(by)[on].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        smooth = (counts * means + m * mean) / (counts + m)
        return smooth
    
    prior = X_train[target].mean()
    train, test = pd.DataFrame(), pd.DataFrame()
    
    for col in cols:
        new_col = col + '_' + target + '_mean'
        target_map = get_smooth_mean_map(X_train, by=col, on=target, m=m)
        train[new_col] = X_train[col].map(target_map)
        test[new_col] = X_test[col].map(target_map).fillna(prior)
    
    return pd.concat([train, test], axis=0).reset_index(drop=True)


# leave-one-out target均值
def get_loo_mean(X_train, X_test, cols, target):
    prior = X_train[target].mean()
    train, test = pd.DataFrame(), pd.DataFrame()
    
    for col in cols:
        new_col = col + '_' + target + '_mean'
        
        target_sum = X_train.groupby(col)[target].transform('sum')
        n_objects = X_train.groupby(col)[target].transform('count')
        
        train[new_col] = (target_sum - X_train[target]) / (n_objects - 1)
        train[new_col].fillna(prior, inplace=True)
        
        test[new_col] = X_test[col].map(X_train.groupby(col)[target].mean())
        test[new_col].fillna(prior, inplace=True)
    return pd.concat([train, test], axis=0).reset_index(drop=True)    


# expanding target均值
def get_expanding_mean(X_train, X_test, cols, target):
    prior = X_train[target].mean()
    train, test = pd.DataFrame(), pd.DataFrame()
    
    for col in cols:
        new_col = col + '_' + target + '_mean'
        
        cumsum = X_train.groupby(col)[target].cumsum() - X_train[target]
        cumcnt = X_train.groupby(col)[target].cumcount()
        train[new_col] = cumsum / cumcnt
        train[new_col].fillna(prior, inplace=True)
        
        test[new_col] = X_test[col].map(X_train.groupby(col)[target].mean())
        test[new_col].fillna(prior, inplace=True)
    return pd.concat([train, test], axis=0).reset_index(drop=True)
