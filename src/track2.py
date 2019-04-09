import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from deepctr import SingleFeat, VarLenFeat
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from model import xDeepFM
from utils import *

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_session(tf.Session(config=config))


def get_input(use_count=False, use_unique=False, use_video=False, use_audio=False, use_title=False, ONLINE_FLAG=False, SAMPLE_FLAG=True, VALIDATION_FRAC=0.2, target='finish'):
    train_file = 'track2/sample_train.txt' if SAMPLE_FLAG else 'track2/final_track2_train.txt'
    test_file = 'track2/sample_test_no_answer.txt' if SAMPLE_FLAG else 'track2/final_track2_test_no_anwser.txt'
    video_file = 'track2/sample_video_features.csv' if SAMPLE_FLAG else 'track2/track2_video_features_mms.csv'
    face_file = 'track2/sample_face2.csv' if SAMPLE_FLAG else 'track2/face_df2.csv'
    audio_file = 'track2/sample_audio_features.csv' if SAMPLE_FLAG else 'track2/track2_audio_features.csv'
    title_file = 'track2/sample_title.txt' if SAMPLE_FLAG else 'track2/track2_title.txt'

    data = pd.read_csv(train_file, sep='\t', names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 
                                                    'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    print('training set read completed.')
    if ONLINE_FLAG:
        test_data = pd.read_csv(test_file, sep='\t', names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 
                                                            'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
        train_size = data.shape[0]
        data = data.append(test_data).reset_index(drop=True)
    else:
        train_size = int(data.shape[0]*(1-VALIDATION_FRAC))
    print('test set read completed.')

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', ]
    dense_features = []

    data['video_duration'] = pd.qcut(data['video_duration'], q=10, labels=False, duplicates='drop')
    sparse_features.append('video_duration')

    data['creat_time'] = data['creat_time'] % (24 * 3600) / 3600
    data['creat_time'] = pd.qcut(data['creat_time'], q=24, labels=False, duplicates='drop')
    sparse_features.append('creat_time')

    if use_count:
        data['uid-author_id'] = data['uid'].astype(str) + '-' + data['author_id'].astype(str)
        data['uid-did'] = data['uid'].astype(str) + '-' + data['did'].astype(str)
        data['did-channel'] = data['did'].astype(str) + '-' + data['channel'].astype(str)

        # 计数特征
        cols = ['uid', 'did', 'item_id', 'author_id', 'uid-author_id']
        for c in cols:
            data[c + '_cnt'] = data[c].map(data[c].value_counts())
            data[c + '_cnt'] = pd.qcut(data[c + '_cnt'], q=10, labels=False, duplicates='drop')
            sparse_features.append(c + '_cnt')

        # 均值特征  
        df = get_expanding_mean(data[:train_size], data[train_size:], 
                                ['uid-author_id', 'uid-did', 'did-channel', 'uid', 'did', 'item_id'], 
                                'finish')
        dense_features += list(df.columns)
        data = pd.concat([data, df], axis=1)

    if use_unique:
        data['uid_icity_nunique'] = data['uid'].map(data.groupby('uid')['item_city'].nunique())
        data['uid_icity_nunique'] = pd.qcut(data['uid_icity_nunique'], q=10, labels=False, duplicates='drop')
        sparse_features.append('uid_icity_nunique')

        data['uid_item_nunique'] = data['uid'].map(data.groupby('uid')['item_id'].nunique())
        data['uid_item_nunique'] = pd.qcut(data['uid_item_nunique'], q=10, labels=False, duplicates='drop')
        sparse_features.append('uid_item_nunique')

        data['uid_author_nunique'] = data['uid'].map(data.groupby('uid')['author_id'].nunique())
        data['uid_author_nunique'] = pd.qcut(data['uid_author_nunique'], q=10, labels=False, duplicates='drop')
        sparse_features.append('uid_author_nunique')

        data['uid_music_nunique'] = data['uid'].map(data.groupby('uid')['music_id'].nunique())
        data['uid_music_nunique'] = pd.qcut(data['uid_music_nunique'], q=10, labels=False, duplicates='drop')
        sparse_features.append('uid_music_nunique')

        data['item_ucity_nunique'] = data['item_id'].map(data.groupby('item_id')['user_city'].nunique())
        data['item_ucity_nunique'] = pd.qcut(data['item_ucity_nunique'], q=10, labels=False, duplicates='drop')
        sparse_features.append('item_ucity_nunique')

        data['item_uid_nunique'] = data['item_id'].map(data.groupby('item_id')['uid'].nunique())    
        data['item_uid_nunique'] = pd.qcut(data['item_uid_nunique'], q=30, labels=False, duplicates='drop')
        sparse_features.append('item_uid_nunique')

        data['author_uid_nunique'] = data['author_id'].map(data.groupby('author_id')['uid'].nunique())
        data['author_uid_nunique'] = pd.qcut(data['author_uid_nunique'], q=20, labels=False, duplicates='drop')
        sparse_features.append('author_uid_nunique')

    print('generate stats feats completed.')

    if use_video:
        video_feats = pd.read_csv(video_file)
        print('video feats read completed.')

        data = pd.merge(data, video_feats, how='left', on='item_id')
        for i in range(128):
            col = 'vd' + str(i)
            data[col].fillna(0, inplace=True)
        print('merge video feats completed.')

    if use_audio:
        audio_feats = pd.read_csv(audio_file)
        print('audio feats read completed.')

        data = pd.merge(data, audio_feats, how='left', on='item_id')
        for i in range(128):
            col = 'ad' + str(i)
            data[col].fillna(0, inplace=True)
        print('merge audio feats completed.')

    if use_title:
        max_len = 47 
        title_feats = pd.read_json(title_file, lines=True)
        print('title feats read completed')

        def get_title_len(d):
            return sum(d.values())
        title_feats['title_len'] = title_feats['title_features'].apply(get_title_len)
        prior = title_feats['title_len'].mean()

        dense_features.append('title_len')
        title_feats['title_features'] = title_feats['title_features'].apply(lambda x: list(x.keys()))

        data = pd.merge(data, title_feats, how='left', on='item_id')
        for row in data.loc[data.title_features.isna(), 'title_features'].index:
            data.at[row, 'title_features'] = []
        data['title_len'].fillna(prior, inplace=True)
        print('merge title feats completed')

    data[sparse_features] = data[sparse_features].fillna('-1', )

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    if len(dense_features) > 0:
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique()) for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0) for feat in dense_features]
    sequence_feature_list = []

    if use_title:
        sequence_feature_list.append(VarLenFeat('title', 134545, max_len, 'sum'))

    print('data preprocess completed.')

    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]

    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]

    if use_title:
        train_model_input += [pad_sequences(train['title_features'], maxlen=max_len, padding='post')]
        test_model_input += [pad_sequences(test['title_features'], maxlen=max_len, padding='post')]

    if use_video:
        vd_cols = ['vd' + str(i) for i in range(128)]
        video_input = data[vd_cols].values
        train_model_input += [video_input[:train_size]]
        test_model_input += [video_input[train_size:]]

    if use_audio:
        ad_cols = ['ad' + str(i) for i in range(128)]
        audio_input = data[ad_cols].values
        train_model_input += [audio_input[:train_size]]
        test_model_input += [audio_input[train_size:]]

    print('input process completed.')
    print(f'use sparse feats: [{",".join(sparse_features)}]')
    print(f'use dense feats: [{",".join(dense_features)}]')

    train_labels, test_labels = train[target].values, test[target].values
    feature_dim_dict = {"sparse": sparse_feature_list, "dense": dense_feature_list, "sequence": sequence_feature_list}

    if ONLINE_FLAG:
        return feature_dim_dict, train_model_input, train_labels, test_model_input, test_labels, test_data
    return feature_dim_dict, train_model_input, train_labels, test_model_input, test_labels


def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def main():
    input_params = {
        'use_count': False,       # 是否使用计数、均值特征
        'use_unique': False,      # 是否使用nunique等频离散化特征
        'use_video': False,       # 是否使用视频特征
        'use_audio': False,       # 是否使用音频特征
        'use_title': False,       # 是否使用标题特征
        'ONLINE_FLAG': False,     # True为线上提交，输出提交文件；False为线下测试
        'SAMPLE_FLAG': False,     # True为小样本测试程序正确性
        'VALIDATION_FRAC': 0.2,   # 线下验证集大小，按照顺序划分
        'target': 'finish'        # 预测目标
    }

    if input_params['ONLINE_FLAG']:
        feature_dim_dict, train_model_input, train_labels, test_model_input, test_labels, test_data = get_input(**input_params)
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability', 'like': 'like_probability'}, inplace=True)
    else:
        feature_dim_dict, train_model_input, train_labels, test_model_input, test_labels = get_input(**input_params)
     
    iterations = 10 # 跑多次取平均
    for i in range(iterations):                
        print(f'iteration {i + 1}/{iterations}')

        model = xDeepFM(feature_dim_dict, 
                    embedding_size=8, seed=1278, init_std=0.0001, l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                    cin_layer_size=(256, 256, 256), cin_split_half=True, cin_activation='relu',
                    hidden_size=(256, 256), activation='relu', keep_prob=1, use_bn=False, l2_reg_deep=0,
                    final_activation='sigmoid', use_video=input_params['use_video'], use_audio=input_params['use_audio'])        
        model.compile("adagrad", "binary_crossentropy", metrics=[auc])

        if input_params['ONLINE_FLAG']:
            history = model.fit(train_model_input, train_labels, batch_size=4096, epochs=1, verbose=1)
            pred_ans = model.predict(test_model_input, batch_size=4096)
            auc_train = history.history['auc'][-1]

        else:
            history = model.fit(train_model_input, train_labels, batch_size=4096, epochs=1, verbose=1, 
                                validation_data=(test_model_input, test_labels))
           
            auc_train, auc_valid = history.history['auc'][-1], history.history['val_auc'][-1]
            print(f'train score = {auc_train:.5f}\nvalid score = {auc_valid:.5f}')
    
        if input_params['ONLINE_FLAG']:
            tmp_df = pd.DataFrame({'uid': result['uid'], 'item_id': result['item_id']})
            tmp_df['finish_probability'] = pred_ans
            tmp_df.to_csv(f'output/finish_{auc_train:.5f}.csv', index=False, float_format='%.6f')
            
            if i == 0:
                result['finish_probability'] = tmp_df['finish_probability'] / iterations
            else:
                result['finish_probability'] += tmp_df['finish_probability'] / iterations
    
    if input_params['ONLINE_FLAG']:
        result.to_csv('output/finish_result.csv', index=None, float_format='%.6f')

    print('done.')
    
    
if __name__ == '__main__':
    main()
