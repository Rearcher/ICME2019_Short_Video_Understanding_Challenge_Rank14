import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 将音频特征文件转换为标准的csv
audio_feats = pd.read_json('track2/track2_audio_features.txt', lines=True)
audio_feats.drop_duplicates(subset='item_id', inplace=True)
audio_df = pd.DataFrame(audio_feats.audio_feature_128_dim.tolist(), columns=['ad' + str(i) for i in range(128)])
audio_df['item_id'] = video_feats['item_id']
audio_df.to_csv('track2/track2_audio_features.csv', index=False, float_format='%.4f')

# 将视频特征文件转换为标准的csv
video_feats = pd.read_json('track2/track2_video_features.txt', lines=True)
video_df = pd.DataFrame(video_feats.video_feature_dim_128.tolist(), columns=['vd' + str(i) for i in range(128)])
video_df['item_id'] = video_feats['item_id']
video_df.fillna(0, inplace=True)
cols = ['vd' + str(i) for i in range(128)]
video_df[cols] = MinMaxScaler().fit_transform(video_df[cols])
video_df.to_csv('track2/track2_video_features_mms.csv', index=False, float_format='%.4f')
