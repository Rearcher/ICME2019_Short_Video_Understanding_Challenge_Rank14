# ICME2019短视频内容理解与推荐竞赛Rank 14方案
[比赛链接](https://www.biendata.com/competition/icmechallenge2019/)

队员：[Yumho](https://github.com/gyh75520), [byxshr](https://github.com/byxshr)

## 文件说明
- `utils.py`: 提供了生成label均值特征的函数
- `preprocess.py`: 视频特征和音频特征预处理
- `model.py`: 模型文件
- `track2.py`: 训练文件

## 机器配置
- 内存256G，TITAN Xp 12G显存 * 2

## 模型
基于[DeepCTR](https://github.com/shenweichen/DeepCTR)的xDeepFM模型，做了些修改来支持视频特征和音频特征的输入。其中视频特征和音频特征通过`128->embedding_size`的神经网络做embedding，拼接到所有特征的embedding向量后面。

具体参数设置可查看`track2.py`。

## 特征
1. 原始特征（uid, user_city, item_id, author_id, item_city, music_id, did, video_duration）
2. 计数特征，即统计某个字段的出现次数（uid, did, item_id, author_id, uid-author_id）
3. label均值特征，即根据某个字段分组统计每个分组的标签均值（uid, did, item_id, uid-author_id, uid-did, did-channel）
4. nunique特征，例如uid_item_nunique，是统计每个uid下有多少不同的item_id，等频离散化
    - uid_icity_nunique
    - uid_item_nunique
    - uid_author_nunique
    - uid_music_nunique
    - item_ucity_nunique
    - item_uid_nunique
    - author_uid_nunique
5. 视频特征
6. 音频特征
7. 标题特征，提取视频标题的不重复字段，当作序列特征输入，最后做sum pooling得到embedding向量

## 成绩
最终成绩是跑10次取平均；  
track1：a/b榜都是15；  
track2：a榜第14，分数为0.79405 (0.73,0.93)；b榜第14，分数为0.79485 (0.74,0.93)

## 参考
- Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.(https://arxiv.org/pdf/1803.05170.pdf)
- [Bytedance_ICME2019_challenge_baseline](https://github.com/shenweichen/Bytedance_ICME2019_challenge_baseline)
- [Data-Competition-TopSolution](https://github.com/Smilexuhc/Data-Competition-TopSolution)
