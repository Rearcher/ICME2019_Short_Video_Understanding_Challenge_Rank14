import tensorflow as tf
from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.interaction import CIN
from deepctr.layers.core import MLP, PredictionLayer
from deepctr.layers.utils import concat_fun
from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.regularizers import l2


def xDeepFM_MTL(feature_dim_dict, embedding_size=8, seed=1024, init_std=0.0001, l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                cin_layer_size=(256, 256), cin_split_half=True, cin_activation='relu',
                hidden_size=(256, 256), activation='relu', keep_prob=1, use_bn=False, l2_reg_deep=0,
                task_net_size=(128,), task_activation='relu', task_keep_prob=1, task_use_bn=False, task_l2=0,
                final_activation='sigmoid', use_video=False, use_audio=False):

    check_feature_config_dict(feature_dim_dict)
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')

    deep_emb_list, linear_logit, inputs_list = preprocess_input_embedding(
        feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed, True)
    
    if use_video:
        video_input = tf.keras.layers.Input(shape=(128,), name='video')
        video_emb = tf.keras.layers.Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding))(video_input)
        video_emb = tf.keras.layers.Reshape((1, embedding_size), input_shape=(embedding_size,))(video_emb)
        deep_emb_list.append(video_emb)
        inputs_list.append(video_input)
        
    if use_audio:
        audio_input = tf.keras.layers.Input(shape=(128,), name='audio')
        audio_emb = tf.keras.layers.Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding))(audio_input)
        audio_emb = tf.keras.layers.Reshape((1, embedding_size), input_shape=(embedding_size,))(audio_emb)
        deep_emb_list.append(audio_emb)
        inputs_list.append(audio_input)
        
    fm_input = concat_fun(deep_emb_list, axis=1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, cin_activation, cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None,)(exFM_out)

    deep_input = tf.keras.layers.Flatten()(fm_input)
        
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob, use_bn, seed)(deep_input)

    finish_out = MLP(task_net_size, task_activation, task_l2, task_keep_prob, task_use_bn, seed)(deep_out)
    finish_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(finish_out)

    like_out = MLP(task_net_size, task_activation, task_l2, task_keep_prob, task_use_bn, seed)(deep_out)
    like_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(like_out)
    
    finish_logit = tf.keras.layers.add([linear_logit, finish_logit, exFM_logit])
    like_logit = tf.keras.layers.add([linear_logit, like_logit, exFM_logit])

    output_finish = PredictionLayer(final_activation, name='finish')(finish_logit)
    output_like = PredictionLayer(final_activation, name='like')(like_logit)
    
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[output_finish, output_like])
    return model


def xDeepFM(feature_dim_dict, embedding_size=8, seed=1024, init_std=0.0001, l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
            cin_layer_size=(256, 256), cin_split_half=True, cin_activation='relu', 
            hidden_size=(256, 256), activation='relu', keep_prob=1, use_bn=False, l2_reg_deep=0,
            final_activation='sigmoid', use_video=False, use_audio=False):

    check_feature_config_dict(feature_dim_dict)
    deep_emb_list, linear_logit, inputs_list = preprocess_input_embedding(
        feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed, True)
    
    if use_video:
        video_input = tf.keras.layers.Input(shape=(128,), name='video')
        video_emb = tf.keras.layers.Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding))(video_input)
        video_emb = tf.keras.layers.Reshape((1, embedding_size), input_shape=(embedding_size,))(video_emb)
        deep_emb_list.append(video_emb)
        inputs_list.append(video_input)
              
    if use_audio:
        audio_input = tf.keras.layers.Input(shape=(128,), name='audio')
        audio_emb = tf.keras.layers.Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding))(audio_input)
        audio_emb = tf.keras.layers.Reshape((1, embedding_size), input_shape=(embedding_size,))(audio_emb)
        deep_emb_list.append(audio_emb)
        inputs_list.append(audio_input)
        
    fm_input = concat_fun(deep_emb_list, axis=1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, cin_activation, cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None,)(exFM_out)

    deep_input = tf.keras.layers.Flatten()(fm_input)
        
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob, use_bn, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)
    
    final_logit = tf.keras.layers.add([linear_logit, deep_logit, exFM_logit])
    output = PredictionLayer(final_activation, name='output')(final_logit)
    
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
