import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import parsing_ops
pb_file_path = os.getcwd()

def decodefn(record_bytes):
    return parsing_ops.parse_single_example(
        record_bytes,
        {
            "city":tf.io.FixedLenFeature([], dtype = tf.)
        },
        None,
        "exmaple"
    )


with tf.Session(graph=tf.Graph()) as sess:
    value_city = u"北京".encode('utf-8')   # 城市
    value_use_day = 7                      #最近7天打开淘宝次数
    value_pay = 289.4                      # 最近7 天消费金额
    value_poi = [b"123", b"456", b"789"]   #最近7天浏览电铺
    
    '''
    下面生成ByteList，Int64List和FloatList
    '''
    bl_city = tf.train.BytesList(value = [value_city])  ## tf.train.ByteList入参是list，所以要转为list
    il_use_day = tf.train.Int64List(value = [value_use_day])
    fl_pay = tf.train.FloatList(value = [value_pay])
    bl_poi = tf.train.BytesList(value = value_poi)
    
    '''
    下面生成tf.train.Feature
    '''
    feature_city = tf.train.Feature(bytes_list = bl_city)
    feature_use_day = tf.train.Feature(int64_list = il_use_day)
    feature_pay = tf.train.Feature(float_list = fl_pay)
    feature_poi = tf.train.Feature(bytes_list = bl_poi)
    '''
    下面定义tf.train.Features
    '''
    feature_dict = {"city":feature_city,"use_day":feature_use_day,"pay":feature_pay,"poi":feature_poi}
    features = tf.train.Features(feature = feature_dict)

    example = tf.train.Example(features = features)
    
    decodefn(example)
    tf.io.write_graph(sess.graph, logdir="./", name="parse_single_case_1.pb", as_text=False)