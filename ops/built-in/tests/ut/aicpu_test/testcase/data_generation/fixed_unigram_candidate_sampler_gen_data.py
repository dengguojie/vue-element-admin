"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf
import numpy as np
import csv
from tensorflow.python.framework import constant_op, dtypes

def write_file_txt(file_name, data, fmt="%s"):
    """
    prama1: file_name: the file which store the data
    param2: data: data which will be stored
    param3: fmt: format
    """
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def read_file_txt(file_name, dtype, delim=None):
    """
    prama1: file_name: the file which store the data
    param2: dtype: data type
    param3: delim: delimiter which is used to split data
    """
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


def read_file_txt_to_bool(file_name, delim=None):
    """
    prama1: file_name: the file which store the data
    param2: delim: delimiter which is used to split data
    """
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    """
    prama1: data_file: the file which store the generation data
    param2: shape: data shape
    param3: dtype: data type
    param4: rand_type: the method of generate data, select from "randint, uniform"
    param5: data lower limit
    param6: data upper limit
    """
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    if dtype == np.float16:
        data = np.array(rand_data, dtype=np.float32)
        write_file_txt(data_file, data, fmt="%s")
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(rand_data, dtype=dtype)
        write_file_txt(data_file, data, fmt="%s")
    return data


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config


def gen_random_data_seed_0():
    data_files=["fixed_unigram_candidate_sampler/data/fucs_true_classes_1.txt",
                "fixed_unigram_candidate_sampler/data/fucs_unigrams_1.txt",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_1.txt",
                "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_1.txt",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_1.txt"]
    true_classes = np.array([[1, 2, 3, 4, 5]], np.int64)
    write_file_txt(data_files[0], true_classes, fmt="%s")
    true_classes = tf.constant(true_classes, dtype=tf.int64)
    
    unigrams = list(gen_data_file(data_files[1], [5], np.float32, "uniform", 0.1, 1))

    re = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=true_classes,
        num_true=5,
        num_sampled=2,
        unique=True,
        range_max=5,
        vocab_file='',
        distortion=1.0,
        num_reserved_ids=0,
        num_shards=1,
        shard=0,
        unigrams=unigrams,
        seed=0,
    )

    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[2], data[0], fmt="%s")
    write_file_txt(data_files[3], data[1], fmt="%s")
    write_file_txt(data_files[4], data[2], fmt="%s")


def gen_random_data_seed_1():
    data_files=["fixed_unigram_candidate_sampler/data/fucs_true_classes_2.txt",
                "fixed_unigram_candidate_sampler/data/fucs_unigrams_2.txt",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_2.txt",
                "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_2.txt",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_2.txt"]
    true_classes = np.array([[1, 2, 3, 4, 5, 6]], np.int64)
    write_file_txt(data_files[0], true_classes, fmt="%s")
    true_classes = tf.constant(true_classes, dtype=tf.int64)
    
    unigrams = list(gen_data_file(data_files[1], [6], np.float32, "uniform", 0.1, 1))

    re = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=true_classes,
        num_true=6,
        num_sampled=3,
        unique=True,
        range_max=6,
        vocab_file='',
        distortion=1.0,
        num_reserved_ids=0,
        num_shards=1,
        shard=0,
        unigrams=unigrams,
        seed=1,
    )

    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[2], data[0], fmt="%s")
    write_file_txt(data_files[3], data[1], fmt="%s")
    write_file_txt(data_files[4], data[2], fmt="%s")


def gen_random_data_unique_false():
    data_files=["fixed_unigram_candidate_sampler/data/fucs_true_classes_3.txt",
                "fixed_unigram_candidate_sampler/data/fucs_unigrams_3.txt",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_3.txt",
                "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_3.txt",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_3.txt"]
    true_classes = np.array([[1, 2, 3, 4, 5, 6]], np.int64)
    write_file_txt(data_files[0], true_classes, fmt="%s")
    true_classes = tf.constant(true_classes, dtype=tf.int64)
    
    unigrams = list(gen_data_file(data_files[1], [6], np.float32, "uniform", 0.1, 1))

    re = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=true_classes,
        num_true=6,
        num_sampled=3,
        unique=False,
        range_max=6,
        vocab_file='',
        distortion=1.0,
        num_reserved_ids=0,
        num_shards=1,
        shard=0,
        unigrams=unigrams,
        seed=2,
    )

    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[2], data[0], fmt="%s")
    write_file_txt(data_files[3], data[1], fmt="%s")
    write_file_txt(data_files[4], data[2], fmt="%s")


def gen_random_data_file():
    data_files=["fixed_unigram_candidate_sampler/data/fucs_true_classes_4.txt",
                "fixed_unigram_candidate_sampler/data/fucs_unigrams_4.csv",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_4.txt",
                "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_4.txt",
                "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_4.txt"]
    true_classes = np.array([[1, 2, 3, 4, 5, 6]], np.int64)
    write_file_txt(data_files[0], true_classes, fmt="%s")
    true_classes = tf.constant(true_classes, dtype=tf.int64)

    rand_data = np.random.uniform(low=0.1, high=1, size=[6])
    unigrams = np.array(rand_data, dtype=np.float32)

    csvfile = open(data_files[1], 'w', newline='')
    writer = csv.writer(csvfile)
    for i in range(len(unigrams)):
        item = unigrams[i]
        writer.writerow([i, item])
    csvfile.close()

    re = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=true_classes,
        num_true=6,
        num_sampled=3,
        unique=False,
        range_max=6,
        vocab_file=data_files[1],
        distortion=1.0,
        num_reserved_ids=0,
        num_shards=1,
        shard=0,
        unigrams=list(),
        seed=2,
    )

    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[2], data[0], fmt="%s")
    write_file_txt(data_files[3], data[1], fmt="%s")
    write_file_txt(data_files[4], data[2], fmt="%s")


def run():
    gen_random_data_seed_0()
    gen_random_data_seed_1()
    gen_random_data_unique_false()
    gen_random_data_file()
