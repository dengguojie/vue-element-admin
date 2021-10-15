#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
KMeansCentroids golden data generation function
"""
# Third-Party Packages
import numpy as np

def deal_with_large_data(samples, centroids, sum_square_centroids, sum_square_samples,
                         use_actual_distance, impl_mode):
    """ golden data
    """
    m = samples.shape[0]
    n, d = centroids.shape
    np_sum_array = np.zeros((n, d), "float32")
    np_count_array = np.zeros((n, 1), "float32")
    np_total_distance = 0
    m_part = 64
    m_loop = m // m_part
    for idx in range(m_loop):
        matmul_res = np.matmul(samples[idx * m_part:(idx + 1) * m_part, :], centroids.T).astype("float32")

        np_distance = -2 * matmul_res
        np_distance += sum_square_centroids

        if impl_mode == "high_performance":
            np_distance = np_distance.astype(np.float16)

        argmin_index = np.argmin(np_distance, axis=1)
        argmin = np.min(np_distance, axis=1)

        if impl_mode == "high_performance":
            argmin = argmin.astype(np.float32)

        if use_actual_distance:
            argmin += sum_square_samples[idx * m_part:(idx + 1) * m_part, :].flatten()

        for i in range(len(argmin)):
            np_sum_array[argmin_index[i]] += samples[idx * m_part + i]
            np_count_array[argmin_index[i]][0] += 1
        np_total_distance += np.sum(argmin, axis=0, keepdims=True).astype("float32")

    return np_sum_array, np_count_array, np_total_distance

def calc_expect_func(x, y, sum_square_y, sum_square_x,
                     segment_sum, segment_count, kmean_total_distance, use_actual_distance, impl_mode):
    """ golden data
    """
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    m = shape_x[0]
    n, d = shape_y
    data_x = x.get("value")
    data_y = y.get("value")
    data_sum_square_x = sum_square_x.get("value")
    data_sum_square_y = sum_square_y.get("value")

    if m >= 65536 and n >= 1024:
        return deal_with_large_data(data_x, data_y, data_sum_square_y, data_sum_square_x,
                                    use_actual_distance, impl_mode)

    matmul_res = np.matmul(data_x, data_y.T).astype("float32")

    np_distance = -2 * matmul_res
    np_distance += data_sum_square_y

    if impl_mode == "high_performance":
        np_distance = np_distance.astype(np.float16)

    argmin_index = np.argmin(np_distance, axis=1)
    argmin = np.min(np_distance, axis=1)

    if impl_mode == "high_performance":
        argmin = argmin.astype(np.float32)

    if use_actual_distance:
        argmin += data_sum_square_x.flatten()

    np_sum_array = np.zeros((n, d), "float32")
    np_count_array = np.zeros((n, 1), "float32")

    for i in range(len(argmin)):
        np_sum_array[argmin_index[i]] += data_x[i]
        np_count_array[argmin_index[i]][0] += 1

    np_total_distance = np.sum(argmin, axis=0, keepdims=True).astype("float32")

    return np_sum_array, np_count_array, np_total_distance
