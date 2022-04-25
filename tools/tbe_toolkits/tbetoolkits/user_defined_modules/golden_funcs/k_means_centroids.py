# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from .registry import register_golden

def deal_with_large_data(samples, centroids, sum_square_centroids, sum_square_samples, use_actual_distance=False):
    """ golden data
    """
    # only support m % 64 = 0
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
        if use_actual_distance:
            np_distance += sum_square_samples[idx * m_part:(idx + 1) * m_part, :]
        np_distance += sum_square_centroids

        argmin_index = np.argmin(np_distance, axis=1)
        argmin = np.min(np_distance, axis=1)

        for i in range(len(argmin)):
            np_sum_array[argmin_index[i]] += samples[idx * m_part + i]
            np_count_array[argmin_index[i]][0] += 1
        np_total_distance += np.sum(argmin, axis=0, keepdims=True).astype("float32")

    return np_sum_array, np_count_array, np_total_distance

@register_golden(["k_means_centroids"])
def _kmeans(sample, centroid, sum_square_centroid, sum_square_sample, use_actual_distance=False):
    """ golden data
    """
    m, d = sample.shape
    n = centroid.shape[0]

    if m >= 65536 and n >= 1024:
        return deal_with_large_data(sample, centroid, sum_square_centroid, sum_square_sample,
                                    use_actual_distance=use_actual_distance)

    matmul_res = np.matmul(sample, centroid.T).astype("float32")

    np_distance = -2 * matmul_res
    if use_actual_distance:
        np_distance += sum_square_sample
    np_distance += sum_square_centroid
    argmin_index = np.argmin(np_distance, axis=1)
    argmin = np.min(np_distance, axis=1)

    np_sum_array = np.zeros((n, d), "float32")
    np_count_array = np.zeros((n, 1), "float32")

    for i in range(len(argmin)):
        np_sum_array[argmin_index[i]] += sample[i]
        np_count_array[argmin_index[i]][0] += 1
    np_total_distance = np.sum(argmin, axis=0, keepdims=True).astype("float32")

    return np_sum_array, np_count_array, np_total_distance
