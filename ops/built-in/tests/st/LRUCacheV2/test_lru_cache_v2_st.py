#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np


def np_lru_cache_v2(index_list_dic, data_dic, cache_dic, tag_dic, pre_route_count):
    index_list = index_list_dic['value']
    data = data_dic['value']
    cache = cache_dic['value']
    tag = tag_dic['value']
    data_shape = data_dic['shape']
    cache_shape = cache_dic['shape']
    embedding_size = data_shape[-1]
    way_number = 8
    core_num = 8
    set_number = cache_shape[-1] // embedding_size // way_number

    # workspace init
    time_stamp_wsp = set_number * [0 for i in range(way_number)]
    miss_index_wsp = [[] for i in range(set_number)]

    index_set_list = index_list // pre_route_count % set_number
    index_offset_cpu = [-1] * index_list.size
    cpu_not_in_cache_number = 0
    tag = tag.reshape([set_number, way_number])
    for core_id in range(core_num):
        in_cache = 0
        not_in_cache = 0
        for i, index_set in enumerate(index_set_list):
            if index_set % core_num == core_id:
                index_now = index_list[i]
                if index_now in tag[index_set][:]:
                    index_way = list(tag[index_set][:]).index(index_now)
                    index_offset = (index_set * way_number + index_way) * embedding_size
                    in_cache += 1
                    time_stamp_wsp[index_set * way_number + index_way] += 1
                else:
                    cpu_not_in_cache_number += 1
                    index_offset = -1
                    not_in_cache += 1
                    miss_index_wsp[index_set].append(index_now)
                index_offset_cpu[i] = index_offset
        for set_id in range(set_number):
            if set_id % core_num == core_id:
                set_time_stamp = time_stamp_wsp[set_id * way_number:(set_id+1) * way_number]
                set_time_stamp_with_index = [[i, set_time_stamp[i]]  for i in range(way_number)]
                set_time_stamp_with_index.sort(key=lambda x: x[1], reverse=True)
                for i in range(min(way_number, len(miss_index_wsp[set_id]))):
                    miss_index = miss_index_wsp[set_id][i]
                    in_index = miss_index
                    out_index = tag[set_id][set_time_stamp_with_index[way_number - i - 1][0]]
                    tag_offset = set_time_stamp_with_index[way_number - i - 1][0] + set_id * way_number
                    # exchange
                    data[out_index:(out_index + 1)][:] = \
                        cache[tag_offset * embedding_size:(tag_offset + 1) * embedding_size]
                    cache[tag_offset * embedding_size:(tag_offset + 1) * embedding_size] = \
                        data[in_index:(out_index + 1)][:]
                    tag[set_id][set_time_stamp_with_index[way_number - i - 1][0]] = in_index
    index_offset_np = np.array(index_offset_cpu, index_list.dtype)
    not_in_cache_number = np.array(cpu_not_in_cache_number, index_list.dtype)

    return [data, cache, tag, index_offset_np, index_offset_np, not_in_cache_number]


def calc_expect_func(index_list, data, cache, tag, is_last_call, out_data, out_cache, out_tag, index_offset_list,
                     not_in_cache_index_list, not_in_cache_number, pre_route_number):
    res = np_lru_cache_v2(index_list, data, cache, tag, pre_route_number)

    return res