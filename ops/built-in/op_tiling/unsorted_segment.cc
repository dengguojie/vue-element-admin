/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file unsorted_segment.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const std::string UNSORTED_SEGMENT_OP_TYPE = "UnsortedSegment";
const int32_t BYTE_BLOCK = 32;
const int32_t BYTE_INT32 = 4;
const int32_t BYTE_FULL_MASK = 256;
const int32_t ONE_BLOCK_E = 1;
const int32_t ONE_DIV_E = 2;
const int32_t SMALL_E = 3;
const int32_t BIG_E = 4;

const int32_t True = 1, False = 0;
const int32_t UB_TEMPLATE_NUM = 3;

const vector<map<string, int>> SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID = {
    {{"select_key", 10}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", ONE_BLOCK_E},
     {"big_iid", False}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID = {
    {{"select_key", 20}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", ONE_DIV_E},
     {"big_iid", False}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN = {
    {{"select_key", 30}, {"ub_div_id", 0}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", True}},
    {{"select_key", 31}, {"ub_div_id", 1}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", True}},
    {{"select_key", 32}, {"ub_div_id", 2}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID = {
    {{"select_key", 40}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", True}},
    {{"select_key", 41}, {"ub_div_id", 1}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", True}},
    {{"select_key", 42}, {"ub_div_id", 2}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN = {
    {{"select_key", 50}, {"ub_div_id", 0}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 51}, {"ub_div_id", 1}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 52}, {"ub_div_id", 2}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", False}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID = {
    {{"select_key", 60}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 61}, {"ub_div_id", 1}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 62}, {"ub_div_id", 2}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", False}, {"div_oid", False}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN = {
    {{"select_key", 70}, {"ub_div_id", 0}, {"block_align", True}, {"e_level", BIG_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 71}, {"ub_div_id", 1}, {"block_align", True}, {"e_level", BIG_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 72}, {"ub_div_id", 2}, {"block_align", True}, {"e_level", BIG_E},
     {"big_iid", False}, {"div_oid", False}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID = {
    {{"select_key", 80}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", BIG_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 81}, {"ub_div_id", 1}, {"block_align", False}, {"e_level", BIG_E},
     {"big_iid", False}, {"div_oid", False}},
    {{"select_key", 82}, {"ub_div_id", 2}, {"block_align", False}, {"e_level", BIG_E},
     {"big_iid", False}, {"div_oid", False}}};

const vector<map<string, int>> SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID = {
    {{"select_key", 90}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", ONE_BLOCK_E},
     {"big_iid", True}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID = {
    {{"select_key", 100}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", ONE_DIV_E},
     {"big_iid", True}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN = {
    {{"select_key", 110}, {"ub_div_id", 0}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", True}},
    {{"select_key", 111}, {"ub_div_id", 1}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", True}},
    {{"select_key", 112}, {"ub_div_id", 2}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID = {
    {{"select_key", 120}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", True}},
    {{"select_key", 121}, {"ub_div_id", 1}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", True}},
    {{"select_key", 122}, {"ub_div_id", 2}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", True}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN = {
    {{"select_key", 130}, {"ub_div_id", 0}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 131}, {"ub_div_id", 1}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 132}, {"ub_div_id", 2}, {"block_align", True}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", False}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID = {
    {{"select_key", 140}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 141}, {"ub_div_id", 1}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 142}, {"ub_div_id", 2}, {"block_align", False}, {"e_level", SMALL_E},
     {"big_iid", True}, {"div_oid", False}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN = {
    {{"select_key", 150}, {"ub_div_id", 0}, {"block_align", True}, {"e_level", BIG_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 151}, {"ub_div_id", 1}, {"block_align", True}, {"e_level", BIG_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 152}, {"ub_div_id", 2}, {"block_align", True}, {"e_level", BIG_E},
     {"big_iid", True}, {"div_oid", False}}};

const vector<map<string, int>> SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID = {
    {{"select_key", 160}, {"ub_div_id", 0}, {"block_align", False}, {"e_level", BIG_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 161}, {"ub_div_id", 1}, {"block_align", False}, {"e_level", BIG_E},
     {"big_iid", True}, {"div_oid", False}},
    {{"select_key", 162}, {"ub_div_id", 2}, {"block_align", False}, {"e_level", BIG_E},
     {"big_iid", True}, {"div_oid", False}}};

enum EleByte {
    FP16_BYTE = 2, FP32_BYTE = 4, INT32_BYTE = 4, INT8_BYTE = 1, UINT8_BYTE = 1
};

int32_t _ceil_div(int32_t val, int32_t block)
{
    return (val + block - 1) / block;
}

class FrontLast {
public:
    int32_t front{0};
    int32_t last{0};
    int32_t times{0};

    void div_by_num(int32_t total, int32_t _times, int32_t min_part = 1)
    {
        front = _ceil_div(total, _times);
        times = _ceil_div(total, front);
        last = total - (times - 1) * front;

        while ((front < min_part or last < min_part) and front < total) {
            front = front + 1;
            times = _ceil_div(total, front);
            last = total - (times - 1) * front;
        }
    }

    void div_by_part(int32_t total, int32_t part, int32_t front_block = 1,
                    int32_t balance_num = 1)
    {
        times = _ceil_div(total, part);
        front = _ceil_div(total, times);
        front = _ceil_div(front, front_block) * front_block;

        if (front > part) {
            front = max(front / front_block * front_block, front_block);
        }
        times = _ceil_div(total, front);

        if (balance_num > 1) {
            times = _ceil_div(times, balance_num) * balance_num;
            front = _ceil_div(total, times);
            front = _ceil_div(front, front_block) * front_block;
            if (front > part) {
                front = max(front / front_block * front_block, front_block);
            }
            times = _ceil_div(total, front);
        }
        last = total - (times - 1) * front;
    }
};

class CommonScalar {
public:
    FrontLast num_segments_param;
    FrontLast num_segments_loop_param;
    FrontLast e_out_param;
    FrontLast ids_param;
    FrontLast e_out_loop_param;
    int32_t num_segments_core_num;
    float ub_use_rate = 0;
    int32_t input_once_num = 0;
    int32_t output_once_num = 0;

    int32_t select_key = 0;
    int32_t ids_last_burst_len = 0;

    int32_t repeat_time_front_part = 0;
    int32_t repeat_time_last_part = 0;
    int32_t align_scalar = 0;
    int32_t e_lenBurst_front = 0;
    int32_t e_lenBurst_last = 0;
    int32_t e_num_part_ub_num = 0;
};

class UnsortedSegmentTilingCal {
public:
    UnsortedSegmentTilingCal(int32_t _core_num, int32_t _num_segments, int32_t input_byte,
                                int32_t _ids_num, int32_t _e_num, int32_t _ub_size)
    {
        core_num = _core_num;
        num_segments = _num_segments;
        ids_num = _ids_num;
        e_num = _e_num;
        ub_size = _ub_size;
        mask = BYTE_FULL_MASK / input_byte;

        ele_num_per_block = BYTE_BLOCK / input_byte;
        e_max_by_stride = 65535 * ele_num_per_block;

        ids_once_num = ((ub_size / 5 / BYTE_BLOCK) * BYTE_BLOCK) / BYTE_INT32;
        res_ub_size = ub_size - ids_once_num * BYTE_INT32;
        obj_scalar = &scalars[0];

        int32_t ub_div_rates[UB_TEMPLATE_NUM][2] = {{1, 1}, {1, 2}, {2, 1}};
        for (int i = 0; i < UB_TEMPLATE_NUM; i++) {
            int32_t a = ub_div_rates[i][0];
            int32_t b = ub_div_rates[i][1];
            scalars[i].input_once_num = _floor_align_mask(res_ub_size / input_byte / (a + b) * a);
            scalars[i].output_once_num = _floor_align_mask(res_ub_size / input_byte / (a + b) * b);
        }
        _get_select_keys_for_compile();
    }
    ~UnsortedSegmentTilingCal()
    {
        obj_scalar = nullptr;
    }
    // ===================basic param===============================
    int32_t core_num;
    int32_t num_segments;
    int32_t mask;

    int32_t ele_num_per_block;
    int32_t e_max_by_stride;
    int32_t ub_size;

    int32_t ids_once_num;
    int32_t res_ub_size;
    vector<map<string, int>> select_keys_for_compile;
    // ===================scalar param==============================
    CommonScalar scalars[UB_TEMPLATE_NUM];
    CommonScalar* obj_scalar;

    int32_t ids_num;
    int32_t e_num;

    int32_t _floor_align_mask(int32_t val) {
        return max(val / mask * mask, mask);
    }

    void _get_tiling_params(int i, int32_t ids_ele_byte)
    {
        auto& scalar = scalars[i];
        if (e_num % ele_num_per_block == 0) {
            scalar.align_scalar = 0;
        } else {
            scalar.align_scalar = ele_num_per_block - (e_num - (e_num / ele_num_per_block) * ele_num_per_block);
        }

        int32_t e_max_block_ub = min(
                _floor_align_mask(scalar.output_once_num / num_segments), 255 * mask);
        scalar.e_out_param.div_by_part(e_num, e_max_block_ub, mask, core_num);
        scalar.e_out_loop_param.div_by_num(scalar.e_out_param.times, core_num);
        scalar.num_segments_core_num = max(min(core_num / scalar.e_out_loop_param.times,
                                _ceil_div(num_segments*scalar.e_out_param.last, mask)), 1);

        scalar.ids_param.div_by_part(ids_num,
                min(scalar.input_once_num / scalar.e_out_param.front, ids_once_num));
        scalar.e_num_part_ub_num = _ceil_div(scalar.e_out_param.front, mask) * mask;
        scalar.num_segments_param.div_by_part(num_segments,
            scalar.output_once_num / scalar.e_num_part_ub_num, 1, scalar.num_segments_core_num);
        float rate_out = float(scalar.e_out_param.front * scalar.num_segments_param.front) / scalar.output_once_num;
        float rate_in = float(scalar.e_out_param.front * scalar.ids_param.front) / scalar.input_once_num;
        scalar.ub_use_rate = rate_in * rate_out;
        scalar.num_segments_loop_param.div_by_num(scalar.num_segments_param.times, scalar.num_segments_core_num);
        scalar.e_lenBurst_front = _ceil_div(scalar.e_out_param.front, ele_num_per_block);
        scalar.e_lenBurst_last = _ceil_div(scalar.e_out_param.last, ele_num_per_block);
        scalar.repeat_time_front_part = _ceil_div(scalar.e_out_param.front, mask);
        scalar.repeat_time_last_part = _ceil_div(scalar.e_out_param.last, mask);
        scalar.ids_last_burst_len = _ceil_div(ids_num, BYTE_BLOCK / ids_ele_byte);

    }

    void _get_select_keys_for_compile()
    {
        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN.begin(),
        SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN.begin(),
        SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN.begin(),
        SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN.begin(),
        SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN.begin(),
        SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN.begin(),
        SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN.begin(),
        SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN.end());

        select_keys_for_compile.insert(select_keys_for_compile.end(),
        SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID.begin(),
        SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID.end());
    }

    vector<map<string, int>> _get_tiling_mode(int32_t num_segments_core_num)
    {
        int32_t e_vector_num = _ceil_div(e_num, mask);

        if (e_vector_num == 1 and e_num % ele_num_per_block != 0) {
            if (e_num >= ele_num_per_block) {
                if (ids_num <= ids_once_num) {
                    return SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID;
                }
                return SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID;
            }
            if (ids_num <= ids_once_num) {
                return SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID;
            }
            return SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID;
        } else if (e_num % ele_num_per_block == 0) {
            if (ids_num <= ids_once_num) {
                if (e_num <= e_max_by_stride) {
                    if (num_segments_core_num > 1) {
                        return SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN;
                    }
                    return SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN;
                }
                return SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN;
            }
            if (e_num <= e_max_by_stride) {
                if (num_segments_core_num > 1) {
                    return SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN;
                }
                return SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN;
            }
            return SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN;
        }
        if (ids_num <= ids_once_num) {
            if (e_num <= e_max_by_stride) {
                if (num_segments_core_num > 1) {
                    return SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID;
                }
                return SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID;
            }
            return SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID;
        }
        if (e_num <= e_max_by_stride) {
            if (num_segments_core_num > 1) {
                return SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID;
            }
            return SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID;
        }
        return SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID;
    }

    void write_tiling_params(OpRunInfo &run_info)
    {
        ByteBufferPut(run_info.tiling_data, obj_scalar->select_key);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_out_loop_param.times);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_out_loop_param.front);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_out_loop_param.last);
        ByteBufferPut(run_info.tiling_data, obj_scalar->ids_param.times);
        ByteBufferPut(run_info.tiling_data, obj_scalar->ids_param.front);
        ByteBufferPut(run_info.tiling_data, obj_scalar->ids_param.last);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_out_param.times);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_out_param.front);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_out_param.last);
        ByteBufferPut(run_info.tiling_data, obj_scalar->num_segments_param.times);
        ByteBufferPut(run_info.tiling_data, obj_scalar->num_segments_param.front);
        ByteBufferPut(run_info.tiling_data, obj_scalar->num_segments_param.last);
        ByteBufferPut(run_info.tiling_data, obj_scalar->num_segments_loop_param.times);
        ByteBufferPut(run_info.tiling_data, obj_scalar->num_segments_loop_param.front);
        ByteBufferPut(run_info.tiling_data, obj_scalar->num_segments_loop_param.last);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_num_part_ub_num);
        ByteBufferPut(run_info.tiling_data, e_num);
        ByteBufferPut(run_info.tiling_data, obj_scalar->repeat_time_front_part);
        ByteBufferPut(run_info.tiling_data, obj_scalar->repeat_time_last_part);
        ByteBufferPut(run_info.tiling_data, obj_scalar->align_scalar);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_lenBurst_front);
        ByteBufferPut(run_info.tiling_data, obj_scalar->e_lenBurst_last);
        ByteBufferPut(run_info.tiling_data, obj_scalar->ids_last_burst_len);
    }

    void print_tiling_params(const std::string &op_type)
    {
        GELOGD("op [%s] : select_key=%d", op_type.c_str(), obj_scalar->select_key);
        GELOGD("op [%s] : e_out_loop_param.times=%d", op_type.c_str(), obj_scalar->e_out_loop_param.times);
        GELOGD("op [%s] : e_out_loop_param.front=%d", op_type.c_str(), obj_scalar->e_out_loop_param.front);
        GELOGD("op [%s] : e_out_loop_param.last=%d", op_type.c_str(), obj_scalar->e_out_loop_param.last);
        GELOGD("op [%s] : ids_param.times=%d", op_type.c_str(), obj_scalar->ids_param.times);
        GELOGD("op [%s] : ids_param.front=%d", op_type.c_str(), obj_scalar->ids_param.front);
        GELOGD("op [%s] : ids_param.last=%d", op_type.c_str(), obj_scalar->ids_param.last);
        GELOGD("op [%s] : e_out_param.times=%d", op_type.c_str(), obj_scalar->e_out_param.times);
        GELOGD("op [%s] : e_out_param.front=%d", op_type.c_str(), obj_scalar->e_out_param.front);
        GELOGD("op [%s] : e_out_param.last=%d", op_type.c_str(), obj_scalar->e_out_param.last);

        GELOGD("op [%s] : num_segments_param.times=%d", op_type.c_str(), obj_scalar->num_segments_param.times);
        GELOGD("op [%s] : num_segments_param.front=%d", op_type.c_str(), obj_scalar->num_segments_param.front);
        GELOGD("op [%s] : num_segments_param.last=%d", op_type.c_str(), obj_scalar->num_segments_param.last);
        GELOGD("op [%s] : num_segments_loop_param.times=%d", op_type.c_str(),
            obj_scalar->num_segments_loop_param.times);
        GELOGD("op [%s] : num_segments_loop_param.front=%d", op_type.c_str(),
            obj_scalar->num_segments_loop_param.front);
        GELOGD("op [%s] : num_segments_loop_param.last=%d", op_type.c_str(), obj_scalar->num_segments_loop_param.last);

        GELOGD("op [%s] : e_num_part_ub_num=%d", op_type.c_str(), obj_scalar->e_num_part_ub_num);
        GELOGD("op [%s] : e_num=%d", op_type.c_str(), e_num);
        GELOGD("op [%s] : repeat_time_front_part=%d", op_type.c_str(), obj_scalar->repeat_time_front_part);
        GELOGD("op [%s] : repeat_time_last_part=%d", op_type.c_str(), obj_scalar->repeat_time_last_part);
        GELOGD("op [%s] : align_scalar=%d", op_type.c_str(), obj_scalar->align_scalar);
        GELOGD("op [%s] : e_lenBurst_front=%d", op_type.c_str(), obj_scalar->e_lenBurst_front);
        GELOGD("op [%s] : e_lenBurst_last=%d", op_type.c_str(), obj_scalar->e_lenBurst_last);
        GELOGD("op [%s] : ids_last_burst_len=%d", op_type.c_str(), obj_scalar->ids_last_burst_len);
    }
};

/******************COMMON_FUNCTION******************/
int32_t max(const int32_t a, const int32_t b)
{
    if (a > b) {
        return a;
    }
    return b;
}

int32_t min(const int32_t a, const int32_t b)
{
    if (a < b) {
        return a;
    }
    return b;
}

bool unsorted_segment_get_compile_params(
        const std::string &op_type, const nlohmann::json &op_compile_info_json,
        int32_t &core_num, int32_t &ub_size, int32_t &ub_tensor_num)
{
    using namespace nlohmann;
    if (op_compile_info_json == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null");
        return false;
    }
    const auto &allVars = op_compile_info_json["vars"];
    // core num
    if (allVars.count("core_num") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is null");
        return false;
    }
    core_num = allVars["core_num"].get<std::int32_t>();
    // ub size
    if (allVars.count("ub_size") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_size is null");
        return false;
    }
    ub_size = allVars["ub_size"].get<std::int32_t>();
    // ub tensor num
    if (allVars.count("ub_tensor_num") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_tensor_num is null");
        return false;
    }
    ub_tensor_num = allVars["ub_tensor_num"].get<std::int32_t>();
    GELOGD("op [%s] : GetCompileParams, core_num[%d], ub_size[%d].",
                 UNSORTED_SEGMENT_OP_TYPE.c_str(), core_num, ub_size);
    return true;
}

bool unsorted_segment_get_ele_dtype(const std::string &dtype, EleByte &elebyte)
{
    map<string, EleByte> dtype_map{{"float32", FP32_BYTE}, {"float16", FP16_BYTE},
        {"int32", INT32_BYTE}, {"int8", INT8_BYTE}, {"uint8", UINT8_BYTE}};

    if (dtype_map.find(dtype) != dtype_map.end()) {
        elebyte = dtype_map[dtype];
        return true;
    }
    return false;
}

// tiling function
bool UnsortedSegmentTiling(const std::string &op_type,
                            const TeOpParas &op_paras,
                            const nlohmann::json &op_compile_info_json,
                            OpRunInfo &run_info)
{
    GELOGI("op[%s] op tiling begin.", op_type.c_str());
    if (op_paras.inputs.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs is empty.");
        return false;
    }
    if (op_paras.inputs.size() < 2) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs.size() < 2.");
        return false;
    }
    if (op_paras.inputs[0].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input_data tensor is empty.");
        return false;
    }
    if (op_paras.inputs[1].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ids tensor is empty.");
        return false;
    }
    const std::vector<int64_t> &input_shape = op_paras.inputs[0].tensor[0].shape;
    const std::vector<int64_t> &ids_shape = op_paras.inputs[1].tensor[0].shape;
    const int32_t &input_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    const int32_t &ids_size = std::accumulate(ids_shape.begin(), ids_shape.end(), 1, std::multiplies<int>());
    GELOGD("op [%s] : input_size=%d, ids_size=%d", op_type.c_str(), input_size, ids_size);
    if (input_shape.size() < ids_shape.size()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim of input must be greater than or equal with dim of ids");
        return false;
    }
    for (unsigned i = 0; i < ids_shape.size(); i++) {
        GELOGD("op[%s] ids_shape[i] is %d", op_type.c_str(), ids_shape[i]);
        if (input_shape[i] != ids_shape[i]) {
            ge::OpsTwoInputShapeErrReport(
                    "UnsortedSegment", "input_data", "ids", "front shape of input must be equal with ids shape");
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "front shape of input must be equal with ids shape");
            return false;
        }
    }
    int32_t e_size = input_size / ids_size;
    GELOGD("op[%s] e_size is %d", op_type.c_str(), e_size);
    const std::string &input_dtype = op_paras.inputs[0].tensor[0].dtype;
    const std::string &ids_dtype = op_paras.inputs[1].tensor[0].dtype;
    bool flag = False;
    // get input dtype
    EleByte input_ele_byte;
    flag = unsorted_segment_get_ele_dtype(input_dtype, input_ele_byte);
    if (!flag) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_ele_byte failed.");
        return false;
    }

    EleByte output_ele_byte = input_ele_byte;
    int32_t output_ub_ele_num_one_row = BYTE_BLOCK / output_ele_byte;
    GELOGD("op[%s] output_ub_ele_num_one_row is %d", op_type.c_str(), output_ub_ele_num_one_row);
    // get ids dtype
    EleByte ids_ele_byte;
    flag = unsorted_segment_get_ele_dtype(ids_dtype, ids_ele_byte);
    if (!flag) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get ids_ele_byte failed.");
        return false;
    }
    // get compile info
    int32_t core_num = 1;
    int32_t ub_size = 256 * 1024;
    int32_t ub_tensor_num = 0;
    flag = unsorted_segment_get_compile_params(op_type, op_compile_info_json, core_num, ub_size, ub_tensor_num);
    if (!flag) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileParams failed.");
        return false;
    }

    std::string key_num_segments = "num_segments";
    if (op_paras.const_inputs.find(key_num_segments) == op_paras.const_inputs.end()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "num_segments not exists.");
        return false;
    }
    const int32_t *num_segments_ptr = reinterpret_cast<const int32_t *>(
            std::get<0>(op_paras.const_inputs.at(key_num_segments)));
    int32_t num_segments = *num_segments_ptr;
    GELOGD("op [%s] : num_segments=%d", op_type.c_str(), num_segments);

    //==============================================
    // tiling params
    UnsortedSegmentTilingCal params(core_num, num_segments, int32_t(input_ele_byte), ids_size, e_size, ub_size);

    for (int i = 0; i < UB_TEMPLATE_NUM; i++) {
        params._get_tiling_params(i,  ids_ele_byte);
    }
    auto select_mode = params._get_tiling_mode(params.scalars[0].num_segments_core_num);

    float ub_use_rate = 0;
    for (auto the_mode : select_mode) {
        auto* scalar = &(params.scalars[the_mode["ub_div_id"]]);
        if (scalar->ub_use_rate > ub_use_rate) {
            bool is_in_list = False;
            for (auto select_key : params.select_keys_for_compile) {
                if (the_mode["ub_div_id"] == select_key["ub_div_id"]) {
                    is_in_list = True;
                    break;
                }
            }
            if (not is_in_list) {
                continue;
            }
            ub_use_rate = scalar->ub_use_rate;
            scalar->select_key = the_mode["select_key"];
            params.obj_scalar = scalar;

            if (params.obj_scalar->ids_param.times == 1) {
                break;
            }
        }
    }

    int32_t need_core_num = params.obj_scalar->e_out_loop_param.times *
        params.obj_scalar->num_segments_loop_param.times;
    // write tiling params to run_info
    params.write_tiling_params(run_info);
    // cout tiling params
    params.print_tiling_params(op_type);
    // BlockDim, core num used in tik op
    run_info.block_dim = need_core_num;
    // workspace, null for tik op
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;

    GELOGI("op[%s] op tiling success.", op_type.c_str());
    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(UnsortedSegmentMin, UnsortedSegmentTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(UnsortedSegmentMax, UnsortedSegmentTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(UnsortedSegmentProd, UnsortedSegmentTiling);
}
// namespace optiling
