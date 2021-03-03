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
    FrontLast()
    {
        front = 0;
        last = 0;
        times = 0;
    }
    int32_t front;
    int32_t last;
    int32_t times;

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

class UBParam {
public:
    FrontLast ids_param;
    FrontLast e_out_param;
    FrontLast e_out_loop_param;
    FrontLast num_segments_param;
    int32_t e_num_part_ub_num;
    int32_t num_segments_core_num;
    float ub_use_rate;
};

class UnsortedSegmentTilingCal {
public:
    UnsortedSegmentTilingCal(int32_t _core_num, int32_t _num_segments, int32_t input_byte,
                                int32_t _ids_num, int32_t _e_num, int32_t _ub_size)
    {
        // ===================basic param===============================
        core_num = _core_num;
        num_segments = _num_segments;
        ids_num = _ids_num;
        e_num = _e_num;
        ub_size = _ub_size;
        need_core_num = 0;
        num_segments_core_num = 0;
        mask = BYTE_FULL_MASK / input_byte;

        ele_num_per_block = BYTE_BLOCK / input_byte;
        e_max_by_stride = 65535 * ele_num_per_block;

        ids_once_num = ((ub_size / 5 / BYTE_BLOCK) * BYTE_BLOCK) / BYTE_INT32;
        res_ub_size = ub_size - ids_once_num * BYTE_INT32;

        int32_t ub_div_rates[UB_TEMPLATE_NUM][2] = {{1, 1}, {1, 2}, {2, 1}};
        for (int i = 0; i < UB_TEMPLATE_NUM; i++) {
            int32_t a = ub_div_rates[i][0];
            int32_t b = ub_div_rates[i][1];
            ub_tensor_nums[i][0] = _floor_align_mask(res_ub_size / (a + b) * a / input_byte);
            ub_tensor_nums[i][1] = _floor_align_mask(res_ub_size / (a + b) * b / input_byte);
        }
        _get_select_keys_for_compile();


        // ===================scalar param==============================
        select_key_input_scalar = 0;
        ids_last_burst_len_input_scalar = 0;
        repeat_time_front_part = 0;
        repeat_time_last_part = 0;
        align_scalar = 0;
        e_lenBurst_front_input_scalar = 0;
        e_lenBurst_last_input_scalar = 0;
        e_num_part_ub_num_input_scalar = 0;
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
    int32_t ub_tensor_nums[UB_TEMPLATE_NUM][2];
    int32_t need_core_num;
    vector<map<string, int>> select_keys_for_compile;
    int32_t num_segments_core_num;
    const int32_t min_num_segments_per_core = 4;
    // ===================scalar param==============================
    // common params
    int32_t select_key_input_scalar;
    FrontLast e_out_loop_param_input_scalar;
    FrontLast ids_param_input_scalar;
    FrontLast e_out_param_input_scalar;
    FrontLast num_segments_param;
    FrontLast num_segments_loop_param;
    int32_t e_num_part_ub_num_input_scalar;

    // ids params
    int32_t ids_num;
    int32_t ids_last_burst_len_input_scalar;

    // e num params
    int32_t e_num;
    int32_t repeat_time_front_part;
    int32_t repeat_time_last_part;
    int32_t align_scalar;

    int32_t e_lenBurst_front_input_scalar;
    int32_t e_lenBurst_last_input_scalar;

    int32_t _floor_align_mask(int32_t val) {
        return max(val / mask * mask, mask);
    }

    void _get_tiling_params(const vector<map<string, int>>& select_mode, int32_t ids_ele_byte)
    {
        if (e_num % ele_num_per_block == 0) {
            align_scalar = 0;
        } else {
            align_scalar = ele_num_per_block - (e_num - (e_num / ele_num_per_block) * ele_num_per_block);
        }

        UBParam ub_params[UB_TEMPLATE_NUM];
        for (int i = 0; i < UB_TEMPLATE_NUM; i++) {
            int32_t input_once_num_tmp = ub_tensor_nums[i][0];
            int32_t output_once_num_tmp = ub_tensor_nums[i][1];

            int32_t e_max_block_ub = min(
                    _floor_align_mask(output_once_num_tmp / num_segments), 255 * mask);
            ub_params[i].e_out_param.div_by_part(e_num, e_max_block_ub, mask, core_num);

            ub_params[i].e_out_loop_param.div_by_num(ub_params[i].e_out_param.times, core_num);
            ub_params[i].num_segments_core_num = max(min(core_num / ub_params[i].e_out_loop_param.times,
                                    _ceil_div(num_segments, min_num_segments_per_core)), 1);

            ub_params[i].ids_param.div_by_part(ids_num,
                    min(input_once_num_tmp / ub_params[i].e_out_param.front, ids_once_num));

            ub_params[i].e_num_part_ub_num = _ceil_div(ub_params[i].e_out_param.front, mask) * mask;

            ub_params[i].num_segments_param.div_by_part(num_segments,
             output_once_num_tmp / ub_params[i].e_num_part_ub_num, 1, ub_params[i].num_segments_core_num);
            float rate_out =
             float(ub_params[i].e_out_param.front * ub_params[i].num_segments_param.front) / output_once_num_tmp;
            float rate_in = float(ub_params[i].e_out_param.front * ub_params[i].ids_param.front) / input_once_num_tmp;
            ub_params[i].ub_use_rate = rate_in * rate_out;
        }

        float ub_use_rate = 0;
        for (auto the_mode : select_mode) {
            auto ub_param = ub_params[the_mode["ub_div_id"]];
            if (ub_param.ub_use_rate > ub_use_rate) {
                bool is_in_list = False;
                for (auto select_key : select_keys_for_compile) {
                    if (the_mode["ub_div_id"] == select_key["ub_div_id"]) {
                        is_in_list = True;
                        break;
                    }
                }
                if (not is_in_list) {
                    continue;
                }
                ub_use_rate = ub_param.ub_use_rate;
                select_key_input_scalar = the_mode["select_key"];
                ids_param_input_scalar = ub_param.ids_param;
                e_out_param_input_scalar = ub_param.e_out_param;
                e_out_loop_param_input_scalar = ub_param.e_out_loop_param;
                num_segments_param = ub_param.num_segments_param;
                e_num_part_ub_num_input_scalar = ub_param.e_num_part_ub_num;
                num_segments_core_num = ub_param.num_segments_core_num;
                if (ids_param_input_scalar.times == 1) {
                    break;
                }
            }
        }
        num_segments_loop_param.div_by_num(num_segments_param.times, num_segments_core_num);

        need_core_num = e_out_loop_param_input_scalar.times * num_segments_loop_param.times;
        e_lenBurst_front_input_scalar = _ceil_div(e_out_param_input_scalar.front, ele_num_per_block);
        e_lenBurst_last_input_scalar = _ceil_div(e_out_param_input_scalar.last, ele_num_per_block);

        repeat_time_front_part = _ceil_div(e_out_param_input_scalar.front, mask);
        repeat_time_last_part = _ceil_div(e_out_param_input_scalar.last, mask);
        ids_last_burst_len_input_scalar = _ceil_div(ids_num, BYTE_BLOCK / ids_ele_byte);
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

    vector<map<string, int>> _get_tiling_mode()
    {
        int32_t e_vector_num = _ceil_div(e_num, mask);
        int32_t num_segments_estimate_block =
            max(min(core_num / e_vector_num, _ceil_div(num_segments, min_num_segments_per_core)), 1);

        if (e_vector_num == 1 and e_num % ele_num_per_block != 0) {
            if (e_num >= ele_num_per_block) {
                if (e_num >= ele_num_per_block) {
                    return SELECT_KEY_DIV_OID_ONE_E_SMALL_IID_SMALL_OID;
                }
                return SELECT_KEY_DIV_OID_ONE_E_BIG_IID_SMALL_OID;
            }
            if (e_num >= ele_num_per_block) {
                return SELECT_KEY_DIV_OID_BLOCK_E_SMALL_IID_SMALL_OID;
            }
            return SELECT_KEY_DIV_OID_BLOCK_E_BIG_IID_SMALL_OID;
        } else if (e_num % ele_num_per_block == 0) {
            if (ids_num <= ids_once_num) {
                if (e_num <= e_max_by_stride) {
                    if (num_segments_estimate_block > 1) {
                        return SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID_ALIGN;
                    }
                    return SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID_ALIGN;
                }
                return SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID_ALIGN;
            }
            if (e_num <= e_max_by_stride) {
                if (num_segments_estimate_block > 1) {
                    return SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID_ALIGN;
                }
                return SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID_ALIGN;
            }
            return SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID_ALIGN;
        }
        if (ids_num <= ids_once_num) {
            if (e_num <= e_max_by_stride) {
                if (num_segments_estimate_block > 1) {
                    return SELECT_KEY_DIV_E_OID_SMALL_E_SMALL_IID_SMALL_OID;
                }
                return SELECT_KEY_DIV_E_SMALL_E_SMALL_IID_SMALL_OID;
            }
            return SELECT_KEY_DIV_E_BIG_E_SMALL_IID_SMALL_OID;
        }
        if (e_num <= e_max_by_stride) {
            if (num_segments_estimate_block > 1) {
                return SELECT_KEY_DIV_E_OID_SMALL_E_BIG_IID_SMALL_OID;
            }
            return SELECT_KEY_DIV_E_SMALL_E_BIG_IID_SMALL_OID;
        }
        return SELECT_KEY_DIV_E_BIG_E_BIG_IID_SMALL_OID;
    }

    void UnsortedSegmentWriteTilingParams(OpRunInfo &run_info)
    {
        ByteBufferPut(run_info.tiling_data, select_key_input_scalar);
        ByteBufferPut(run_info.tiling_data, e_out_loop_param_input_scalar.times);
        ByteBufferPut(run_info.tiling_data, e_out_loop_param_input_scalar.front);
        ByteBufferPut(run_info.tiling_data, e_out_loop_param_input_scalar.last);
        ByteBufferPut(run_info.tiling_data, ids_param_input_scalar.times);
        ByteBufferPut(run_info.tiling_data, ids_param_input_scalar.front);
        ByteBufferPut(run_info.tiling_data, ids_param_input_scalar.last);
        ByteBufferPut(run_info.tiling_data, e_out_param_input_scalar.times);
        ByteBufferPut(run_info.tiling_data, e_out_param_input_scalar.front);
        ByteBufferPut(run_info.tiling_data, e_out_param_input_scalar.last);

        ByteBufferPut(run_info.tiling_data, num_segments_param.times);
        ByteBufferPut(run_info.tiling_data, num_segments_param.front);
        ByteBufferPut(run_info.tiling_data, num_segments_param.last);
        ByteBufferPut(run_info.tiling_data, num_segments_loop_param.times);
        ByteBufferPut(run_info.tiling_data, num_segments_loop_param.front);
        ByteBufferPut(run_info.tiling_data, num_segments_loop_param.last);

        ByteBufferPut(run_info.tiling_data, e_num_part_ub_num_input_scalar);

        ByteBufferPut(run_info.tiling_data, e_num);
        ByteBufferPut(run_info.tiling_data, repeat_time_front_part);
        ByteBufferPut(run_info.tiling_data, repeat_time_last_part);
        ByteBufferPut(run_info.tiling_data, align_scalar);
        ByteBufferPut(run_info.tiling_data, e_lenBurst_front_input_scalar);
        ByteBufferPut(run_info.tiling_data, e_lenBurst_last_input_scalar);
        ByteBufferPut(run_info.tiling_data, ids_last_burst_len_input_scalar);
    }

    void UnsortedSegmentPrintTilingParams(const std::string &op_type)
    {
        GELOGD("op [%s] : select_key_input_scalar=%d", op_type.c_str(), select_key_input_scalar);
        GELOGD("op [%s] : e_out_loop_param_input_scalar.times=%d",
         op_type.c_str(), e_out_loop_param_input_scalar.times);
        GELOGD("op [%s] : e_out_loop_param_input_scalar.front=%d",
         op_type.c_str(), e_out_loop_param_input_scalar.front);
        GELOGD("op [%s] : e_out_loop_param_input_scalar.last=%d", op_type.c_str(), e_out_loop_param_input_scalar.last);
        GELOGD("op [%s] : ids_param_input_scalar.times=%d", op_type.c_str(), ids_param_input_scalar.times);
        GELOGD("op [%s] : ids_param_input_scalar.front=%d", op_type.c_str(), ids_param_input_scalar.front);
        GELOGD("op [%s] : ids_param_input_scalar.last=%d", op_type.c_str(), ids_param_input_scalar.last);
        GELOGD("op [%s] : e_out_param_input_scalar.times=%d", op_type.c_str(), e_out_param_input_scalar.times);
        GELOGD("op [%s] : e_out_param_input_scalar.front=%d", op_type.c_str(), e_out_param_input_scalar.front);
        GELOGD("op [%s] : e_out_param_input_scalar.last=%d", op_type.c_str(), e_out_param_input_scalar.last);

        GELOGD("op [%s] : num_segments_param.times=%d", op_type.c_str(), num_segments_param.times);
        GELOGD("op [%s] : num_segments_param.front=%d", op_type.c_str(), num_segments_param.front);
        GELOGD("op [%s] : num_segments_param.last=%d", op_type.c_str(), num_segments_param.last);
        GELOGD("op [%s] : num_segments_loop_param.times=%d", op_type.c_str(), num_segments_loop_param.times);
        GELOGD("op [%s] : num_segments_loop_param.front=%d", op_type.c_str(), num_segments_loop_param.front);
        GELOGD("op [%s] : num_segments_loop_param.last=%d", op_type.c_str(), num_segments_loop_param.last);

        GELOGD("op [%s] : e_num_part_ub_num_input_scalar=%d", op_type.c_str(), e_num_part_ub_num_input_scalar);
        GELOGD("op [%s] : e_num=%d", op_type.c_str(), e_num);
        GELOGD("op [%s] : repeat_time_front_part=%d",
         op_type.c_str(), repeat_time_front_part);
        GELOGD("op [%s] : repeat_time_last_part=%d", op_type.c_str(), repeat_time_last_part);
        GELOGD("op [%s] : align_scalar=%d", op_type.c_str(), align_scalar);
        GELOGD("op [%s] : e_lenBurst_front_input_scalar=%d", op_type.c_str(), e_lenBurst_front_input_scalar);
        GELOGD("op [%s] : e_lenBurst_last_input_scalar=%d", op_type.c_str(), e_lenBurst_last_input_scalar);
        GELOGD("op [%s] : ids_last_burst_len_input_scalar=%d", op_type.c_str(), ids_last_burst_len_input_scalar);
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

bool UnsortedSegmentGetUssCompileParams(
        const std::string &op_type, const nlohmann::json &op_compile_info_json,
        int32_t &core_num, int32_t &ub_size, int32_t &ub_tensor_num)
{
    using namespace nlohmann;
    if (op_compile_info_json == nullptr) {
        ge::OpsGetCompileParamsErrReport("UnsortedSegment", "op_compile_info_json");
        OP_LOGE(op_type.c_str(), "op_compile_info_json is null");
        return false;
    }
    const auto &allVars = op_compile_info_json["vars"];
    // core num
    if (allVars.count("core_num") == 0) {
        ge::OpsGetCompileParamsErrReport("UnsortedSegment", "core_num");
        OP_LOGE(op_type.c_str(), "core_num is null");
        return false;
    }
    core_num = allVars["core_num"].get<std::int32_t>();
    // ub size
    if (allVars.count("ub_size") == 0) {
        ge::OpsGetCompileParamsErrReport("UnsortedSegment", "ub_size");
        OP_LOGE(op_type.c_str(), "ub_size is null");
        return false;
    }
    ub_size = allVars["ub_size"].get<std::int32_t>();
    // ub tensor num
    if (allVars.count("ub_tensor_num") == 0) {
        ge::OpsGetCompileParamsErrReport("UnsortedSegment", "ub_tensor_num");
        OP_LOGE(op_type.c_str(), "ub_tensor_num is null");
        return false;
    }
    ub_tensor_num = allVars["ub_tensor_num"].get<std::int32_t>();
    GELOGD("op [%s] : GetCompileParams, core_num[%d], ub_size[%d].",
                 UNSORTED_SEGMENT_OP_TYPE.c_str(), core_num, ub_size);
    return true;
}

bool UnsortedSegmentGetEleDtype(const std::string &dtype, EleByte &elebyte)
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
        ge::OpsOneInputShapeErrReport("UnsortedSegment", "op_paras.inputs", "op_paras.inputs is empty.");
        OP_LOGE(op_type.c_str(), "op_paras.inputs is empty.");
        return false;
    }
    if (op_paras.inputs.size() < 2) {
        ge::OpsOneInputShapeErrReport("UnsortedSegment", "op_paras.inputs", "op_paras.inputs.size() < 2.");
        OP_LOGE(op_type.c_str(), "op_paras.inputs.size() < 2.");
        return false;
    }
    if (op_paras.inputs[0].tensor.empty()) {
        ge::OpsOneInputShapeErrReport("UnsortedSegment", "input_data", "input_data tensor is empty.");
        OP_LOGE(op_type.c_str(), "input_data tensor is empty.");
        return false;
    }
    if (op_paras.inputs[1].tensor.empty()) {
        ge::OpsOneInputShapeErrReport("UnsortedSegment", "ids", "ids tensor is empty.");
        OP_LOGE(op_type.c_str(), "ids tensor is empty.");
        return false;
    }
    const std::vector<int64_t> &input_shape = op_paras.inputs[0].tensor[0].shape;
    const std::vector<int64_t> &ids_shape = op_paras.inputs[1].tensor[0].shape;
    const int32_t &input_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    const int32_t &ids_size = std::accumulate(ids_shape.begin(), ids_shape.end(), 1, std::multiplies<int>());
    GELOGD("op [%s] : input_size=%d, ids_size=%d", op_type.c_str(), input_size, ids_size);
    if (input_shape.size() < ids_shape.size()) {
        ge::OpsTwoInputShapeErrReport(
                "UnsortedSegment", "input_data", "ids", "dim of input must be greater than or equal with dim of ids");
        OP_LOGE(op_type.c_str(), "dim of input must be greater than or equal with dim of ids");
        return false;
    }
    for (unsigned i = 0; i < ids_shape.size(); i++) {
        GELOGD("op[%s] ids_shape[i] is %d", op_type.c_str(), ids_shape[i]);
        if (input_shape[i] != ids_shape[i]) {
            ge::OpsTwoInputShapeErrReport(
                    "UnsortedSegment", "input_data", "ids", "front shape of input must be equal with ids shape");
            OP_LOGE(op_type.c_str(), "front shape of input must be equal with ids shape");
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
    flag = UnsortedSegmentGetEleDtype(input_dtype, input_ele_byte);
    if (!flag) {
        ge::OpsOneInputShapeErrReport("UnsortedSegment", "input_data", "get input_ele_byte failed.");
        OP_LOGE("op[%s] get input_ele_byte failed.", op_type.c_str());
        return false;
    }

    EleByte output_ele_byte = input_ele_byte;
    int32_t output_ub_ele_num_one_row = BYTE_BLOCK / output_ele_byte;
    GELOGD("op[%s] output_ub_ele_num_one_row is %d", op_type.c_str(), output_ub_ele_num_one_row);
    // get ids dtype
    EleByte ids_ele_byte;
    flag = UnsortedSegmentGetEleDtype(ids_dtype, ids_ele_byte);
    if (!flag) {
        ge::OpsOneInputShapeErrReport("UnsortedSegment", "ids", "get ids_ele_byte failed.");
        OP_LOGE("op[%s] get ids_ele_byte failed.", op_type.c_str());
        return false;
    }
    // get compile info
    int32_t core_num = 1;
    int32_t ub_size = 256 * 1024;
    int32_t ub_tensor_num = 0;
    flag = UnsortedSegmentGetUssCompileParams(op_type, op_compile_info_json, core_num, ub_size, ub_tensor_num);
    if (!flag) {
        OP_LOGE("op[%s] GetCompileParams failed.", op_type.c_str());
        return false;
    }

    std::string key_num_segments = "num_segments";
    if (op_paras.const_inputs.find(key_num_segments) == op_paras.const_inputs.end()) {
        ge::OpsOneInputShapeErrReport("UnsortedSegment", "num_segments", "num_segments not exists.");
        OP_LOGE("op[%s] num_segments not exists.", op_type.c_str());
        return false;
    }
    const int32_t *num_segments_ptr = reinterpret_cast<const int32_t *>(
            std::get<0>(op_paras.const_inputs.at(key_num_segments)));
    int32_t num_segments = *num_segments_ptr;
    GELOGD("op [%s] : num_segments=%d", op_type.c_str(), num_segments);

    //==============================================
    // tiling params
    UnsortedSegmentTilingCal params(core_num, num_segments, int32_t(input_ele_byte), ids_size, e_size, ub_size);
    auto select_mode = params._get_tiling_mode();
    params._get_tiling_params(select_mode, ids_ele_byte);

    // write tiling params to run_info
    params.UnsortedSegmentWriteTilingParams(run_info);
    // cout tiling params
    params.UnsortedSegmentPrintTilingParams(op_type);
    // BlockDim, core num used in tik op
    run_info.block_dim = params.need_core_num;
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
