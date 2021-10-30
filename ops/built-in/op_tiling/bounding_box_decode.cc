/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file bounding_box_decode.cc
 * \brief dynamic shape tiling of bounding_box_decode
 */
#include<string>
#include <cmath>
#include <nlohmann/json.hpp>
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
using namespace ge;
using namespace std;

// the number of data contained in each coordinate box
static const int32_t NUMBER_FOUR = 4;
// The maximum number of float16 data that can be stored in UB with pingpang
static const int32_t MAX_UB_ELEMENT_NUMBER_FP16 = 8704;

struct TilingInfo {
    // calcu param
    int32_t each_core_start_address;
    int32_t loop_cycle;
    int32_t start_block_address;
    int32_t block_number;
    int32_t repeat_times;

    // for arg last dim
    int32_t ub_max_size;
    int32_t rois_data_each_block;
    int32_t coreNum;
    int32_t each_repeat_block_number;
};

static void InitTilingParam(TilingInfo& param) {
    param.each_core_start_address = 0;
    param.loop_cycle = 0;
    param.start_block_address = 0;
    param.block_number = 0;
    param.repeat_times = 0;
    param.ub_max_size = 0;
    param.rois_data_each_block = 0;
    param.coreNum = 0;
    param.each_repeat_block_number = 0;
}

static void SetTilingParam(const TilingInfo& param, OpRunInfo& run_info) {
    ByteBufferPut(run_info.tiling_data, param.each_core_start_address);
    ByteBufferPut(run_info.tiling_data, param.loop_cycle);
    ByteBufferPut(run_info.tiling_data, param.start_block_address);
    ByteBufferPut(run_info.tiling_data, param.block_number);
    ByteBufferPut(run_info.tiling_data, param.repeat_times);
}

static void PrintParam(const TilingInfo& param) {
    // output param
    OP_LOGD("BoundingBoxDecodeTiling ", "each_core_start_address:%d", param.each_core_start_address);
    OP_LOGD("BoundingBoxDecodeTiling ", "loop_cycle:%d", param.loop_cycle);
    OP_LOGD("BoundingBoxDecodeTiling ", "start_block_address:%d", param.start_block_address);
    OP_LOGD("BoundingBoxDecodeTiling ", "block_number:%d", param.block_number);
    OP_LOGD("BoundingBoxDecodeTiling ", "repeat_times:%d", param.repeat_times);

    // input param
    OP_LOGD("BoundingBoxDecodeTiling ", "ub_max_size:%d", param.ub_max_size);
    OP_LOGD("BoundingBoxDecodeTiling ", "rois_data_each_block:%d", param.rois_data_each_block);
    OP_LOGD("BoundingBoxDecodeTiling ", "coreNum:%d", param.coreNum);
    OP_LOGD("BoundingBoxDecodeTiling ", "each_repeat_block_number:%d", param.each_repeat_block_number);
}

static void GetCoreParam(int32_t core_num, int32_t rois_data_each_block, int32_t element_number,
                         int32_t& each_core_start_address, int32_t& each_core_calcul_num) {
    each_core_start_address = (element_number / (core_num * NUMBER_FOUR)) * NUMBER_FOUR;
    if (element_number % (core_num * NUMBER_FOUR) == 0) {
        if (each_core_start_address % rois_data_each_block == 0) {
            each_core_calcul_num = each_core_start_address;
        } else {
            each_core_calcul_num = (each_core_start_address / rois_data_each_block + 1) *
                                    rois_data_each_block;
        }
    } else {
        each_core_calcul_num = element_number - each_core_start_address * (core_num - 1);
        if (each_core_calcul_num % rois_data_each_block != 0) {
            each_core_calcul_num = (each_core_calcul_num / rois_data_each_block + 1) *
                                    rois_data_each_block;
        }
    }
    return;
}

void GetLoopParam(int32_t each_core_calcul_num, int32_t rois_data_each_block,
                  int32_t& loop_cycle, int32_t& start_block_address, int32_t& block_number_loop) {
    block_number_loop = each_core_calcul_num / rois_data_each_block;
    start_block_address = block_number_loop / loop_cycle;
    if (loop_cycle > 1) {
        if (block_number_loop % loop_cycle != 0) {
            int32_t block_number = 0;
            block_number = block_number_loop - start_block_address * (loop_cycle - 1);
            while ((block_number * loop_cycle < block_number_loop) ||
                   (block_number * rois_data_each_block > MAX_UB_ELEMENT_NUMBER_FP16)) {
                loop_cycle += 1;
                start_block_address = block_number_loop / loop_cycle;
                block_number = block_number_loop - start_block_address * (loop_cycle - 1);
            }
            block_number_loop = block_number;
        } else {
            block_number_loop = start_block_address;
        }
    }
    return;
}

int32_t GetRepeatCycle(int32_t block_number, int32_t each_repeat_block_number) {
    int32_t repeat_times = 0;
    if (block_number < each_repeat_block_number) {
        repeat_times = 1;
    } else if (block_number % each_repeat_block_number == 0) {
        repeat_times = block_number / each_repeat_block_number;
    } else {
        repeat_times = block_number / each_repeat_block_number + 1;
    }
    return repeat_times;
}

int32_t GetLoopCycle(int32_t each_core_calcul_num, int32_t ub_max_size) {
    int32_t loop_cycle = each_core_calcul_num / ub_max_size;
    if ((each_core_calcul_num % ub_max_size) != 0) {
        loop_cycle = loop_cycle + 1;
    }
    return loop_cycle;
}

/* @brief: calculate the vector calculation repeat_times with each pingpang of each coreNum
   @return repeat_times:the vector calcu repeat_times
 */
bool GetParam(const TeOpParas& op_paras, TilingInfo& param) {
    std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
    int32_t rois_data_each_block = param.rois_data_each_block;

    // calcu element_number
    int64_t element_total = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
        element_total *= input_shape[i];
    }
    int32_t element_number = element_total;
    if (0 == element_number) {
        VECTOR_INNER_ERR_REPORT_TILIING("BoundingBoxDecode", "op[BoundingBoxDecodeTiling]:GetParam fail.element_number is zero");
        return false;
    }

    // calcu each_core_calcul_num and each_core_start_address
    int32_t each_core_calcul_num = 0;
    int32_t each_core_start_address = 0;
    GetCoreParam(param.coreNum, rois_data_each_block, element_number,
                 each_core_start_address, each_core_calcul_num);

    // calcu loop_cycle:the number of pingpang cycles per coreNum
    int32_t loop_cycle = GetLoopCycle(each_core_calcul_num, param.ub_max_size);

    // calcu block_number
    int32_t block_number = 0;
    int32_t start_block_address = 0;
    GetLoopParam(each_core_calcul_num, rois_data_each_block,
                 loop_cycle, start_block_address, block_number);

    // calcu repeat_times
    int32_t repeat_times = GetRepeatCycle(block_number, param.each_repeat_block_number);

    // fill calcu param
    param.each_core_start_address = each_core_start_address;
    param.loop_cycle = loop_cycle;
    param.start_block_address = start_block_address;
    param.block_number = block_number;
    param.repeat_times = repeat_times;
    return true;
}

bool GetCompileParam(const std::string& opType,
                     const nlohmann::json& opCompileInfoJson,
                     TilingInfo& param) {
    using namespace nlohmann;

    const auto& allVars = opCompileInfoJson["vars"];
    if (allVars.count("core_num") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:GetCompileParam, get core_num error");
        return false;
    }
    param.coreNum = allVars["core_num"].get<std::int32_t>();

    if (allVars.count("rois_data_each_block") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:GetCompileParam, get rois_data_each_block error");
        return false;
    }
    param.rois_data_each_block = allVars["rois_data_each_block"].get<std::int32_t>();

    if (allVars.count("each_repeat_block_number") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:GetCompileParam, get each_repeat_block_number error");
        return false;
    }
    param.each_repeat_block_number = allVars["each_repeat_block_number"].get<std::int32_t>();

    if (allVars.count("ub_max_size") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:GetCompileParam, get ub_max_size error");
        return false;
    }
    param.ub_max_size = allVars["ub_max_size"].get<std::int32_t>();

    return true;
}

bool CheckParam(const std::string& opType, TilingInfo& param) {
    if (0 == param.coreNum) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:check fail.coreNum is zero");
        return false;
    }

    if (0 == param.rois_data_each_block) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:check fail.rois_data_each_block is zero");
        return false;
    }

    if (0 == param.each_repeat_block_number) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:check fail.each_repeat_block_number is zero");
        return false;
    }

    if (0 == param.ub_max_size) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[BoundingBoxDecodeTiling]:check fail.ub_max_size is zero");
        return false;
    }

    return true;
}

/*
 * @brief: tiling function of op
 * @param: [in] opType: opType of op
 * @param: [in] opParas: inputs/outputs/attrs of op
 * @param: [in] op_info: compile time generated info of op
 * @param: [out] runInfo: result data
 * @return bool: success or not
*/
bool BoundingBoxDecodeTiling(const string& op_type, const TeOpParas& op_paras,
                             const nlohmann::json& op_info, OpRunInfo& run_info) {
    OP_LOGD(op_type.c_str(), "BoundingBoxDecodeTiling running.");

    TilingInfo param;
    InitTilingParam(param);
    if (!GetCompileParam(op_type, op_info, param)) {
        return false;
    }

    if (!CheckParam(op_type, param)) {
        return false;
    }

    if (!GetParam(op_paras, param)) {
        return false;
    }

    SetTilingParam(param, run_info);
    PrintParam(param);
    run_info.block_dim = param.coreNum;

    OP_LOGD(op_type.c_str(), "BoundingBoxDecodeTiling end.");
    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(BoundingBoxDecode, BoundingBoxDecodeTiling);
}  // namespace optiling
