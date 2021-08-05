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
 * \file assign.cc
 * \brief
 */
#include <string>
#include <math.h>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "external/graph/operator.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

// the minimum num for one core
const int64_t CORE_MINIMUM_NUM = 4;
const int64_t BLOCK_SIZE = 32;  // one block size is 32Bytes


using namespace ge;
namespace optiling {
bool GetAssignCompileParams(const nlohmann::json& op_compile_info, int64_t& core_num, int64_t& ub_size) {
    using namespace nlohmann;
    auto all_vars = op_compile_info["vars"];
    if (all_vars.count("core_num") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING("Assign", "GetAssignCompileParams, get core_num error");
        return false;
    }
    core_num = all_vars["core_num"].get<std::int64_t>();

    if (all_vars.count("ub_size") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING("Assign", "GetAssignCompileParams, get ub_size error");
        return false;
    }
    ub_size = all_vars["ub_size"].get<std::int64_t>();

    return true;
}

bool AssignTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                 utils::OpRunInfo& run_info) {
    using namespace ge;

    OP_TILING_CHECK(op_paras.GetInputsSize() < 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty"), return false);

    const ge::Shape& value_shape = op_paras.GetInputDesc(1).GetShape();
    const ge::DataType input_dtype = op_paras.GetInputDesc(1).GetDataType();

    int64_t value_num = value_shape.GetShapeSize();

    int64_t core_num = 0;
    int64_t ub_size = 0;
    if (!GetAssignCompileParams(op_info, core_num, ub_size)) {
        return false;
    }

    int64_t ele_size = GetSizeByDataType(input_dtype);

    int64_t ele_per_block = BLOCK_SIZE / ele_size;
    int64_t block_count = (value_num + ele_per_block - 1) / ele_per_block;

    int64_t sigment_total = (block_count + CORE_MINIMUM_NUM - 1) / CORE_MINIMUM_NUM;
    int64_t sigment_per_core = (sigment_total + core_num - 1) / core_num;

    int64_t core_used_num = sigment_per_core == 0 ? 1 : (sigment_total + sigment_per_core - 1) / sigment_per_core;
    int64_t block_per_core = sigment_per_core * CORE_MINIMUM_NUM;
    int64_t block_tail_core = block_count - (block_per_core * (core_used_num - 1));

    OP_LOGD(op_type.c_str(), "CompileParams, core_used_num = %d, block_per_core = %d, block_tail_core = %d",
           core_used_num, block_per_core, block_tail_core);

    run_info.AddTilingData(core_used_num);
    run_info.AddTilingData(block_per_core);
    run_info.AddTilingData(block_tail_core);

    run_info.SetBlockDim(core_num);
    OP_LOGI(op_type.c_str(), "tiling run success.");
    return true;
}

// register tiling interface of the Assign op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Assign, AssignTiling);
}  // namespace optiling
