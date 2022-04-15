/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file sparse_apply_ftrl_d.cpp
 * \brief dynamic SparseApplyFtrl op tiling
 */
#include <string>
#include <map>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "op_log.h"

namespace optiling {
const int32_t PARAM_NUM_ADAGRAD = 5;
const int32_t INDEX_2 = 2;
const int32_t INDEX_3 = 3;
const int32_t INDEX_4 = 4;
const int32_t INDEX_5 = 5;

bool CalSparseApplyCommonTiling(const std::string& op_type, const nlohmann::json& op_info, OpRunInfo& run_info,
                                const std::vector<int64_t>& var_shape, const std::vector<int64_t>& grad_shape,
                                const std::vector<int64_t>& indices_shape)
{
    const auto& all_vars = op_info["vars"];
    if (all_vars.count("core_num") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileParams, get %s error", "core_num");
        return false;
    }
    int32_t core_num = all_vars["core_num"].get<std::int32_t>();

    if (all_vars.count("cache_threshold_col") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileParams, get %s error",
            "cache_threshold_col");
        return false;
    }
    int32_t cache_threshold_col = all_vars["cache_threshold_col"].get<std::int32_t>();

    int32_t var_rows = var_shape[0];
    int32_t num_indices = indices_shape[0];
    int32_t each_row_data_num = 1;
    if (grad_shape.size() > 1) {
        for (unsigned int i = 1; i < grad_shape.size(); i++) {
            each_row_data_num *= grad_shape[i];
        }
    }
    int32_t need_core_num = core_num;
    if (num_indices < core_num and each_row_data_num > cache_threshold_col) {
        need_core_num = num_indices;
        if (num_indices * 2 <= core_num and each_row_data_num >= 1024) {
            need_core_num = core_num / num_indices * num_indices;
            if (need_core_num > core_num) {
                need_core_num = core_num;
            }
            if (need_core_num <= 0) {
                need_core_num = 1;
            }
        }
    }
    int32_t num_multi_rows = 32; // must be 32 factor for align
    if (var_rows < num_multi_rows) {
        num_multi_rows = var_rows;
    }
    if (each_row_data_num <= cache_threshold_col) {
        need_core_num = var_rows / num_multi_rows;
    }

    int32_t partial_factor = core_num / num_indices;
    int32_t cols_per_core = 0;
    int32_t cols_last_core = 0;

    cols_per_core = each_row_data_num / partial_factor;
    cols_last_core = each_row_data_num - cols_per_core * (partial_factor - 1);

    // set tiling data
    ByteBufferPut(run_info.tiling_data, var_rows);
    ByteBufferPut(run_info.tiling_data, num_indices);
    ByteBufferPut(run_info.tiling_data, each_row_data_num);
    ByteBufferPut(run_info.tiling_data, need_core_num);
    ByteBufferPut(run_info.tiling_data, num_multi_rows);
    ByteBufferPut(run_info.tiling_data, cols_per_core);
    ByteBufferPut(run_info.tiling_data, cols_last_core);

    // block_dim, core num used in tik op
    run_info.block_dim = need_core_num;
    // workspace, null for tik op
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;
    return true;
}

bool SparseApplyAdadeltaDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                                OpRunInfo& run_info) {
    GELOGI("op[%s] tiling running.", op_type.c_str());
    if (op_info == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info json error.");
        return false;
    }
    // Numbers in brackets indicate dimensions
    if (op_paras.inputs.size() < 4 || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
    op_paras.inputs[2].tensor.empty() || op_paras.inputs[3].tensor.empty() || op_paras.inputs[4].tensor.empty() ||
    op_paras.inputs[5].tensor.empty() || op_paras.inputs[6].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape error.");
        return false;
    }

    std::vector<int64_t> var_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> grad_shape = op_paras.inputs[5].tensor[0].shape;
    std::vector<int64_t> indices_shape = op_paras.inputs[6].tensor[0].shape;

    return CalSparseApplyCommonTiling(op_type, op_info, run_info,
                                      var_shape, grad_shape, indices_shape);
}
// register tiling interface of the SparseApplyAdagradV2D op
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyAdadeltaD, SparseApplyAdadeltaDTiling);

bool SparseApplyAdagradV2DTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                                 OpRunInfo& run_info) {
    GELOGI("op[%s] tiling running.", op_type.c_str());
    if (op_info == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info json error.");
        return false;
    }
    if (op_paras.inputs.size() < 4 || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
    op_paras.inputs[2].tensor.empty() || op_paras.inputs[3].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape error.");
        return false;
    }

    std::vector<int64_t> var_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> grad_shape = op_paras.inputs[2].tensor[0].shape;
    std::vector<int64_t> indices_shape = op_paras.inputs[3].tensor[0].shape;

    return CalSparseApplyCommonTiling(op_type, op_info, run_info,
                                      var_shape, grad_shape, indices_shape);
}
// register tiling interface of the SparseApplyAdagradV2D op
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyAdagradD, SparseApplyAdagradV2DTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyAdagradV2D, SparseApplyAdagradV2DTiling);

bool SparseApplyAdagradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                              OpRunInfo& run_info) {
    GELOGI("op[%s] tiling running.", op_type.c_str());
    if (op_info == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info json error.");
        return false;
    }
    if (op_paras.inputs.size() < PARAM_NUM_ADAGRAD || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
    op_paras.inputs[INDEX_2].tensor.empty() || op_paras.inputs[INDEX_3].tensor.empty() || op_paras.inputs[INDEX_4].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape error.");
        return false;
    }

    std::vector<int64_t> var_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> grad_shape = op_paras.inputs[INDEX_3].tensor[0].shape;
    std::vector<int64_t> indices_shape = op_paras.inputs[INDEX_4].tensor[0].shape;

    return CalSparseApplyCommonTiling(op_type, op_info, run_info,
                                      var_shape, grad_shape, indices_shape);
}
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyAdagrad, SparseApplyAdagradTiling);


bool SparseApplyRMSPropDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                               OpRunInfo& run_info) {
    GELOGI("op[%s] tiling running.", op_type.c_str());
    if (op_info == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info json error.");
        return false;
    }
    // Numbers in brackets indicate dimensions
    if (op_paras.inputs.size() < 4 || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
    op_paras.inputs[2].tensor.empty() || op_paras.inputs[3].tensor.empty() || op_paras.inputs[4].tensor.empty() ||
    op_paras.inputs[5].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape error.");
        return false;
    }

    std::vector<int64_t> var_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> grad_shape = op_paras.inputs[4].tensor[0].shape;
    std::vector<int64_t> indices_shape = op_paras.inputs[5].tensor[0].shape;

    return CalSparseApplyCommonTiling(op_type, op_info, run_info,
                                      var_shape, grad_shape, indices_shape);
}
// register tiling interface of the SparseApplyAdagradV2D op
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyRMSPropD, SparseApplyRMSPropDTiling);

bool SparseApplyFtrlV2DTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                              OpRunInfo& run_info) {
    GELOGI("op[%s] tiling running.", op_type.c_str());
    if (op_info == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info json error.");
        return false;
    }
    // Numbers in brackets indicate dimensions
    if (op_paras.inputs.size() < 4 || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
    op_paras.inputs[2].tensor.empty() || op_paras.inputs[3].tensor.empty() || op_paras.inputs[4].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape error.");
        return false;
    }

    std::vector<int64_t> var_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> grad_shape = op_paras.inputs[3].tensor[0].shape;
    std::vector<int64_t> indices_shape = op_paras.inputs[4].tensor[0].shape;

    return CalSparseApplyCommonTiling(op_type, op_info, run_info,
                                      var_shape, grad_shape, indices_shape);
}
// register tiling interface of the SparseApplyAdagradV2D op
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyFtrlV2D, SparseApplyFtrlV2DTiling);
}  // namespace optiling
