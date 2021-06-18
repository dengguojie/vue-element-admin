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
 * \file sparse_apply_ftrl_d.cpp
 * \brief dynamic SparseApplyFtrl op tiling
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include <map>

namespace optiling {

bool CheckSparseApplyAdagradV2DTensorShape(const std::string& op_type,
                            const TeOpParas& op_paras, int32_t& var_row_elem) {
    std::vector<int64_t> var_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> accum_shape = op_paras.inputs[1].tensor[0].shape;
    std::vector<int64_t> grad_shape = op_paras.inputs[2].tensor[0].shape;
    std::vector<int64_t> indices_shape = op_paras.inputs[3].tensor[0].shape;
    if (var_shape.size() == 0 || accum_shape.size() == 0 || grad_shape.size() == 0 || indices_shape.size() == 0) {
        ge::OpsOneInputShapeErrReport(op_type.c_str(), "indices",
                                      "Size of var_shape or accum_shape or grad_shape or indices_shape is 0");
        OP_LOGE(op_type.c_str(), "Size of var_shape, accum_shape, grad_shape, indices_shape is %u, %u, %u, %u",
                                      var_shape.size(), accum_shape.size(), grad_shape.size(), indices_shape.size());
        return false;
    }
    int32_t var_dims = var_shape.size();

    if (indices_shape[0] != grad_shape[0]) {
        ge::OpsOneInputShapeErrReport(op_type.c_str(), "indices",
                                      "the shape 0 of indices must be equal to the shape 0 of grad");
        OP_LOGE(op_type.c_str(), "op [SparseApplyAdagradV2DTiling] : grad shape[0] must be equal to indices shape[0]");
        return false;
    }
    return true;
}

/*
 * @brief: tiling function of op
 * @param [in] op_type: op_type of the op
 * @param [in] op_paras: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool SparseApplyAdagradV2DTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                            OpRunInfo& run_info) {
    GELOGI("op[%s] tiling running.", op_type.c_str());
    if (op_info == nullptr) {
        OP_LOGE(op_type.c_str(), "SparseApplyAdagradV2DTiling: op_info json error.");
        return false;
    }
    if (op_paras.inputs.size() < 4 || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
    op_paras.inputs[2].tensor.empty() || op_paras.inputs[3].tensor.empty()) {
        ge::OpsOneInputShapeErrReport(op_type.c_str(), "var or accum",
                                  "The length of inputs is less than 4 or the inputs is empty");
        OP_LOGE(op_type.c_str(), "SparseApplyAdagradV2DTiling: input shape error.");
        return false;
    }

    int32_t var_row_elem = 1;
    // check inputs shape
    bool ret = CheckSparseApplyAdagradV2DTensorShape(op_type, op_paras, var_row_elem);
    if (!ret) {
        OP_LOGE(op_type.c_str(), "op[parseApplyFtrlTiling] SparseApplyAdagradV2DTiling: inputs shape are invalid.");
        return ret;
    }

    std::vector<int64_t> var_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> grad_shape = op_paras.inputs[2].tensor[0].shape;
    std::vector<int64_t> indices_shape = op_paras.inputs[3].tensor[0].shape;
    int32_t var_rows = var_shape[0];
    int32_t num_indices = indices_shape[0];
    int32_t each_row_data_num = 1;
    if (grad_shape.size() > 1) {
        for (int i = 1; i < grad_shape.size(); i++) {
            each_row_data_num *= grad_shape[i];
        }
    }

    const auto& all_vars = op_info["vars"];
    if (all_vars.count("core_num") == 0) {
        ge::OpsGetCompileParamsErrReport(op_type.c_str(), "core_num");
        OP_LOGE(op_type.c_str(), "op [SparseApplyAdagradV2DTiling] : GetCompileParams, get %s error", "core_num");
        return false;
    }
    int32_t core_num = all_vars["core_num"].get<std::int32_t>();

    if (all_vars.count("cache_threshold_col") == 0) {
        ge::OpsGetCompileParamsErrReport(op_type.c_str(), "cache_threshold_col");
        OP_LOGE(op_type.c_str(), "op [SparseApplyAdagradV2DTiling] : GetCompileParams, get %s error",
                "cache_threshold_col");
        return false;
    }
    int32_t cache_threshold_col = all_vars["cache_threshold_col"].get<std::int32_t>();

    int32_t need_core_num = core_num;
    if (num_indices < core_num and each_row_data_num > cache_threshold_col) {
        need_core_num = num_indices;
        if (num_indices * 2 <= core_num and each_row_data_num >= 1024){
            need_core_num = core_num / num_indices * num_indices;
            if (need_core_num > core_num){
                need_core_num=core_num;
            }
            if(need_core_num <= 0) {
                need_core_num=1;
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

// register tiling interface of the SparseApplyAdagradV2D op
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyAdagradV2D, SparseApplyAdagradV2DTiling);

}  // namespace optiling
