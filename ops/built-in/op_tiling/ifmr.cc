/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file ifmr.cc
 * \brief
 */
#include <cmath>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"


namespace {
  constexpr int32_t BLOCK_DIM_SIZE = 30;
}

namespace optiling {
using namespace ge;

bool IFMRTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                OpRunInfo& run_info) {
    OP_LOGI(op_type.c_str(), "IFMRTiling running.");
    OP_TILING_CHECK(op_paras.inputs.empty(), OP_LOGE(op_type.c_str(), "op_paras.inputs cannot be empty."),
		    return false);

    const std::vector<int64_t> src_shape = op_paras.inputs[0].tensor[0].shape;
    const std::vector<int64_t> dst_shape = op_paras.outputs[0].tensor[0].shape;
    const int barrier_workspace_size = 960;
    const int loss_workspace_size = 491520;
    const int64_t max_size = 2147483647;

    int64_t data_size;

    data_size = std::accumulate(src_shape.begin(), src_shape.end(), 1, std::multiplies<int64_t>());
    OP_LOGI("ifmr data num is %d", data_size);

    if (data_size > max_size) {
        OP_LOGE("ifmr data num overflow int32");
        return false;
    }

    int32_t data_num = static_cast<int32_t>(data_size);
    ByteBufferPut(run_info.tiling_data, data_num);

    run_info.block_dim = BLOCK_DIM_SIZE;
    std::vector<int64_t> workspace;
    workspace.push_back(barrier_workspace_size);
    workspace.push_back(loss_workspace_size);
    run_info.workspaces = workspace;
    OP_LOGI(op_type.c_str(), "IFMRTiling run success.");
    return true;
}

// register tiling interface of the IFMR op.
REGISTER_OP_TILING_FUNC_BUFFERED(IFMR, IFMRTiling);
}  // namespace optiling
