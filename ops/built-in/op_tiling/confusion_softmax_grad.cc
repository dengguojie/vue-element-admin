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
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {

bool ConfusionSoftmaxGradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                                OpRunInfo& run_info) {

    std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> output_shape = op_paras.outputs[0].tensor[0].shape;

    int32_t n = input_shape[0];
    int32_t d = input_shape[1];
    int32_t c = input_shape[2];

    run_info.block_dim = 32;

    int32_t base_key = 10000000;

    int32_t block_tiling_axis = 0;
    int32_t ub_tiling_axis = 1;
    int32_t dim_len = 3;
    int32_t tiling_key = base_key + 1 + block_tiling_axis * dim_len + ub_tiling_axis;
    int32_t block_dim = 32;

    run_info.block_dim = block_dim;
    run_info.tiling_key = tiling_key;
    ByteBufferPut(run_info.tiling_data, n);
    ByteBufferPut(run_info.tiling_data, d);
    ByteBufferPut(run_info.tiling_data, c);

    int32_t ub_split_inner = 1;
    for (int32_t i = d; i > 0; i--) {
        if (d % i != 0) {
            continue;
        }
        if ((i * c) > 15360) {
            continue;
        }

        ub_split_inner = i;
        break;
    }

    ByteBufferPut(run_info.tiling_data, (int32_t)1);
    ByteBufferPut(run_info.tiling_data, (int32_t)ub_split_inner);
    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(ConfusionSoftmaxGrad, ConfusionSoftmaxGradTiling);

}
