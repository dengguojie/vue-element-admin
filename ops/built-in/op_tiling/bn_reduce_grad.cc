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
 * \file bn_reduce_grad.cpp
 * \brief
 */
#include "eletwise.h"
#include <algorithm>
#include "vector_tiling.h"
#include "error_log.h"
#include "../fusion_pass/common/fp16_t.hpp"

namespace optiling {
bool BnTrainingReduceGradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                                OpRunInfo& run_info) {

    bool ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
    if (!ret) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "bn_training_reduce_grad tiling failed.");
        return false;
    }

    const std::vector<int64_t>& input_x_shapes = op_paras.inputs[0].tensor[0].shape;
    OP_LOGD(op_type, "get shape (%lld,%lld,%lld,%lld,%lld)",
            input_x_shapes[0], input_x_shapes[1], input_x_shapes[2], input_x_shapes[3], input_x_shapes[4]);

    float reduce_mean_cof = 1.0;
    int64_t num = input_x_shapes[0] * input_x_shapes[2] * input_x_shapes[3];
    if (num == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "bn_training_reduce_grad invalid dim value 0. (%ld,%ld,%ld,%ld,%ld)",
            input_x_shapes[0], input_x_shapes[1], input_x_shapes[2], input_x_shapes[3], input_x_shapes[4]);
        return false;
    }
    reduce_mean_cof = (float)(reduce_mean_cof / num);

    if (op_info.count("reduce_mean_cof_dtype") > 0) {
        const std::string reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();

        ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
        ByteBufferPut(run_info.tiling_data, (float)(-1.0 * reduce_mean_cof));

        OP_LOGD(op_type, "bn_training_reduce_grad write tilingdata num_rec= %f", reduce_mean_cof);
    }

    return true;
}

// register tiling interface of the bn_training_reduce_grad op.
REGISTER_OP_TILING_FUNC_BUFFERED(BNTrainingReduceGrad, BnTrainingReduceGradTiling);
}  // namespace optiling
