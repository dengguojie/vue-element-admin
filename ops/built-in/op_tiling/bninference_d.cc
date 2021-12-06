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

#include <iostream>
#include "vector_tiling.h"
#include "error_log.h"
#include "op_log.h"

using namespace std;

namespace optiling {
bool BNInferenceDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                        OpRunInfo& run_info) {
    GELOGD("op [%s] Enter BNInferenceDTiling inputs size:%d", op_type.c_str(), op_paras.inputs.size());

    OP_TILING_CHECK((op_info.count("broadcast_mean_shape") < 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [broadcast_mean_shape]"),
                    return false);

    std::vector<int64_t> broadcast_mean_shape = op_info["broadcast_mean_shape"];

    if (op_paras.inputs.empty() || op_paras.inputs.size() < 3 ||
        op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Length of inputs is less than 3 or some inputs are empty.");
        return false;
    }

    const std::vector<int64_t> input_shape_x = op_paras.inputs[0].tensor[0].shape;

    OP_TILING_CHECK((broadcast_mean_shape.size() > input_shape_x.size()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "shape of mean is lager than shape of x."), return false);
    // modify shape mean & varince
    for (size_t i = 0; i < broadcast_mean_shape.size(); i++) {
        broadcast_mean_shape[i] = broadcast_mean_shape[i] == -1 ? input_shape_x[i] : broadcast_mean_shape[i];
    }

    TeOpParas op_paras_tmp = op_paras;

    // mean
    op_paras_tmp.inputs[1].tensor[0].shape = broadcast_mean_shape;

    // remove varince  scale offset
    GELOGD("BNInferenceDTiling before erase op_paras_tmp inputs size:%d", op_paras_tmp.inputs.size());
    size_t input_size = op_paras.inputs.size();
    for (size_t i = 0; i < input_size; i++) {
        op_paras_tmp.inputs.erase(op_paras_tmp.inputs.end() - 1);
        if (op_paras_tmp.inputs.size() <= 2) {
            break;
        }
    }
    GELOGD("BNInferenceDTiling after erase op_paras_tmp inputs size:%d", op_paras_tmp.inputs.size());

    bool ret = EletwiseTiling(op_type, op_paras_tmp, op_info, run_info);
    return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(BNInferenceD, BNInferenceDTiling);
}  // namespace optiling
