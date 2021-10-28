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
 * \file bitwise.cc
 * \brief
 */
#include <cctype>
#include <iostream>
#include <sstream>
#include "eletwise.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling.h"

namespace optiling {
bool BitwiseTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   OpRunInfo& run_info) {
    std::vector<int64_t> x1_runtime_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> x2_runtime_shape = op_paras.inputs[1].tensor[0].shape;

    std::string shape_dtype = op_paras.inputs[0].tensor[0].dtype;

    std::vector<int64_t> x1_broadcast_shape(x1_runtime_shape.size());
    std::vector<int64_t> x2_broadcast_shape(x2_runtime_shape.size());

    std::copy(x1_runtime_shape.begin(), x1_runtime_shape.end(), x1_broadcast_shape.begin());
    std::copy(x2_runtime_shape.begin(), x2_runtime_shape.end(), x2_broadcast_shape.begin());

    int64_t len_diff = x1_runtime_shape.size() - x2_runtime_shape.size();
    if (len_diff > 0) {
        x2_broadcast_shape.insert(x2_broadcast_shape.begin(), len_diff, 1);
    }
    else if (len_diff < 0) {
        x1_broadcast_shape.insert(x1_broadcast_shape.begin(), std::abs(len_diff), 1);
    }

    //make the shape as same as dsl, the shape should add a dim (2,) when dtype from int32 to int16
    if(shape_dtype == "int32") {
        x1_broadcast_shape.push_back(2);
        x2_broadcast_shape.push_back(2);
    }

    TeOpParas op_paras_tmp = op_paras;
    //update new shape
    op_paras_tmp.inputs[0].tensor[0].shape = std::move(x1_broadcast_shape);
    op_paras_tmp.inputs[0].tensor[0].dtype = "int16";
    op_paras_tmp.inputs[1].tensor[0].shape = std::move(x2_broadcast_shape);
    op_paras_tmp.inputs[1].tensor[0].dtype = "int16";

    bool ret = EletwiseTiling(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info, run_info);
    return ret;
}

// register tiling interface of the BitwiseAnd op.
REGISTER_OP_TILING_FUNC_BUFFERED(BitwiseAnd, BitwiseTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(BitwiseOr, BitwiseTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(BitwiseXor, BitwiseTiling);
}  // namespace optiling
