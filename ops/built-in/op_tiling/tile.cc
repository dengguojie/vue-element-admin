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
 * \file tile.cc
 * \brief
 */
#include <sstream>
#include <cctype>
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling.h"
#include <iostream>

namespace optiling {
bool TileTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                OpRunInfo& run_info) {
    OP_TILING_CHECK((op_info.count("compile_shape") <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [compile_shape]"), return false);
    OP_TILING_CHECK(op_paras.inputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty"),
                    return false);
    OP_TILING_CHECK(op_paras.inputs[0].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[0].tensor cannot be empty"), return false);

    std::vector<int64_t> x_runtime_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> compile_shape = op_info["compile_shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> multiples_value;

    OP_TILING_CHECK(op_paras.inputs.size() < 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs's size should be >= 2"), return false);
    OP_TILING_CHECK(op_paras.inputs[1].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[1].tensor cannot be empty"),
                    return false);
    std::string multiples_dtype = op_paras.inputs[1].tensor[0].dtype;
    auto pointer = std::get<0>(op_paras.const_inputs.at("multiples"));
    auto size = std::get<1>(op_paras.const_inputs.at("multiples"));

    uint32_t count =
      (multiples_dtype == "int64") ? size / sizeof(int64_t) : (multiples_dtype == "int32") ? size / sizeof(int32_t) : 0;
    OP_TILING_CHECK(!count,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input multiples shape cannot be empty"),
                    return false);

    if (multiples_dtype == "int64") {
        auto* data = (int64_t*)pointer;
        while (count--) {
            multiples_value.push_back(*data++);
        }
    }
    if (multiples_dtype == "int32") {
        auto* data = (int32_t*)pointer;
        while (count--) {
            multiples_value.push_back(*data++);
        }
    }


    std::vector<int64_t> broadcast_input = {};
    std::vector<int64_t> broadcast_multiples = {};
    std::vector<int64_t> output_shape = {};

    // align shape for multiples and input shapes
    uint64_t len_diff = multiples_value.size() - compile_shape.size();
    OP_TILING_CHECK(
        (len_diff < 0),
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "length of multiples should not be less than input_x's dimension"),
        return false);
    x_runtime_shape.insert(x_runtime_shape.begin(), len_diff, 1);
    compile_shape.insert(compile_shape.begin(), len_diff, 1);

    std::stringstream debugStr;
    debugStr << "multiples const value: ";
    for (size_t i = 0; i < multiples_value.size(); i++) {
        debugStr << multiples_value[i] << ",";
    }

    debugStr << " x_runtime_shape :";
    for (size_t i = 0; i < x_runtime_shape.size(); i++) {
        debugStr << x_runtime_shape[i] << ",";
    }

    debugStr << " compile_shape value :";
    for (size_t i = 0; i < compile_shape.size(); i++) {
        debugStr << compile_shape[i] << ",";
    }
    GELOGD("print input shapes: %s", debugStr.str().c_str());


    for (uint64_t i = 0; i < multiples_value.size(); i++) {
        if (compile_shape[i] != 1) {
            broadcast_input.push_back(1);
            broadcast_input.push_back(x_runtime_shape[i]);
            broadcast_multiples.push_back(multiples_value[i]);
            broadcast_multiples.push_back(x_runtime_shape[i]);

            output_shape.push_back(x_runtime_shape[i] * multiples_value[i]);
        } else {
            broadcast_input.push_back(1);
            broadcast_multiples.push_back(multiples_value[i]);
            output_shape.push_back(multiples_value[i]);
        }
    }

    std::stringstream ss;
    ss << "broadcast_input: ";
    for (size_t i = 0; i < broadcast_input.size(); i++) {
        ss << broadcast_input[i] << ",";
    }

    ss << " broadcast_multiples :";
    for (size_t i = 0; i < broadcast_multiples.size(); i++) {
        ss << broadcast_multiples[i] << ",";
    }

    ss << " output_shape value :";
    for (size_t i = 0; i < output_shape.size(); i++) {
        ss << output_shape[i] << ",";
    }
    GELOGD("get broadcast_input shape: %s", ss.str().c_str());

    TeOpParas op_paras_tmp = op_paras;
    // update new shape
    op_paras_tmp.inputs[0].tensor[0].shape = std::move(broadcast_multiples);
    op_paras_tmp.inputs[1].tensor[0].shape = std::move(broadcast_input);
    op_paras_tmp.outputs[0].tensor[0].shape = std::move(output_shape);

    bool ret = EletwiseTiling(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info, run_info);
    return ret;
}

// register tiling interface of the Tile op.
REGISTER_OP_TILING_FUNC_BUFFERED(Tile, TileTiling);
}  // namespace optiling
