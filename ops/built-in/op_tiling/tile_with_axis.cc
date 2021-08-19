/*
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <iostream>
#include "vector_tiling.h"
#include "error_log.h"
#include "op_log.h"

using namespace std;

namespace optiling {

bool TileWithAxisTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                        OpRunInfo& run_info) {
    std::cout << "Enter TileWithAxisTiling" << std::endl;

    int64_t ori_axis_value = op_info["ori_axis_value"];
    int64_t axis = op_info["attr_axis"];
    int64_t tiles = op_info["attr_tiles"];
    std::cout << "TileWithAxisTiling ori_axis_value:" << ori_axis_value << std::endl;
    std::cout << "TileWithAxisTiling axis:" << axis << std::endl;
    std::cout << "TileWithAxisTiling tiles:" << tiles << std::endl;

    std::vector<int64_t> shape_x = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> shape_y(shape_x);

    if (ori_axis_value != 1) {
        shape_x.insert(shape_x.begin() + axis, 1);
        shape_y.insert(shape_y.begin() + axis, tiles);
    } else {
        shape_y[axis] = tiles;
    }

    // update output shape
    TeOpParas op_paras_tmp = op_paras;
    op_paras_tmp.inputs[0].tensor[0].shape = std::move(shape_x);
    TeOpTensorArg another_input(op_paras_tmp.inputs[0]);
    another_input.tensor[0].shape = std::move(shape_y);
    op_paras_tmp.inputs.push_back(another_input);

    bool ret = EletwiseTiling(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info, run_info);
    std::cout << "Leave TileWithAxisTiling" << std::endl;
    return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(TileWithAxis, TileWithAxisTiling);
}  // namespace optiling
