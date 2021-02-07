/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <iostream>
#include "error_log.h"
#include "eletwise.h"
#include "vector_tiling.h"
#include "op_log.h"

namespace optiling {

bool ScaleTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   OpRunInfo& run_info) {
    CHECK((op_info.count("_boardcast_scale_shape") > 0),
          "op [%s] : compile info not contain [_boardcast_scale_shape]", op_type.c_str());
    OP_LOGD("op [%s] Enter SCALETILING inputs size:%d", op_type.c_str(), op_paras.inputs.size());

    std::vector<int64_t> boardcast_scale_shape = op_info["_boardcast_scale_shape"];

    CHECK(!op_paras.inputs.empty(), "op [%s] : op_paras.inputs cannot be empty", op_type.c_str());
    CHECK(!op_paras.inputs[0].tensor.empty(), "op [%s] : op_paras.inputs[0].tensor cannot be empty", op_type.c_str());

    const std::vector<int64_t> input_shape_x = op_paras.inputs[0].tensor[0].shape;

    //print debug
    for (size_t i = 0; i < boardcast_scale_shape.size(); i++) {
        OP_LOGD("SCALETILING boardcast_scale_shape i=%d value=%d", i, boardcast_scale_shape[i]);
    }

    for (size_t i = 0; i < boardcast_scale_shape.size(); i++) {
        boardcast_scale_shape[i] = boardcast_scale_shape[i] == -1 ? input_shape_x[i] : boardcast_scale_shape[i];
    }

    TeOpParas op_paras_tmp = op_paras;
    if (op_paras_tmp.inputs.size() >= 2) {
        op_paras_tmp.inputs[1].tensor[0].shape = boardcast_scale_shape;
    }

    // remove varince  scale offset
    OP_LOGD("ScaleTiling before erase op_paras_tmp inputs size:%d", op_paras_tmp.inputs.size());
    size_t input_size = op_paras.inputs.size();
    for (size_t i = 0; i < input_size; i++) {
        op_paras_tmp.inputs.erase(op_paras_tmp.inputs.end() - 1);
        if (op_paras_tmp.inputs.size() <= 2) {
            break;
        }
    }
    OP_LOGD("ScaleTiling after erase op_paras_tmp inputs size:%d", op_paras_tmp.inputs.size());

    bool ret = EletwiseTiling(op_type, op_paras_tmp, op_info, run_info);
    return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Scale, ScaleTiling);
}  // namespace optiling
