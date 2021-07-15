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
#include "vector_tiling.h"
#include "op_log.h"

namespace optiling {

bool BiasTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   OpRunInfo& run_info) {
    OP_TILING_CHECK((op_info.count("boardcast_bias_shape") <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [boardcast_bias_shape]"),
                    return false);

    std::vector<int64_t> boardcast_bias_shape = op_info["boardcast_bias_shape"];

    OP_TILING_CHECK(op_paras.inputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty"),
                    return false);
    OP_TILING_CHECK(op_paras.inputs.size() <= 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs size need more than 1"), return false);
    OP_TILING_CHECK(op_paras.inputs[0].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[0].tensor cannot be empty"), return false);

    const std::vector<int64_t> input_shape_x = op_paras.inputs[0].tensor[0].shape;
    const std::vector<int64_t> input_shape_bias = op_paras.inputs[1].tensor[0].shape;

    OP_TILING_CHECK(input_shape_x.size() != boardcast_bias_shape.size(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input_shape_x dims size need same with boardcast_bias"),
                    return false);

    if (input_shape_x.size() == input_shape_bias.size()) {
        for (size_t i = 0; i < boardcast_bias_shape.size(); i++) {
            boardcast_bias_shape[i] = boardcast_bias_shape[i] == -1 ? input_shape_bias[i] : boardcast_bias_shape[i];
        }
    } else {
        for (size_t i = 0; i < boardcast_bias_shape.size(); i++) {
            boardcast_bias_shape[i] = boardcast_bias_shape[i] == -1 ? input_shape_x[i] : boardcast_bias_shape[i];
        }
    }

    TeOpParas op_paras_tmp = op_paras;
    op_paras_tmp.inputs[1].tensor[0].shape = boardcast_bias_shape;

    bool ret = EletwiseTiling(op_type, op_paras_tmp, op_info, run_info);
    return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Bias, BiasTiling);
}  // namespace optiling
