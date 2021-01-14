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
#include "eletwise.h"
#include "vector_tiling.h"
#include "error_log.h"
#include "op_log.h"

using namespace std;

namespace optiling {

bool BNInferenceDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   OpRunInfo& run_info) {
                       
    std::cout << "BNInferenceDTiling inputs size:" << op_paras.inputs.size() << std::endl;
    GELOGD("op [%s] Enter BNInferenceDTiling inputs size:%d", op_type.c_str(), op_paras.inputs.size());

    CHECK((op_info.count("_boardcast_mean_shape") > 0),
          "op [%s] : compile info not contain [_boardcast_mean_shape]", op_type.c_str());

    std::vector<int64_t> boardcast_mean_shape = op_info["_boardcast_mean_shape"];

    CHECK(!op_paras.inputs.empty(), "op [%s] : op_paras.inputs cannot be empty", op_type.c_str());
    CHECK(!op_paras.inputs[0].tensor.empty(), "op [%s] : op_paras.inputs[0].tensor cannot be empty", op_type.c_str());

    const std::vector<int64_t> input_shape_x = op_paras.inputs[0].tensor[0].shape;

    // modify shape mean & varince
    for (size_t i = 0; i < boardcast_mean_shape.size(); i++) {
        boardcast_mean_shape[i] = boardcast_mean_shape[i] == -1 ? input_shape_x[i] : boardcast_mean_shape[i];
        std::cout << "BNInferenceDTiling inputs size:" << i  << "=>" << boardcast_mean_shape[i] << std::endl;
    }

    TeOpParas op_paras_tmp = op_paras;

    // mean
    op_paras_tmp.inputs[1].tensor[0].shape = boardcast_mean_shape;

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

    Eletwise eletwise(op_type, op_paras_tmp, op_info);
    bool ret = eletwise.DoTiling();
    std::cout << "BNInferenceDTiling ret:" << ret << std::endl;
    ret = ret && eletwise.WriteTilingData(run_info);

    return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(BNInferenceD, BNInferenceDTiling);
}  // namespace optiling
