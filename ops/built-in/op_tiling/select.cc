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

#include "error_log.h"
#include "eletwise.h"
#include "op_tiling.h"

namespace optiling {

bool SelectTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info) {
    CHECK((op_info.count("_boardcast_condition_fill") > 0),
          "op [%s] : compile info not contain [_boardcast_condition_fill]", op_type.c_str());

    const std::vector<int64_t> boardcast_condition_fill = op_info["_boardcast_condition_fill"];

    TeOpParas op_paras_tmp = op_paras;

    CHECK(!op_paras_tmp.inputs.empty(), "op [%s] : op_paras.inputs cannot be empty", op_type.c_str());
    CHECK(!op_paras_tmp.inputs[0].tensor.empty(),
          "op [%s] : op_paras.inputs[0].tensor cannot be empty", op_type.c_str());
    if (!boardcast_condition_fill.empty()) {
        op_paras_tmp.inputs[0].tensor[0].shape.insert(op_paras_tmp.inputs[0].tensor[0].shape.end(),
                                                      boardcast_condition_fill.begin(),
						      boardcast_condition_fill.end());
    }
    op_paras_tmp.inputs.erase(op_paras_tmp.inputs.end() - 1);
    Eletwise eletwise(op_type, op_paras_tmp, op_info);
    bool ret = eletwise.DoTiling();
    ret = ret && eletwise.WriteTilingData(run_info);
    return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Select, SelectTiling);
}  // namespace optiling
