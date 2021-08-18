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
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling.h"

namespace optiling {

bool NoneZeroTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                    OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "op NoneZeroTiling run start.");
  int32_t core_num = op_info["block_dim"].get<int32_t>();
  std::vector<int64_t> workspace = op_info["workspace"];
  OP_LOGD(op_type.c_str(), "original compile info is : %s.", op_info.dump().c_str());

  run_info.block_dim = core_num;
  run_info.workspaces = workspace;
  OP_LOGI(op_type.c_str(), "op NoneZeroTiling run success.");
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(NonZero, NoneZeroTiling);
} // namespace optiling
