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
