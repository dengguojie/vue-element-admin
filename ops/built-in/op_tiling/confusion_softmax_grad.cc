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
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

namespace {
  constexpr int32_t BLOCK_DIM_NUM = 32;
}

namespace optiling {
bool ConfusionSoftmaxGradTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                                utils::OpRunInfo& run_info) {
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  std::vector<int64_t> input_shape = operator_info->MutableInputDesc(0)->MutableShape().GetDims();
  std::vector<int64_t> output_shape = operator_info->MutableOutputDesc(0)->MutableShape().GetDims();

  int32_t n = input_shape[0];
  int32_t d = input_shape[1];
  int32_t c = input_shape[2];

  run_info.SetBlockDim(BLOCK_DIM_NUM);

  int32_t base_key = 10000000;

  int32_t block_tiling_axis = 0;
  int32_t ub_tiling_axis = 1;
  int32_t dim_len = 3;
  int32_t tiling_key = base_key + 1 + block_tiling_axis * dim_len + ub_tiling_axis;
  int32_t block_dim = 32;

  run_info.SetBlockDim(block_dim);
  run_info.SetTilingKey(tiling_key);
  run_info.AddTilingData(n);
  run_info.AddTilingData(d);
  run_info.AddTilingData(c);

  int32_t ub_split_inner = 1;
  for (int32_t i = d; i > 0; i--) {
    if (d % i != 0) {
      continue;
    }
    if ((i * c) > 15360) {
      continue;
    }

    ub_split_inner = i;
    break;
  }

  run_info.AddTilingData((int32_t)1);
  run_info.AddTilingData((int32_t)ub_split_inner);
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED_V2(ConfusionSoftmaxGrad, ConfusionSoftmaxGradTiling);
}  // namespace optiling
