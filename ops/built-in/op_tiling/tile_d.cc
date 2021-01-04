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
 * \file tile_d.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include <sstream>
#include <cctype>
#include "op_tiling.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "eletwise.h"
#include "vector_tiling.h"

namespace optiling {

bool TileDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                 OpRunInfo& run_info) {
  CHECK((op_info.find("_tiling_info") != op_info.end()),
        "op [%s] : compile info not contain [_tiling_info]", op_type.c_str());
  const std::vector<int64_t>& tiling_info = op_info["_tiling_info"];

  CHECK(!op_paras.inputs.empty(), "op [%s] : op_paras.inputs cannot be empty", op_type.c_str());
  CHECK(!op_paras.inputs[0].tensor.empty(), "op [%s] : op_paras.inputs[0].tensor cannot be empty", op_type.c_str());
  std::vector<int64_t> runtime_shape(op_paras.inputs[0].tensor[0].shape);

  // use assign init vector
  size_t shape_size = (tiling_info.size() - tiling_info[0] - 1) / 2;
  std::vector<int64_t> broadcast_input(shape_size);
  std::vector<int64_t> broadcast_multiples(shape_size);
  broadcast_input.assign(tiling_info.begin() + tiling_info[0] + 1, tiling_info.end() - shape_size);
  broadcast_multiples.assign(tiling_info.end() - shape_size, tiling_info.end());
  int64_t count = 1;
  for (size_t i = 0; i < shape_size; i++) {
    if (broadcast_input[i] == -1) {
      broadcast_input[i] = broadcast_multiples[i] = runtime_shape[tiling_info[count]];
      count++;
    }
    if (tiling_info[0] + 1 == count) {
      break;
    }
  }

  TeOpParas op_paras_tmp = std::move(op_paras);
  // update new shape
  CHECK(!op_paras_tmp.inputs.empty(),
        "op [%s] : op_paras_tmp.inputs cannot be empty", op_type.c_str());
  CHECK(!op_paras_tmp.inputs[0].tensor.empty(),
        "op [%s] : op_paras_tmp.inputs[0].tensor cannot be empty", op_type.c_str());
  op_paras_tmp.inputs[0].tensor[0].shape = std::move(broadcast_input);

  // create other input multiples
  TeOpTensorArg multiples_input(op_paras_tmp.inputs[0]);
  CHECK(!multiples_input.tensor.empty(),
        "op [%s] : multiples_input.tensor cannot be empty", op_type.c_str());
  multiples_input.tensor[0].shape = std::move(broadcast_multiples);
  op_paras_tmp.inputs.push_back(multiples_input);
  Eletwise eletwise(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info);
  bool ret = eletwise.DoTiling();
  ret = ret && eletwise.WriteTilingData(run_info);
  return ret;
}

// register tiling interface of the TileD op.
REGISTER_OP_TILING_FUNC_BUFFERED(TileD, TileDTiling);
}  // namespace optiling
