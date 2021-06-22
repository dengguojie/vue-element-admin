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
#include "vector_tiling.h"
#include "error_log.h"

namespace optiling {

bool TileDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                 OpRunInfo& run_info) {
  V_OP_TILING_CHECK((op_info.find("tiling_info") != op_info.end()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [tiling_info]"),
                    return false);
  const std::vector<int64_t>& tiling_info = op_info["tiling_info"];

  V_OP_TILING_CHECK(!op_paras.inputs.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty"),
                    return false);
  V_OP_TILING_CHECK(!op_paras.inputs[0].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[0].tensor cannot be empty"),
                    return false);
  std::vector<int64_t> runtime_shape(op_paras.inputs[0].tensor[0].shape);

  // use assign init vector
  V_CHECK_GT(tiling_info.size(), 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "tiling_info index out of range"),
             return false);
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
  V_OP_TILING_CHECK(!op_paras_tmp.inputs.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras_tmp.inputs cannot be empty"),
                    return false);
  V_OP_TILING_CHECK(!op_paras_tmp.inputs[0].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras_tmp.inputs[0].tensor cannot be empty"),
                    return false);
  op_paras_tmp.inputs[0].tensor[0].shape = std::move(broadcast_input);

  // create other input multiples
  TeOpTensorArg multiples_input(op_paras_tmp.inputs[0]);
  V_OP_TILING_CHECK(!multiples_input.tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "multiples_input.tensor cannot be empty"),
		    return false);
  multiples_input.tensor[0].shape = std::move(broadcast_multiples);
  op_paras_tmp.inputs.push_back(multiples_input);
  bool ret = EletwiseTiling(op_type, const_cast<TeOpParas&>(op_paras_tmp), op_info, run_info);
  return ret;
}

// register tiling interface of the TileD op.
REGISTER_OP_TILING_FUNC_BUFFERED(TileD, TileDTiling);
}  // namespace optiling
