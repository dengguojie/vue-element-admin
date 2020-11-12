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
#include "register/op_tiling.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "eletwise.h"
#include "vector_tiling.h"

namespace optiling {

bool TileDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                 OpRunInfo& run_info) {
  CHECK((op_info.count("_compile_shape") > 0),
        "op [%s] : compile info not contain [_compile_shape]", op_type.c_str());
  CHECK((op_info.count("_origin_multiples") > 0),
        "op [%s] : compile info not contain [_origin_multiples]", op_type.c_str());

  std::vector<int64_t> compile_shape = op_info["_compile_shape"].get<std::vector<int64_t>>();
  std::vector<int64_t> origin_multiples = op_info["_origin_multiples"].get<std::vector<int64_t>>();

  CHECK(!op_paras.inputs.empty(), "op [%s] : op_paras.inputs cannot be empty", op_type.c_str());
  CHECK(!op_paras.inputs[0].tensor.empty(), "op [%s] : op_paras.inputs[0].tensor cannot be empty", op_type.c_str());
  std::vector<int64_t> runtime_shape = op_paras.inputs[0].tensor[0].shape;

  std::vector<int64_t> broadcast_input = {};
  std::vector<int64_t> broadcast_multiples = {};

  // align shape for multiples and input shapes
  uint64_t len_diff = origin_multiples.size() - compile_shape.size();
  compile_shape.insert(compile_shape.begin(), len_diff, 1);
  runtime_shape.insert(runtime_shape.begin(), len_diff, 1);

  for (uint64_t i = 0; i < origin_multiples.size(); i++) {
    if (compile_shape[i] != 1 && origin_multiples[i] != 1) {
      broadcast_input.push_back(1);
      broadcast_input.push_back(runtime_shape[i]);
      broadcast_multiples.push_back(origin_multiples[i]);
      broadcast_multiples.push_back(runtime_shape[i]);
    } else {
      broadcast_input.push_back(runtime_shape[i]);
      broadcast_multiples.push_back(origin_multiples[i]);
    }
  }

  TeOpParas op_paras_tmp = op_paras;
  // update new shape
  CHECK(!op_paras_tmp.inputs.empty(),
        "op [%s] : op_paras_tmp.inputs cannot be empty", op_type.c_str());
  CHECK(!op_paras_tmp.inputs[0].tensor.empty(),
        "op [%s] : op_paras_tmp.inputs[0].tensor cannot be empty", op_type.c_str());
  op_paras_tmp.inputs[0].tensor[0].shape = std::move(broadcast_input);

  // create other input multiples
  TeOpTensorArg multiples_input;
  multiples_input.arg_type = op_paras_tmp.inputs[0].arg_type;
  multiples_input.tensor = op_paras_tmp.inputs[0].tensor;
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
REGISTER_OP_TILING_FUNC(TileD, TileDTiling);
}  // namespace optiling
