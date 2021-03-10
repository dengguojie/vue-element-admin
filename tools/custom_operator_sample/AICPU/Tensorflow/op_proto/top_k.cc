/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file top_k.cc
 * \brief
 */
#include "top_k.h"

#include <cmath>
#include <string>
#include <vector>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "strided_slice_infer_shape.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"

namespace ge {
// ----------------TopK Op-------------------
IMPLEMT_VERIFIER(TopK, TopKVerify) { return GRAPH_SUCCESS; }

IMPLEMT_COMMON_INFERFUNC(TopKInferShape) {
  const vector<string> depend_names = {"k"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  Tensor k_tensor;
  bool unkonwn_dim_flag{false};
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constdata failed, unknown dim.");
    unkonwn_dim_flag = true;
  }

  // Tensor::GetData() return a uint8 ptr. However the definition of k is int32.
  // So here use int32* ptr to get the k value
  int64_t k = UNKNOWN_DIM;
  if (!unkonwn_dim_flag && k_tensor.GetData() != nullptr) {
    DataType dtype = op.GetInputDesc("k").GetDataType();
    if (dtype == DT_INT32) {
      k = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(k_tensor.GetData())));
    } else if (dtype == DT_INT64) {
      k = *(reinterpret_cast<int64_t*>(k_tensor.GetData()));
    } else {
      OP_LOGE(op.GetName().c_str(), "The type of k Error!");
      return GRAPH_FAILED;
    }
  }

  if (TopKInferCommon(op, k) == false) {
    OP_LOGE(op.GetName().c_str(), "TopKInferCommon Failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TopK, TopKInferShape);
VERIFY_FUNC_REG(TopK, TopKVerify);
// ----------------TopK Op End-------------------
}  // namespace ge
