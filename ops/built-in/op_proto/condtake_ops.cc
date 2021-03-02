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
 * \file condtake_ops.cpp
 * \brief
 */
#include "inc/condtake_ops.h"
#include <unordered_set>
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
IMPLEMT_INFERFUNC(CondTake, CondTakeInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> range;
  //out_data
  auto output_data_desc = op_desc->MutableOutputDesc(0);
  output_data_desc->SetShape(x_desc->GetShape());
  output_data_desc->SetDataType(DT_FLOAT);
  //out_index
  auto output_index_desc = op_desc->MutableOutputDesc(1);
  output_index_desc->SetShape(x_desc->GetShape());
  output_index_desc->SetDataType(DT_INT32);
  if(x_desc->GetShapeRange(range) == GRAPH_SUCCESS){
    output_data_desc->SetShapeRange(range);
    output_index_desc->SetShapeRange(range);    
  }
  //valid_num
 auto output_num_desc = op_desc->MutableOutputDesc(2);
  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_range.push_back(std::pair<int64_t, int64_t>{1, 1} );
  output_num_desc->SetShape(ge::GeShape({1}));
  output_num_desc->SetShapeRange(y_range);
  output_num_desc->SetDataType(DT_INT32);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(CondTake, CondTakeInfer);
}  // namespace ge
