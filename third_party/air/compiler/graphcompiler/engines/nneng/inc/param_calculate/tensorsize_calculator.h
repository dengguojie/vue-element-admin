/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_TENSORSIZE_CALCULATOR_H
#define FUSION_ENGINE_INC_COMMON_TENSORSIZE_CALCULATOR_H

#include "graph_optimizer/graph_optimize_register_error_codes.h"

#include <map>
#include <string>
#include "graph/compute_graph.h"
#include "graph/op_desc.h"

namespace fe {
class TensorSizeCalculator {
 public:
  /**
   * Calculate the tensor size of input and output of each opdesc
   * @param op_desc opdesc object
   * @param op_impl_type op impl type
   * @return status SUCCESS or FAILED
   */
  static Status CalculateOpTensorSize(ge::OpDesc &op_desc);

 private:
  static Status CalcSingleTensorSize(const ge::OpDesc &op_desc, const ge::GeTensorDescPtr &tensor_desc_ptr,
      const string &direction, size_t i, bool output_real_calc_flag, int64_t &tensor_size);

  static Status CalcInputOpTensorSize(const ge::OpDesc &op_desc, const int32_t &output_real_calc_flag);

  static Status CalcOutputOpTensorSize(const ge::OpDesc &op_desc, const int32_t &output_real_calc_flag);
};
}  // namespace fe

#endif  // FUSION_ENGINE_INC_COMMON_TENSORSIZE_CALCULATOR_H
