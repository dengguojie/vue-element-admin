/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file vector_op_info.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_VECTOR_OP_INFO_H_
#define OPS_BUILT_IN_OP_TILING_VECTOR_OP_INFO_H_


#include <vector>

#include "graph/types.h"
#include "exe_graph/runtime/shape.h"

namespace optiling {
struct AutoTilingCompileInfo;
class OpInfoImpl;
class OpInfo {
public:
  explicit OpInfo(const std::vector<std::vector<int64_t>>& _op_input_shapes,
                  const ge::DataType& _op_in_type,
                  const std::vector<std::vector<int32_t>>& _op_axes);

  explicit OpInfo(const std::vector<std::vector<int64_t>>& _op_input_shapes,
                  const ge::DataType& _op_in_type);
  explicit OpInfo(const AutoTilingCompileInfo* _compile_info);
  ~OpInfo() = default;

  void SetInputShape(const std::vector<gert::Shape>* _op_input_ge_shapes);
  void SetInputShape(const std::vector<std::vector<int64_t>>* _op_input_shapes);
  void SetAxes(const std::vector<int64_t>* _op_axes);
  void SetInputType(ge::DataType* _op_in_type);

  // Compatible code, please do not use
  const std::vector<std::vector<int64_t>>& GetInputShape() const;
  // Compatible code, please do not use
  const std::vector<std::vector<int32_t>>& GetReduceAxes() const;
  // Compatible code, please do not use
  const ge::DataType& GetInType() const;

private:
  friend class OpInfoImplGetter;
  std::shared_ptr<OpInfoImpl> op_info_impl;
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_VECTOR_OP_INFO_H_
