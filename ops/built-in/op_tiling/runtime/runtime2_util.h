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
 * \file runtime2_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_

#include "context_util.h"
#include "register/op_impl_registry.h"
#include "runtime/continuous_vector.h"
#include "runtime/storage_shape.h"
#include "runtime/tiling_context.h"
#include "runtime/tiling_parse_context.h"
#include <nlohmann/json.hpp>

namespace optiling {
template <typename T>
T* MutableCompileInfo(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}

std::unique_ptr<nlohmann::json> GetJsonObj(gert::TilingParseContext* context);

bool AddWorkspace(gert::TilingContext* context, const size_t workspace);
int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end);
}  // namespace optiling

#endif  // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_
