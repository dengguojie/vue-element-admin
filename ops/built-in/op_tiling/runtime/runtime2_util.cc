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
 * \file runtime2_util.cc
 * \brief
 */
#include "../error_log.h"
#include "runtime2_util.h"

namespace optiling {
std::unique_ptr<nlohmann::json> GetJsonObj(gert::TilingParseContext* context) {
  auto json_str = context->GetCompiledJson();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, json_str, nullptr);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  return parsed_object_cinfo;
}

bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
  *workspace_size = workspace;
  return true;
}

int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end) {
  int64_t size = 1;
  for (size_t i = begin; i < end; i++) {
    size *= shape[i];
  }
  return size;
}
}  // namespace optiling
