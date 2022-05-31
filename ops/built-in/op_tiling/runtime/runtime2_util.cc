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

int64_t CeilAlign(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = (u_value + d_value - 1) / d_value * d_value;

  return res_value;
}

int64_t GetRemainder(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = u_value % d_value;

  return res_value;
}

static bool CalcReducMeanCof(const gert::Shape& input_shape, const std::vector<int32_t>& reduce_axis,
                             float& reduce_mean_cof) {
  const size_t dim_len = input_shape.GetDimNum();
  const size_t ori_reduce_axis_len = reduce_axis.size();
  // init reduce_mean_cof is 1.0
  reduce_mean_cof = 1.0;
  for (size_t i = 0; i < ori_reduce_axis_len; i++) {
    OP_TILING_CHECK(
        !ops::IsDimValid(dim_len, reduce_axis[i]),
        VECTOR_INNER_ERR_REPORT_TILIING("CalcReducMeanCof", "%s",
                                        ops::GenInvalidDimMsg("reduce_axis", i, dim_len, reduce_axis[i]).c_str()),
        return false);

    // convert reduce axis (like: -1 -> (dim_len - 1))
    int32_t single_reduce_axis = reduce_axis[i] < 0 ? reduce_axis[i] + dim_len : reduce_axis[i];

    int64_t reduce_dim = input_shape.GetDim(single_reduce_axis);
    OP_TILING_CHECK(reduce_dim == 0, OP_LOGI("CalcReducMeanCof", "the reduce dim is 0, will not use reduce_mean_cof"),
                    return true);
    reduce_mean_cof = reduce_mean_cof / reduce_dim;
  }
  OP_LOGD("CalcReducMeanCof", "CalcReducMeanCof cof is %1f", reduce_mean_cof);

  return true;
}

bool AddReducMeanCof(const gert::Shape& input_shape, const ge::DataType input_dtype,
                     const std::vector<int32_t>& reduce_axis, gert::TilingData* tiling_data) {
  float reduce_mean_cof = 1.0;
  bool calcu_flag = CalcReducMeanCof(input_shape, reduce_axis, reduce_mean_cof);
  OP_LOGD("AddReducMeanCof", "AddReducMeanCof dtype is %s", ops::ToString(input_dtype).c_str());
  switch (input_dtype) {
    case ge::DT_FLOAT:
      tiling_data->Append((float)reduce_mean_cof);
      return calcu_flag;
    case ge::DT_FLOAT16:
      tiling_data->Append((fe::fp16_t)reduce_mean_cof);
      tiling_data->Append((uint16_t)0);
      return calcu_flag;
    default:
      OP_LOGW("AddReducMeanCof", "AddReducMeanCof of dtype[%s] has not implement.", ops::ToString(input_dtype).c_str());
      return false;
  }
}
}  // namespace optiling
