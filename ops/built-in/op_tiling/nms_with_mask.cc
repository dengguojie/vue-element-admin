/*
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
const std::string NMS_OP_TYPE = "NMSWithMask";

bool GetNMSWithMaskCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info_json,
                                 int32_t& max_boxes_num) {
  using namespace nlohmann;
  if (op_compile_info_json == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null");
    return false;
  }

  const auto& all_vars = op_compile_info_json["vars"];
  // max boxes num
  if (all_vars.count("max_boxes_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "max_boxes_num is null");
    return false;
  }
  max_boxes_num = all_vars["max_boxes_num"].get<std::int32_t>();

  GELOGD("op [%s] : GetNMSWithMaskCompileParams, max_boxes_num[%d].", NMS_OP_TYPE.c_str(), max_boxes_num);
  return true;
}

bool NMSWithMaskTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info_json,
                       OpRunInfo& run_info) {
  GELOGI("op [%s] NMSWithMaskTiling running.", op_type.c_str());

  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  int32_t boxes_num = input_shape[0];
  int32_t max_boxes_num = 0;

  bool ret = GetNMSWithMaskCompileParams(op_type, op_compile_info_json, max_boxes_num);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetNMSWithMaskCompileParams failed.");
    return false;
  }
  GELOGI("op[%s] GetNMSWithMaskCompileParams success.", op_type.c_str());

  if (boxes_num > max_boxes_num) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input boxes number exceeds the maximum value that UB can store.");
    return false;
  }

  // write tiling params to run_info
  ByteBufferPut(run_info.tiling_data, boxes_num);
  GELOGD("op [%s]: input boxes number=%d.", op_type.c_str(), boxes_num);

  run_info.block_dim = 1;
  run_info.workspaces = {};

  GELOGI("op [%s] NMSWithMaskTiling run success.", op_type.c_str());
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(NMSWithMask, NMSWithMaskTiling);
}  // namespace optiling
