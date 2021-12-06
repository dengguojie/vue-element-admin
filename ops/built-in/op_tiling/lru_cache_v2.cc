/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file lru_cache_v2.cc
 * \brief
 */
#include <string>
#include <cmath>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/util.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
// Mode0 support all case
const int64_t TILING_MODE_0 = 0;
// use 8 cores all time
const int64_t BLOCKDIM = 8;
constexpr int32_t INPUT_DESC_SIZE = 5;
constexpr int32_t OUTPUT_DESC_SIZE = 6;

struct LRUCacheV2CompileParams {
  int64_t set_num;
  int64_t time_stamp_wsp_size;
  int64_t miss_index_bytes;
  std::string op_type;
};

struct LRUCacheV2TilingParams {
  int64_t tiling_key;
  // input dim num
  int64_t tiling_index_list_len;
};

void InitRunningParams(LRUCacheV2TilingParams& params) {
  params.tiling_key = TILING_MODE_0;
  params.tiling_index_list_len = 1;
}

static bool GetLRUCacheV2CompileParams(const nlohmann::json& compile_info, LRUCacheV2CompileParams& compile_params) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  if (allVars.count("set_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get set_num error");
    return false;
  }
  if (allVars.count("time_stamp_wsp_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get time_stamp_wsp_size error");
    return false;
  }
  if (allVars.count("miss_index_bytes") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get miss_index_bytes error");
    return false;
  }
  compile_params.set_num = allVars["set_num"].get<std::int64_t>();
  compile_params.time_stamp_wsp_size = allVars["time_stamp_wsp_size"].get<std::int64_t>();
  compile_params.miss_index_bytes = allVars["miss_index_bytes"].get<std::int64_t>();
  return true;
}

void SetRuningParams(const LRUCacheV2TilingParams& params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(params.tiling_key);
  run_info.AddTilingData(params.tiling_index_list_len);
}

void PrintTilingParams(const LRUCacheV2TilingParams& params, const std::string& op_type) {
  OP_LOGD(op_type, "tiling_key=%ld. ", params.tiling_key);
  OP_LOGD(op_type, "tiling_index_list_len=%ld.", params.tiling_index_list_len);
}

void _printTensorValue(const LRUCacheV2CompileParams& compile_params, const std::vector<int64_t>& in,
                       const std::string& name) {
  using namespace std;
  string vec_str;
  for (auto item : in) {
    vec_str += to_string(item);
    vec_str += ",";
  }
  OP_LOGD(compile_params.op_type, "Func[_printTensorValue] [%s]: [%s].", name.c_str(), vec_str.c_str());
}

static bool GetTilingParam(const std::vector<int64_t>& input_shape, LRUCacheV2TilingParams& tiling_params) {
  tiling_params.tiling_key = TILING_MODE_0;
  tiling_params.tiling_index_list_len = input_shape[0];
  return true;
}

bool LRUCacheV2Tiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_compile_info,
                      utils::OpRunInfo& run_info) {
  using namespace ge;
  OP_LOGI(op_type, "begin to run tiling.");
  PROFILING_TILING_INIT(op_type.c_str());
  if (op_compile_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info json error.");
    return false;
  }
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get operator_info error.");
    return false;
  }
  for (int i = 0; i < INPUT_DESC_SIZE; i++) {
    auto opdesc = operator_info->MutableInputDesc(i);
    if (opdesc == nullptr) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_%d shape error.", i);
      return false;
    }
  }
  for (int i = 0; i < OUTPUT_DESC_SIZE; i++) {
    auto opdesc = operator_info->MutableOutputDesc(i);
    if (opdesc == nullptr) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get output_%d shape error.", i);
      return false;
    }
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  // begin to get compile data
  LRUCacheV2CompileParams compile_params;
  compile_params.op_type = op_type;
  if (!GetLRUCacheV2CompileParams(op_compile_info, compile_params)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info from nlohmann json failed.");
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  auto opdesc = operator_info->MutableInputDesc(0);
  const std::vector<int64_t>& input_shape_const = opdesc->GetShape().GetDims();
  std::vector<int64_t> input_shape = input_shape_const;
  _printTensorValue(compile_params, input_shape, "index_list shape");
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  // end to get compile data
  LRUCacheV2TilingParams run_params;
  InitRunningParams(run_params);
  GetTilingParam(input_shape, run_params);
  SetRuningParams(run_params, run_info);
  PrintTilingParams(run_params, op_type);
  run_info.SetBlockDim(BLOCKDIM);
  // add miss_index workspace size
  auto miss_index_wsp_size = run_params.tiling_index_list_len * compile_params.set_num * 
	                     compile_params.miss_index_bytes;
  run_info.AddWorkspace(miss_index_wsp_size);
  // add time_stamp workspace size
  run_info.AddWorkspace(compile_params.time_stamp_wsp_size);
  OP_LOGD(op_type, "miss_index workspace size=%ld.", miss_index_wsp_size);
  OP_LOGD(op_type, "time_stamp workspace size=%ld.", compile_params.time_stamp_wsp_size);
  PROFILING_TILING_END();
  OP_LOGI(op_type, "end to run tiling, success!");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(LRUCacheV2, LRUCacheV2Tiling);
}  // namespace optiling
