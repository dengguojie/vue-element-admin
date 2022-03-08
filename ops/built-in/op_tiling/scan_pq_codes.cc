/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#include <nlohmann/json.hpp>
#include <string>
#include <algorithm>
#include <vector>
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace {
  constexpr int32_t INPUT_SIZE = 2;
  constexpr int32_t SPLIT_COUNT_NUM = 2;
}

namespace optiling {
struct ScanPQCodesTilingParams {
  int64_t bucketNumTotal;
  int64_t bucketStartBase;
  int64_t bucketNumLow;
  int64_t bucketNumHigh;
  int64_t highCoreNum;
};

void ScanPQCodesWriteTilingParams(const ScanPQCodesTilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.bucketNumTotal);
  ByteBufferPut(run_info.tiling_data, params.bucketStartBase);
  ByteBufferPut(run_info.tiling_data, params.bucketNumLow);
  ByteBufferPut(run_info.tiling_data, params.bucketNumHigh);
  ByteBufferPut(run_info.tiling_data, params.highCoreNum);
}

void ScanPQCodesPrintTilingParams(const std::string& opType, const ScanPQCodesTilingParams& params) {
  OP_LOGD("ScanPQCodes", "op [ScanPQCodes] : params.bucketNumTotal=%d", params.bucketNumTotal);
  OP_LOGD("ScanPQCodes", "op [ScanPQCodes] : params.bucketStartBase=%d", params.bucketStartBase);
  OP_LOGD("ScanPQCodes", "op [ScanPQCodes] : params.bucketNumLow=%d", params.bucketNumLow);
  OP_LOGD("ScanPQCodes", "op [ScanPQCodes] : params.bucketNumHigh=%d", params.bucketNumHigh);
  OP_LOGD("ScanPQCodes", "op [ScanPQCodes] : params.highCoreNum=%d", params.highCoreNum);
}

int64_t ScanPQCodesCeilDiv(int64_t dividend, int64_t divisor) {
  return (dividend + divisor - 1) / divisor;
}

const int64_t TOTAL_CORE_NUM = 15;
const int64_t AI_CORE_NUM = 8;
const int64_t VECTOR_CORE_NUM = 7;
bool ScanPQCodesTiling(const std::string& opType, const TeOpParas& op_paras, const nlohmann::json& op_compile_info_json,
                       OpRunInfo& run_info) {
  OP_LOGI("========================ScanPQCodesTiling running====================");
  if (op_paras.inputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.inputs cannot be empty");
    return false;
  }
  if (op_paras.inputs.size() < INPUT_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.inputs.size() < 2.");
    return false;
  }
  if (op_paras.inputs[1].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "bucket_list tensor shape cannot be empty");
    return false;
  }
  std::vector<int64_t> bucketShape = op_paras.inputs[1].tensor[0].shape;
  const auto& allVars = op_compile_info_json["vars"];
  if (allVars.count("core_nums") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get core_nums fail.");
    return false;
  }
  int64_t coreNums = allVars["core_nums"].get<std::int64_t>();
  if (coreNums == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, coreNums is 0.");
    return false;
  }
  int64_t splitCount = allVars["split_count"].get<std::int64_t>();
  if (splitCount < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, split_count is 0.");
    return false;
  }
  int64_t splitIndex = allVars["split_index"].get<std::int64_t>();
  int64_t bucketNumTotal = 0;
  int64_t bucketStartBase = 0;
  int64_t bucketNumLow = 0;
  int64_t bucketNumHigh = 0;
  int64_t highCoreNum = 0;
  int64_t aiMaxBucketNums = ScanPQCodesCeilDiv(bucketShape[0], TOTAL_CORE_NUM) * AI_CORE_NUM;
  if (splitCount == SPLIT_COUNT_NUM) {
    if (splitIndex == 0) {
      bucketNumTotal = (aiMaxBucketNums < bucketShape[0]) ? aiMaxBucketNums : bucketShape[0];
      bucketStartBase = 0;
    } else {
      bucketStartBase = (aiMaxBucketNums < bucketShape[0]) ? aiMaxBucketNums : bucketShape[0];
      bucketNumTotal = (bucketShape[0] >= bucketStartBase) ? (bucketShape[0] - bucketStartBase) : 0;
    }
  } else {
    bucketNumTotal = bucketShape[0];
    bucketStartBase = 0;
  }
  coreNums = (splitIndex == 0) ? AI_CORE_NUM : VECTOR_CORE_NUM;
  bucketNumLow = bucketNumTotal / coreNums;
  bucketNumHigh = ScanPQCodesCeilDiv(bucketNumTotal, coreNums);
  highCoreNum = bucketNumTotal % coreNums;

  ScanPQCodesTilingParams params{bucketNumTotal, bucketStartBase, bucketNumLow, bucketNumHigh, highCoreNum};
  run_info.block_dim = coreNums;
  ScanPQCodesWriteTilingParams(params, run_info);
  ScanPQCodesPrintTilingParams(opType, params);
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(ScanPQCodes, ScanPQCodesTiling);
}
