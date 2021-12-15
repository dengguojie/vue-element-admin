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
 * \file gen_adc.cc
 * \brief tiling function of op
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace {
  constexpr int32_t INPUT_LENGTH = 4;
}

namespace optiling {
const int64_t BLOCK_SIZE = 32;  // one block size is 32 Bytes

const map<string, int64_t> BUCKET_LIST_DTYPES = {
    {"int32", sizeof(int32_t)}, {"int8", sizeof(int8_t)}, {"int16", sizeof(int16_t)}, {"int64", sizeof(int64_t)}};

struct GenADCParams {
  int64_t rowNumEachCore;
  int64_t remainingRow;
  int64_t bucketListBurstLen;
  int64_t remainBucketListBurstLen;
  int64_t coreUsedNum;
  int64_t dimNs;
};

bool CheckGenADCParams(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opInfo) {
  OP_LOGD(opType.c_str(), "op[GenADCTiling] CheckGenADCParams begin.");

  if (opInfo == nullptr) {
    OP_LOGE(opType.c_str(), "op[GenADCTiling] opInfo json error.");
    return false;
  }
  if (opParas.inputs.size() < INPUT_LENGTH) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "inputs", "The length of inputs is less than 4");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] The length of inputs is less than 4.");
    return false;
  }
  if (opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() || opParas.inputs[2].tensor.empty() ||
      opParas.inputs[3].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "inputs", "Some of inputs is empty");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Some of inputs is empty.");
    return false;
  }
  if (opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "adc_tables",
                                   "The length of outputs is less than 1 or the outputs is empty");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] The length of outputs is less than 1 or the outputs is empty.");
    return false;
  }

  std::vector<int64_t> bucketListShape = opParas.inputs[3].tensor[0].shape;
  if (bucketListShape.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "bucket list", "The shape of bucket list is empty");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] The shape of bucket list is empty.");
    return false;
  }
  if (bucketListShape[0] < 1) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "bucket list", "The shape[0] of bucket list is less than 1");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] The shape[0] (%d) of bucket list is less than 1.", bucketListShape[0]);
    return false;
  }

  OP_LOGD(opType.c_str(), "op[GenADCTiling] CheckGenADCParams run success.");
  return true;
}

bool GetGenADCCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum) {
  OP_LOGD(opType.c_str(), "op[GenADCTiling] GetGenADCCompileParams begin.");

  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Failed to get core_num.");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  OP_LOGD(opType.c_str(), "op[GenADCTiling] GetGenADCCompileParams run success.");
  return true;
}

int64_t CeilDiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

void InitGenADCParams(GenADCParams& params) {
  params.rowNumEachCore = 0;
  params.remainingRow = 0;
  params.bucketListBurstLen = 0;
  params.remainBucketListBurstLen = 0;
  params.coreUsedNum = 1;
  params.dimNs = 0;
}

void SetGenADCParams(const GenADCParams& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.rowNumEachCore);
  ByteBufferPut(runInfo.tiling_data, runParams.remainingRow);
  ByteBufferPut(runInfo.tiling_data, runParams.bucketListBurstLen);
  ByteBufferPut(runInfo.tiling_data, runParams.remainBucketListBurstLen);
  ByteBufferPut(runInfo.tiling_data, runParams.coreUsedNum);
  ByteBufferPut(runInfo.tiling_data, runParams.dimNs);
}

void PrintGenADCParams(const GenADCParams& runParams) {
  OP_LOGD("GenADC", "op [GenADCTiling] : rowNumEachCore=%d.", runParams.rowNumEachCore);
  OP_LOGD("GenADC", "op [GenADCTiling] : remainingRow=%d.", runParams.remainingRow);
  OP_LOGD("GenADC", "op [GenADCTiling] : bucketListBurstLen=%d.", runParams.bucketListBurstLen);
  OP_LOGD("GenADC", "op [GenADCTiling] : remainBucketListBurstLen=%d.", runParams.remainBucketListBurstLen);
  OP_LOGD("GenADC", "op [GenADCTiling] : coreUsedNum=%d.", runParams.coreUsedNum);
  OP_LOGD("GenADC", "op [GenADCTiling] : dimNs=%d.", runParams.dimNs);
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] opInfo: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool GenADCTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opInfo,
                  OpRunInfo& runInfo) {
  OP_LOGD(opType.c_str(), "op[GenADCTiling] tiling run begin.");

  bool checkResult = CheckGenADCParams(opType, opParas, opInfo);
  if (!checkResult) {
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Failed to check input params.");
    return false;
  }

  int64_t coreNum = 0;
  if (!GetGenADCCompileParams(opType, opInfo, coreNum)) {
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Failed to get parameters from compile info.");
    return false;
  }
  if (coreNum < 1) {
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Core num is less than 1.");
    return false;
  }
  OP_LOGD("GenADC", "op [GenADCTiling] : coreNum=%d.", coreNum);

  GenADCParams runParams;
  InitGenADCParams(runParams);

  std::vector<int64_t> bucketListShape = opParas.inputs[3].tensor[0].shape;
  int64_t totalBuckets = std::accumulate(bucketListShape.begin(), bucketListShape.end(), 1, std::multiplies<int64_t>());

  runParams.rowNumEachCore = CeilDiv(totalBuckets, coreNum);
  runParams.coreUsedNum = CeilDiv(totalBuckets, runParams.rowNumEachCore);
  runParams.remainingRow = totalBuckets - (runParams.rowNumEachCore * (runParams.coreUsedNum - 1));

  int64_t eleSize = sizeof(float);
  const std::string bucketListDtype = opParas.inputs[3].tensor[0].dtype;
  if (BUCKET_LIST_DTYPES.find(bucketListDtype) != BUCKET_LIST_DTYPES.end()) {
    eleSize = BUCKET_LIST_DTYPES.find(bucketListDtype)->second;
  }

  int64_t dataEachBlock = BLOCK_SIZE / eleSize;
  if (dataEachBlock < 1) {
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Size of the data type (%s) is larger than block size (%d).",
            bucketListDtype.c_str(), BLOCK_SIZE);
    return false;
  }

  runParams.bucketListBurstLen = CeilDiv(runParams.rowNumEachCore, dataEachBlock);
  runParams.remainBucketListBurstLen = CeilDiv(runParams.remainingRow, dataEachBlock);

  runParams.dimNs = bucketListShape[0];

  SetGenADCParams(runParams, runInfo);
  runInfo.block_dim = coreNum;

  PrintGenADCParams(runParams);

  OP_LOGD(opType.c_str(), "op[GenADCTiling] tiling run success.");
  return true;
}

// register tiling interface of the GenADC op.
REGISTER_OP_TILING_FUNC_BUFFERED(GenADC, GenADCTiling);
}  // namespace optiling
