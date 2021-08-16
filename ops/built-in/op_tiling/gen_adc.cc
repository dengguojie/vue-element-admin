/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

namespace optiling {

const int64_t CORE_MINIMUM_NUM = 2;  // the minimum num for one core
const int64_t BLOCK_SIZE = 32;  // one block size is 32Bytes

const map<string, int64_t> BUCKET_LIST_DTYPES = {
    {"int32", sizeof(int32_t)}, {"int8", sizeof(int8_t)}, {"int16", sizeof(int16_t)}, {"int64", sizeof(int64_t)}};

bool GetCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum) {
  OP_LOGD(opType.c_str(), "op[GenADCTiling] GetCompileParams begin.");
  using namespace nlohmann;

  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Failed to get core_num.");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  OP_LOGD(opType.c_str(), "op[GenADCTiling] GetCompileParams run success.");
  return true;
}

int64_t CeilDiv(int64_t dividend, int64_t divisor) {
  return (dividend + divisor - 1) / divisor;
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

  if (opInfo == nullptr) {
    OP_LOGE(opType.c_str(), "op[GenADCTiling] opInfo json error.");
    return false;
  }
  if (opParas.inputs.size() < 4 || opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() ||
      opParas.inputs[2].tensor.empty() || opParas.inputs[3].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices",
                                  "The length of inputs is less than 4 or the inputs is empty");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] input parameters missing.");
    return false;
  }
  if (opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "adc_tables",
                                   "The length of outputs is less than 1 or the outputs is empty");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] output parameters missing.");
    return false;
  }

  std::vector<int64_t> bucketListShape = opParas.inputs[3].tensor[0].shape;
  if (bucketListShape.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "bucket list", "The shape of bucket list is empty");
    OP_LOGE(opType.c_str(), "op[GenADCTiling] bucket list shape error.");
    return false;
  }
  int64_t dimNs = bucketListShape[0];

  const std::string bucketListDtype = opParas.inputs[3].tensor[0].dtype;

  int64_t coreNum = 0;
  if (!GetCompileParams(opType, opInfo, coreNum)) {
    OP_LOGE(opType.c_str(), "op[GenADCTiling] Failed to get parameters from compile info.");
    return false;
  }

  int64_t valueNum = std::accumulate(bucketListShape.begin(), bucketListShape.end(), 1, std::multiplies<int64_t>());

  int64_t sigmentTotal = CeilDiv(valueNum, CORE_MINIMUM_NUM);
  int64_t sigmentPerCore = CeilDiv(sigmentTotal, coreNum);

  int64_t coreUsedNum = sigmentPerCore == 0 ? 1 : CeilDiv(sigmentTotal, sigmentPerCore);
  int64_t rowNumEachCore = sigmentPerCore * CORE_MINIMUM_NUM;
  int64_t remainingRow = valueNum - (rowNumEachCore * (coreUsedNum - 1));

  int64_t eleSize = sizeof(float);
  if (BUCKET_LIST_DTYPES.find(bucketListDtype) != BUCKET_LIST_DTYPES.end()) {
    eleSize = BUCKET_LIST_DTYPES.find(bucketListDtype)->second;
  }

  int64_t dataEachBlock = BLOCK_SIZE / eleSize;
  int64_t bucketListBurstLen = CeilDiv(rowNumEachCore, dataEachBlock);
  int64_t remainBucketListBurstLen = CeilDiv(remainingRow, dataEachBlock);

  ByteBufferPut(runInfo.tiling_data, rowNumEachCore);
  ByteBufferPut(runInfo.tiling_data, remainingRow);
  ByteBufferPut(runInfo.tiling_data, bucketListBurstLen);
  ByteBufferPut(runInfo.tiling_data, remainBucketListBurstLen);
  ByteBufferPut(runInfo.tiling_data, coreUsedNum);
  ByteBufferPut(runInfo.tiling_data, dimNs);

  runInfo.block_dim = coreNum;

  OP_LOGD("GenADC", "op [GenADCTiling] : rowNumEachCore=%d.", rowNumEachCore);
  OP_LOGD("GenADC", "op [GenADCTiling] : remainingRow=%d.", remainingRow);
  OP_LOGD("GenADC", "op [GenADCTiling] : bucketListBurstLen=%d.", bucketListBurstLen);
  OP_LOGD("GenADC", "op [GenADCTiling] : remainBucketListBurstLen=%d.", remainBucketListBurstLen);
  OP_LOGD("GenADC", "op [GenADCTiling] : coreUsedNum=%d.", coreUsedNum);
  OP_LOGD("GenADC", "op [GenADCTiling] : dimNs=%d.", dimNs);

  OP_LOGD(opType.c_str(), "op[GenADCTiling] tiling run success.");
  return true;
}

// register tiling interface of the GenADC op.
REGISTER_OP_TILING_FUNC_BUFFERED(GenADC, GenADCTiling);

}  // namespace optiling
