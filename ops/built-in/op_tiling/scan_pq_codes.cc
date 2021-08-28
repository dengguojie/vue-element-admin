#include <nlohmann/json.hpp>
#include <string>
#include <algorithm>
#include <vector>
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"
#include <iostream>

namespace optiling {
struct ScanPQCodesTilingParams {
  int64_t bucketNumTotal;
  int64_t bucketNumPerCore;
  int64_t bucketNumLeft;
  int64_t coreUsedNum;
};

void ScanPQCodesWriteTilingParams(const ScanPQCodesTilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.bucketNumTotal);
  ByteBufferPut(run_info.tiling_data, params.bucketNumPerCore);
  ByteBufferPut(run_info.tiling_data, params.bucketNumLeft);
  ByteBufferPut(run_info.tiling_data, params.coreUsedNum);
}

void ScanPQCodesPrintTilingParams(const std::string& opType, const ScanPQCodesTilingParams& params) {
  OP_LOGD("op [%s] : params.coreUsedNum=%d", opType.c_str(), params.coreUsedNum);
}

int64_t CeilDivPQ(int64_t dividend, int64_t divisor) {
  return (dividend + divisor - 1) / divisor;
}

bool ScanPQCodesTiling(const std::string& opType, const TeOpParas& op_paras, const nlohmann::json& op_compile_info_json,
                       OpRunInfo& run_info) {
  OP_LOGI("========================ScanPQCodesTiling running.====================");
  if (op_paras.inputs.size() < 2) {
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
  int32_t coreNum = allVars["core_nums"].get<std::int32_t>();
  if (coreNum == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, coreNum is 0.");
    return false;
  }
  int64_t bucketNumTotal = bucketShape[0];
  int32_t bucketNumPerCore = CeilDivPQ(bucketNumTotal, coreNum);
  int32_t bucketNumLeft = bucketNumTotal % bucketNumPerCore;
  int32_t coreUsedNum = CeilDivPQ(bucketNumTotal, bucketNumPerCore);
  ScanPQCodesTilingParams params{bucketNumTotal, bucketNumPerCore, bucketNumLeft, coreUsedNum};
  ScanPQCodesWriteTilingParams(params, run_info);
  ScanPQCodesPrintTilingParams(opType, params);
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(ScanPQCodes, ScanPQCodesTiling);
}  // namespace optiling
