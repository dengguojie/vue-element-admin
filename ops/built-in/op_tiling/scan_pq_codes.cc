#include <nlohmann/json.hpp>
#include <string>
#include <algorithm>
#include <vector>
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
struct ScanPQCodesTilingParams{
  int64_t bucketNumTotal;
  int64_t bucketNumPerCore;
  int64_t bucketNumLeft;
  int64_t coreUsedNum;
  int64_t bucketStartBase;
};

void ScanPQCodesWriteTilingParams(const ScanPQCodesTilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.bucketNumTotal);
  ByteBufferPut(run_info.tiling_data, params.bucketNumPerCore);
  ByteBufferPut(run_info.tiling_data, params.bucketNumLeft);
  ByteBufferPut(run_info.tiling_data, params.coreUsedNum);
  ByteBufferPut(run_info.tiling_data, params.bucketStartBase);
}

void ScanPQCodesPrintTilingParams(const std::string& opType, const ScanPQCodesTilingParams& params) {
  OP_LOGD("op [ScanPQCodes] : params.bucketNumTotal=%d", params.bucketNumTotal);
  OP_LOGD("op [ScanPQCodes] : params.bucketNumPerCore=%d", params.bucketNumPerCore);
  OP_LOGD("op [ScanPQCodes] : params.bucketNumLeft=%d", params.bucketNumLeft);
  OP_LOGD("op [ScanPQCodes] : params.coreUsedNum=%d", params.coreUsedNum);
  OP_LOGD("op [ScanPQCodes] : params.coreUsedNum=%d", params.bucketStartBase);
}

int64_t ScanPQCodesCeilDiv(int64_t dividend, int64_t divisor) {
  return (dividend + divisor - 1) / divisor;
}

bool ScanPQCodesTiling(const std::string& opType, const TeOpParas& op_paras, const nlohmann::json& op_compile_info_json,
                       OpRunInfo& run_info) {
  OP_LOGI("========================ScanPQCodesTiling running.====================");
  if (op_paras.inputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.inputs cannot be empty");
    return false;
  }
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
  int64_t bucketNumPerCore = 0;
  int64_t bucketNumLeft = 0;
  int64_t coreUsedNum = 0;
  int64_t bucketStartBase = 0;
  if (splitCount == 1) {
    bucketNumTotal = bucketShape[0];
  } else {
    bucketNumTotal = (splitIndex < splitCount - 1) ? bucketShape[0] / splitCount :
                      bucketShape[0] - (bucketShape[0] / splitCount);
  }
  bucketNumPerCore = ScanPQCodesCeilDiv(bucketNumTotal, coreNums);
  bucketNumLeft = bucketNumTotal % bucketNumPerCore;
  coreUsedNum = ScanPQCodesCeilDiv(bucketNumTotal, bucketNumPerCore);
  bucketStartBase = bucketNumTotal * splitIndex;
  ScanPQCodesTilingParams params{bucketNumTotal, bucketNumPerCore, bucketNumLeft, coreUsedNum, bucketStartBase};
  ScanPQCodesWriteTilingParams(params, run_info);
  ScanPQCodesPrintTilingParams(opType, params);
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(ScanPQCodes, ScanPQCodesTiling);
}
