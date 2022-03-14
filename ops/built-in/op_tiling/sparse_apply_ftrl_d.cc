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
 * \file sparse_apply_ftrl_d.cc
 * \brief dynamic SparseApplyFtrl op tiling
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_log.h"
#include "error_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "../op_proto/util/error_util.h"

namespace optiling {
const int32_t BLOCK_SIZE = 32;
const int32_t VECTOR_SIZE = 256;
// The 4KB space of UB is used to store indices data
const int32_t UB_INDICES_SIZE = 4 * 1024;
const int32_t UB_2K_SIZE = 2 * 1024;

// one row size of var is 32B aligned
const int32_t TILING_MODE_1 = 1;
// one row size of var is smaller than 32B
const int32_t TILING_MODE_2 = 2;
// indices num is smaller than (coreNum/2), and one row size of var is 32B aligned and large than 1024 elements
const int32_t TILING_MODE_3 = 3;
const int32_t VAR_SHAPE_POSITION = 0;
const int32_t ACCUM_SHAPE_POSITION = 1;
const int32_t LINEAR_SHAPE_POSITION = 2;
const int32_t GRAD_SHAPE_POSITION = 3;
const int32_t INDICES_SHAPE_POSITION = 4;

constexpr int32_t OP_INPUT_TENSOR_SIZE = 5;

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "indices_dsize"};

struct SparseApplyFtrldTilingParams {
  int32_t tilingMode;
  int32_t needCoreNum;
  int32_t tailProcessCore;
  int32_t indicesNumEachCore;
  int32_t indicesNumRemaining;
  int32_t indicesLoopNum;
  int32_t indicesNumLast;
  int32_t varRowElem;
  int32_t varRows;
  int32_t indicesStep;
  int32_t numMultiRows;
  int32_t partialFactor;
  int32_t elemsPerCore;
  int32_t elemsLastCore;
  int32_t elemsCoreLoop;
  int32_t elemsCoreRemain;
  int32_t elemsLastCoreLoop;
  int32_t elemsLastCoreRemain;
};

void InitRunningParams(SparseApplyFtrldTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.needCoreNum = 0;
  params.tailProcessCore = 0;
  params.indicesNumEachCore = 0;
  params.indicesNumRemaining = 0;
  params.indicesLoopNum = 0;
  params.indicesNumLast = 0;
  params.varRowElem = 0;
  params.varRows = 0;
  params.indicesStep = 0;
  params.numMultiRows = 32;
  params.partialFactor = 0;
  params.elemsPerCore = 0;
  params.elemsLastCore = 0;
  params.elemsCoreLoop = 0;
  params.elemsCoreRemain = 0;
  params.elemsLastCoreLoop = 0;
  params.elemsLastCoreRemain = 0;
}

bool CalculationTilingData(const std::string& opType, const int32_t& coreNum, const int32_t& varElemBlock,
                           const int32_t& varRowElem, const int32_t& varRows, const int32_t& indicesNums,
                           const int32_t& onePartElem, SparseApplyFtrldTilingParams& runParams) {
  if (varRowElem < varElemBlock) {
    runParams.tilingMode = TILING_MODE_2;

    runParams.indicesNumEachCore = indicesNums;
    runParams.indicesNumRemaining = 0;

    if (varRows < runParams.numMultiRows) {
      runParams.numMultiRows = varRows;
    }

    if (runParams.numMultiRows == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "op SparseApplyFtrlTiling: numMultiRows is 0.");
      return false;
    }
    runParams.needCoreNum = varRows / runParams.numMultiRows;
    if (runParams.needCoreNum > coreNum) {
      runParams.needCoreNum = coreNum;
    }
    if (runParams.needCoreNum <= 0) {
      runParams.needCoreNum = 1;
    }
    runParams.indicesStep = varRows / runParams.needCoreNum;
  } else if ((varRowElem >= varElemBlock) && (varRowElem % varElemBlock == 0)) {
    if (indicesNums * 2 < coreNum && varRowElem >= 1024) {  // 1024 = 32*32
      runParams.tilingMode = TILING_MODE_3;
      runParams.indicesNumEachCore = indicesNums;
      runParams.indicesNumRemaining = 0;
      runParams.needCoreNum = indicesNums;
      runParams.partialFactor = coreNum / runParams.needCoreNum;

      if (runParams.partialFactor == 0 || onePartElem == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op SparseApplyFtrlTiling: partialFactor or onePartElem is 0.");
        return false;
      }
      runParams.needCoreNum = runParams.needCoreNum * runParams.partialFactor;
      runParams.elemsPerCore = varRowElem / runParams.partialFactor;
      runParams.elemsLastCore = varRowElem - (runParams.partialFactor - 1) * runParams.elemsPerCore;
      runParams.elemsCoreLoop = runParams.elemsPerCore / onePartElem;
      runParams.elemsCoreRemain = runParams.elemsPerCore % onePartElem;
      runParams.elemsLastCoreLoop = runParams.elemsLastCore / onePartElem;
      runParams.elemsLastCoreRemain = runParams.elemsLastCore % onePartElem;
    } else {
      runParams.tilingMode = TILING_MODE_1;
      if (runParams.needCoreNum == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op SparseApplyFtrlTiling: needCoreNum is 0.");
        return false;
      }
      runParams.indicesNumEachCore = indicesNums / runParams.needCoreNum;
      runParams.indicesNumRemaining = indicesNums % runParams.needCoreNum;
      if (indicesNums <= runParams.needCoreNum) {
        runParams.needCoreNum = indicesNums;
        runParams.tailProcessCore = 0;
        runParams.indicesNumEachCore = 1;
        runParams.indicesNumRemaining = 0;
      }
    }
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op SparseApplyFtrlTiling: inputs var row elements is not 32B aligned.");
    return false;
  }
  return true;
}

void SetRuningParams(const SparseApplyFtrldTilingParams& params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(params.tilingMode);
  run_info.AddTilingData(params.needCoreNum);
  run_info.AddTilingData(params.tailProcessCore);
  run_info.AddTilingData(params.indicesNumEachCore);
  run_info.AddTilingData(params.indicesNumRemaining);
  run_info.AddTilingData(params.indicesLoopNum);
  run_info.AddTilingData(params.indicesNumLast);
  run_info.AddTilingData(params.varRowElem);
  run_info.AddTilingData(params.varRows);
  run_info.AddTilingData(params.indicesStep);
  run_info.AddTilingData(params.numMultiRows);
  run_info.AddTilingData(params.partialFactor);
  run_info.AddTilingData(params.elemsPerCore);
  run_info.AddTilingData(params.elemsLastCore);
  run_info.AddTilingData(params.elemsCoreLoop);
  run_info.AddTilingData(params.elemsCoreRemain);
  run_info.AddTilingData(params.elemsLastCoreLoop);
  run_info.AddTilingData(params.elemsLastCoreRemain);
}

bool PrepareTilingData(const std::string& opType, const int32_t& coreNum, const int32_t& indicesNums,
                       const int32_t& varRowElem, const int32_t& varRows, const int32_t& varElemBlock,
                       const int32_t& onePartElem, int32_t& needCoreNum, const int32_t& ubIndicesNum,
                       utils::OpRunInfo& run_info) {
  SparseApplyFtrldTilingParams runParams;
  InitRunningParams(runParams);

  runParams.varRowElem = varRowElem;
  runParams.varRows = varRows;
  runParams.needCoreNum = needCoreNum;
  bool calculation_status =
      CalculationTilingData(opType, coreNum, varElemBlock, varRowElem, varRows, indicesNums, onePartElem, runParams);
  if (!calculation_status) {
    return false;
  }
  // useless in TILING_MODE_3, because indices nums is smaller than core nums
  if (ubIndicesNum == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SparseApplyFtrlTiling: inputs ubIndicesNum is 0.");
    return false;
  }
  runParams.indicesLoopNum = runParams.indicesNumEachCore / ubIndicesNum;
  int32_t indices_nums_once = ubIndicesNum;
  if (runParams.indicesNumEachCore % indices_nums_once != 0) {
    runParams.indicesNumLast = runParams.indicesNumEachCore % indices_nums_once;
  }
  GELOGD("op [SparseApplyFtrlTiling] : indicesNumEachCore=%d, varRowElem=%d, indicesNums=%d",
         runParams.indicesNumEachCore, varRowElem, indicesNums);

  needCoreNum = runParams.needCoreNum;
  // set tiling data
  SetRuningParams(runParams, run_info);

  return true;
}

bool ParamCheck(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                int32_t& varRowElem) {
  if (op_info.size() != COMPILE_INFO_KEY.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SparseApplyFtrlTiling: op_info json error.");
    return false;
  }

  auto operator_info = OpDescUtils::GetOpDescFromOperator(opParas);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed.");
    return false;
  }

  if (operator_info->GetInputsSize() < OP_INPUT_TENSOR_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info GetInputsSize check failed.");
    return false;
  }

  auto input_var_dec = operator_info->MutableInputDesc(VAR_SHAPE_POSITION);
  auto input_accum_dec = operator_info->MutableInputDesc(ACCUM_SHAPE_POSITION);
  auto input_linear_dec = operator_info->MutableInputDesc(LINEAR_SHAPE_POSITION);
  auto input_grad_dec = operator_info->MutableInputDesc(GRAD_SHAPE_POSITION);
  auto input_indices_dec = operator_info->MutableInputDesc(INDICES_SHAPE_POSITION);
  if (input_var_dec == nullptr ||
      input_accum_dec == nullptr ||
      input_linear_dec == nullptr ||
      input_grad_dec == nullptr ||
      input_indices_dec == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SparseApplyFtrlTiling: input shape error.");
    return false;
  }

  // check inputs shape
  const ge::GeShape& var_shape = input_var_dec->MutableShape();
  const ge::GeShape& accum_shape = input_accum_dec->MutableShape();
  const ge::GeShape& linear_shape = input_linear_dec->MutableShape();
  const ge::GeShape& grad_shape = input_grad_dec->MutableShape();
  const ge::GeShape& indice_shape = input_indices_dec->MutableShape();

  if (indice_shape.GetDim(0) != grad_shape.GetDim(0)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                    "op [SparseApplyFtrlTiling] : "
                                    "grad shape[0] must be equal to indices shape[0]");
    return false;
  }

  uint32_t var_dims = var_shape.GetDimNum();
  if (accum_shape.GetDimNum() < var_dims ||
      linear_shape.GetDimNum() < var_dims ||
      grad_shape.GetDimNum() < var_dims) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                    "op [SparseApplyFtrlTiling] : "
                                    "grad shape[0] must be equal to indices shape[0]");
    return false;
  }

  for (uint32_t i = 0; i < var_dims; i++) {
    int64_t var_dim = var_shape.GetDim(i);
    if (var_dim != accum_shape.GetDim(i) || var_dim != linear_shape.GetDim(i)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                      "op [SparseApplyFtrlTiling] : "
                                      "accum and linear shape must be equal to var shape");
      return false;
    }
    if (i > 0) {
      if (var_dim != grad_shape.GetDim(i)) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [SparseApplyFtrlTiling] : grad shape is invalid");
        return false;
      }
      varRowElem *= var_dim;
    }
  }

  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool SparseApplyFtrlDTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                            utils::OpRunInfo& run_info) {
  OP_LOGD(opType, "tiling running.");

  int32_t var_row_elem = 1;

  bool paramCheckResult = ParamCheck(opType, opParas, op_info, var_row_elem);
  if (!paramCheckResult) {
    return false;
  }
  auto operator_info = OpDescUtils::GetOpDescFromOperator(opParas);
  const ge::GeShape& var_shape = operator_info->MutableInputDesc(VAR_SHAPE_POSITION)->MutableShape();
  const ge::GeShape& indices_shape = operator_info->MutableInputDesc(INDICES_SHAPE_POSITION)->MutableShape();

  int32_t var_rows = var_shape.GetDim(0);
  int32_t indices_nums = indices_shape.GetDim(0);

  int32_t var_d_size = 4;  // only support float32
  const int32_t CO_EXSIT_PART = 6;
  int32_t var_elem_block = BLOCK_SIZE / var_d_size;
  int32_t var_elem_vector = VECTOR_SIZE / var_d_size;

  // get compile info
  int32_t core_num = static_cast<int32_t>(op_info[0]);
  int32_t ub_size = static_cast<int32_t>(op_info[1]);
  int32_t indices_d_size = static_cast<int32_t>(op_info[2]);

  if (indices_d_size == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SparseApplyFtrlTiling: inputs indicesDSize is 0.");
    return false;
  }

  int32_t ub_indices_num = UB_INDICES_SIZE / indices_d_size;
  int32_t remain_ub_size = ub_size - UB_2K_SIZE - UB_INDICES_SIZE;
  int32_t one_part_ub_size = remain_ub_size / CO_EXSIT_PART;
  int32_t one_part_elem = one_part_ub_size / var_d_size;
  one_part_elem = one_part_elem - one_part_elem % var_elem_vector;

  if (var_row_elem > one_part_elem) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                    "SparseApplyFtrlTiling: "
                                    "inputs var row elements is too large, is not support yet.");
    return false;
  }

  int32_t need_core_num = core_num;
  bool prepare_status = PrepareTilingData(opType, core_num, indices_nums, var_row_elem, var_rows, var_elem_block, one_part_elem,
                                          need_core_num, ub_indices_num, run_info);
  if (!prepare_status) {
    return false;
  }

  // block_dim, core num used in tik op
  run_info.SetBlockDim(need_core_num);
  OP_LOGD(opType, "op tiling run success.");

  return true;
}

// register tiling interface of the SparseApplyFtrlD op
REGISTER_OP_TILING_V3_WITH_VECTOR(SparseApplyFtrlD, SparseApplyFtrlDTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
