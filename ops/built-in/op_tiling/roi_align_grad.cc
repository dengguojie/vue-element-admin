/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file roi_align_grad.cpp
 * \brief tiling function of op
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"

namespace optiling {
using namespace ge;
using namespace std;
const int64_t BLOCK_SIZE = 32;

// A. block tiling: indices tiling
// 1. one params row size is smaller than 32B
// params is not cache
const int64_t TILING_MODE_1 = 1;

struct ROIAlignGradTilingParams {
  int64_t tilingMode;
  int64_t real_core_num;
  int64_t rois_n;
  int64_t rois_row_lenth;
  int64_t c1_num;
  int64_t x_height;
  int64_t x_width;
};

void InitROIAlignGradParams(ROIAlignGradTilingParams& params) {
  params.tilingMode = 1;
  params.real_core_num = 1;
  params.rois_n = 1;
  params.rois_row_lenth = 5;
  params.c1_num = 1;
  params.x_height = 1;
  params.x_width = 1;
}

void SetROIAlignGradParams(ROIAlignGradTilingParams& Params, OpRunInfo& runInfo) {
  // set tiling data
  ByteBufferPut(runInfo.tiling_data, Params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, Params.real_core_num);
  ByteBufferPut(runInfo.tiling_data, Params.rois_n);
  ByteBufferPut(runInfo.tiling_data, Params.rois_row_lenth);
  ByteBufferPut(runInfo.tiling_data, Params.c1_num);
  ByteBufferPut(runInfo.tiling_data, Params.x_height);
  ByteBufferPut(runInfo.tiling_data, Params.x_width);
}

void PrintROIAlignGradParams(const ROIAlignGradTilingParams& params) {
  OP_LOGD("[ROIAlignGradTiling]", "tilingMode=%d.", params.tilingMode);
  OP_LOGD("[ROIAlignGradTiling]", "real_core_num=%d.", params.real_core_num);
  OP_LOGD("[ROIAlignGradTiling]", "rois_n=%d.", params.rois_n);
  OP_LOGD("[ROIAlignGradTiling]", "rois_row_lenth=%d.", params.rois_row_lenth);
  OP_LOGD("[ROIAlignGradTiling]", "c1_num=%d.", params.c1_num);
  OP_LOGD("[ROIAlignGradTiling]", "x_height=%d.", params.x_height);
  OP_LOGD("[ROIAlignGradTiling]", "x_width=%d.", params.x_width);
}

static bool CheckTensorShape(const std::string& opType, std::vector<int64_t> y_diff_shape,
                             std::vector<int64_t> x_diff_shape, std::vector<int64_t> rois_shape) {
  int64_t y_diff_shape_dims = y_diff_shape.size();
  int64_t x_diff_shape_dims = x_diff_shape.size();
  int64_t rois_shape_dims = rois_shape.size();

  if (x_diff_shape_dims != y_diff_shape_dims || x_diff_shape_dims != 5) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [ROIAlignGradTiling] : CheckTensorShape, shape of x_diff or y_diff check failed.");
    return false;
  }

  if (rois_shape_dims != 2) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [ROIAlignGradTiling] : CheckTensorShape, dims of rois must be 2.");
    return false;
  }

  return true;
}

static bool GetCompileParams(const std::string& opType, const nlohmann::json& opCompileInfoJson,
                             int64_t& coreNum, int64_t& ubSize) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [ROIAlignGradTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [ROIAlignGradTiling] : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();

  return true;
}

static void CalcBlockNum(const int64_t& core_num, const int64_t& rois_n,
                         const int64_t& c1_num, int64_t& real_core_num) {
  if ((rois_n * c1_num) > core_num) {
    real_core_num = core_num;
  }
  else {
    real_core_num = rois_n * c1_num;
  }
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool ROIAlignGradTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                        OpRunInfo& runInfo) {
  OP_LOGI("op[%s] ROIAlignGradTiling running.", opType.c_str());
  if (op_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op ROIAlignGradTiling: op_info json error.");
    return false;
  }

  // get compile info
  int64_t ub_size = 0;
  int64_t core_num = 0;
  bool flag = GetCompileParams(opType, op_info, core_num, ub_size);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "ROIAlignGradTiling: GetCompileParams error.");
    return false;
  }

  std::vector < int64_t > y_diff_shape = opParas.inputs[0].tensor[0].shape;
  std::vector < int64_t > rois_shape = opParas.inputs[1].tensor[0].shape;
  std::vector < int64_t > x_diff_shape = opParas.outputs[0].tensor[0].shape;
  flag = true;
  flag = CheckTensorShape(opType, y_diff_shape, x_diff_shape, rois_shape);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "ROIAlignGradTiling: params check failed.");
    return false;
  }

  ROIAlignGradTilingParams runParams;
  InitROIAlignGradParams(runParams);

  int64_t real_core_num = 0;
  CalcBlockNum(core_num, y_diff_shape[1], rois_shape[0], real_core_num);
  runParams.real_core_num = real_core_num;

  int64_t rois_n = rois_shape[0];
  int64_t c1_num = y_diff_shape[1];
  int64_t x_width = x_diff_shape[3];
  runParams.rois_n = rois_n;
  runParams.c1_num = c1_num;
  runParams.rois_row_lenth = rois_shape[1];
  runParams.x_width = x_width;
  runParams.x_height = x_diff_shape[2];

  if (c1_num == 16) {
    if (x_width <= 80) {
      runParams.tilingMode = 2;
    } else {
      runParams.tilingMode = 1;
    }
  }
  else {
    runParams.tilingMode = 3;
  }

  SetROIAlignGradParams(runParams, runInfo);
  PrintROIAlignGradParams(runParams);

  // block_dim, core num used in tik op
  runInfo.block_dim = runParams.real_core_num;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}
// register tiling interface of the ROIAlignGrad op.
REGISTER_OP_TILING_FUNC_BUFFERED(ROIAlignGrad, ROIAlignGradTiling);

}  // namespace optiling
