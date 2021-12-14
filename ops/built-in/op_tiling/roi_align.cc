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

/*!
 * \file roi_align.cpp
 * \brief tiling function of op
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"

namespace optiling {
using namespace ge;
using namespace std;
const int64_t BLOCK_SIZE = 32;

// A. block tiling: indices tiling
// 1. one params row size is smaller than 32B
// params is not cache
const int64_t TILING_MODE_1 = 1;
const int64_t HEIGHT_INDEX = 2;
const int64_t WIDTH_INDEX = 3;

struct ROIAlignTilingParams {
  int64_t tilingMode;
  int64_t real_core_num;
  int64_t rois_n;
  int64_t rois_row_lenth;
  int64_t c1_num;
  int64_t x_height;
  int64_t x_width;
};

void InitROIAlignParams(ROIAlignTilingParams& params) {
  params.tilingMode = 1;
  params.real_core_num = 1;
  params.rois_n = 1;
  params.rois_row_lenth = 5;
  params.c1_num = 1;
  params.x_height = 1;
  params.x_width = 1;
}

void SetROIAlignParams(const ROIAlignTilingParams& Params, utils::OpRunInfo& runInfo) {
  // set tiling data
  runInfo.AddTilingData(Params.tilingMode);
  runInfo.AddTilingData(Params.real_core_num);
  runInfo.AddTilingData(Params.rois_n);
  runInfo.AddTilingData(Params.rois_row_lenth);
  runInfo.AddTilingData(Params.c1_num);
  runInfo.AddTilingData(Params.x_height);
  runInfo.AddTilingData(Params.x_width);
}

void PrintROIAlignParams(const ROIAlignTilingParams& params) {
  OP_LOGD("[ROIAlignTiling]", "tilingMode=%d.", params.tilingMode);
  OP_LOGD("[ROIAlignTiling]", "real_core_num=%d.", params.real_core_num);
  OP_LOGD("[ROIAlignTiling]", "rois_n=%d.", params.rois_n);
  OP_LOGD("[ROIAlignTiling]", "rois_row_lenth=%d.", params.rois_row_lenth);
  OP_LOGD("[ROIAlignTiling]", "c1_num=%d.", params.c1_num);
  OP_LOGD("[ROIAlignTiling]", "x_height=%d.", params.x_height);
  OP_LOGD("[ROIAlignTiling]", "x_width=%d.", params.x_width);
}

static bool CheckTensorShape(const std::string& opType, GeShape& x_diff_shape, GeShape& rois_shape) {
  int64_t x_diff_shape_dims = x_diff_shape.GetDimNum();
  int64_t rois_shape_dims = rois_shape.GetDimNum();

  if (x_diff_shape_dims != 5) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [ROIAlignTiling] : CheckTensorShape, shape of x_diff check failed.");
    return false;
  }

  if (rois_shape_dims != 2) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [ROIAlignTiling] : CheckTensorShape, dims of rois must be 2.");
    return false;
  }

  return true;
}

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size"};

static void CalcBlockNum(const int64_t& core_num, const int64_t& rois_n, const int64_t& c1_num,
                         int64_t& real_core_num) {
  if ((rois_n * c1_num) > core_num) {
    real_core_num = core_num;
  } else {
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
bool ROIAlignTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                    utils::OpRunInfo& runInfo) {
  PROFILING_TILING_INIT(opType.c_str());
  OP_LOGI("op[%s] ROIAlignTiling running.", opType.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  GeShape& feature_map_shape = input_desc->MutableShape();

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  GeShape& rois_shape = input_desc->MutableShape();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  // get compile info
  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "parse op_info failed."), return false);
  int64_t core_num = op_info[0];
  int64_t ub_size = op_info[1];
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  bool flag = true;
  flag = CheckTensorShape(opType, feature_map_shape, rois_shape);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "ROIAlignTiling: params check failed.");
    return false;
  }

  ROIAlignTilingParams runParams;
  InitROIAlignParams(runParams);

  int64_t real_core_num = 1;
  CalcBlockNum(core_num, feature_map_shape.GetDim(1), rois_shape.GetDim(0), real_core_num);
  runParams.real_core_num = real_core_num;

  int64_t rois_n = rois_shape.GetDim(0);
  int64_t c1_num = feature_map_shape.GetDim(1);
  int64_t x_width = feature_map_shape.GetDim(WIDTH_INDEX);
  runParams.rois_n = rois_n;
  runParams.c1_num = c1_num;
  runParams.rois_row_lenth = rois_shape.GetDim(1);
  runParams.x_width = x_width;
  runParams.x_height = feature_map_shape.GetDim(HEIGHT_INDEX);

  if (c1_num == 16) {
    runParams.tilingMode = 1;
  } else {
    runParams.tilingMode = 2;
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  SetROIAlignParams(runParams, runInfo);
  PrintROIAlignParams(runParams);

  // block_dim, core num used in tik op
  runInfo.SetBlockDim(runParams.real_core_num);
  OP_LOGI("op[%s] tiling run success.", opType.c_str());
  PROFILING_TILING_END();

  return true;
}
// register tiling interface of the ROIAlign op.
REGISTER_OP_TILING_V3_WITH_VECTOR(ROIAlign, ROIAlignTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
