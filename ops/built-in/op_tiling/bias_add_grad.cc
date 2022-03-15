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

#include <unordered_map>
#include <algorithm>
#include "error_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
struct BiasAddGradCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  bool is_unknown_rank;
};

static constexpr size_t SHAPE_DIM_NUM_MIN = 2;
static constexpr size_t SHAPE_DIM_NUM_FRACTAL_NZ1 = 4;
static constexpr size_t SHAPE_DIM_NUM_FRACTAL_NZ2 = 5;
// FORMAT_FRACTAL_Z C1HWNiNoC0
static constexpr size_t SHAPE_DIM_NUM_FRACTAL_Z = 6;
// FORMAT_FRACTAL_Z_3D DC1HWNiNoC0
static constexpr size_t SHAPE_DIM_NUM_FRACTAL_Z_3D = 7;
// FORMAT_NDC1HWC0
static constexpr size_t SHAPE_DIM_NUM_NDC1HWC0 = 6;
// FORMAT_NC1HWC0
static constexpr size_t SHAPE_DIM_NUM_NC1HWC0 = 5;
// FORMAT_ND c dim index
static constexpr int32_t FORMAT_ND_C_INDEX = -1;
// FORMAT_FRACTAL_NZ c0 index
static constexpr int32_t FORMAT_NZ_C0_INDEX = -1;
// FORMAT_FRACTAL_NZ c1 index
static constexpr int32_t FORMAT_NZ_C1_INDEX = -4;

struct format_dimnum_axis {
  ge::Format format;
  size_t dim_num;
  std::vector<int32_t> reduce_axis;
};

bool CalcShapeAndAxes(const ge::Format format, const bool data_nchw, std::vector<int64_t>& shape,
                      std::vector<int32_t>& axes) {
  size_t dim_num = shape.size();
  if (dim_num < SHAPE_DIM_NUM_MIN) {
    OP_LOGW("BiasAddGrad", "dim_num < %ud", SHAPE_DIM_NUM_MIN);
    return false;
  }

  static const format_dimnum_axis fda[] = {
    // C1HWNiNoC0
    {ge::FORMAT_FRACTAL_Z, SHAPE_DIM_NUM_FRACTAL_Z, {1, 2, 3, 4}},
    // DC1HWNiNoC0
    {ge::FORMAT_FRACTAL_Z_3D, SHAPE_DIM_NUM_FRACTAL_Z_3D, {0, 2, 3, 4, 5}},
    // NC1HWC0
    {ge::FORMAT_NC1HWC0, SHAPE_DIM_NUM_NC1HWC0, {0, 2, 3}},
    // NDC1HWC0
    {ge::FORMAT_NDC1HWC0, SHAPE_DIM_NUM_NDC1HWC0, {0, 1, 3, 4}}
  };

  for (uint32_t i = 0; i < sizeof(fda) / sizeof(fda[0]); i++) {
    if (format == fda[i].format) {
      if (dim_num != fda[i].dim_num) {
        OP_LOGW("BiasAddGrad", "dim_num != %ud", fda[i].dim_num);
        return false;
      }
      axes = fda[i].reduce_axis;
      return true;
    }
  }
  std::vector<int32_t> c_axes;
  if (format == ge::FORMAT_FRACTAL_NZ) {
    if (data_nchw) {
      if (dim_num == SHAPE_DIM_NUM_FRACTAL_NZ1) {
        axes = {1, 2};
      } else if (dim_num == SHAPE_DIM_NUM_FRACTAL_NZ2) {
        axes = {0, 1, 4};
      } else {
        c_axes.push_back(1);
      }
    } else {
      if (dim_num < SHAPE_DIM_NUM_FRACTAL_NZ1) {
        OP_LOGW("BiasAddGrad", "dim_num < %ud", SHAPE_DIM_NUM_FRACTAL_NZ1);
        return false;
      }
      c_axes.push_back(dim_num + FORMAT_NZ_C0_INDEX);
      c_axes.push_back(dim_num + FORMAT_NZ_C1_INDEX);
    }
  } else {
    // ND
    if (data_nchw) {
      c_axes.push_back(1);
    } else {
      c_axes.push_back(dim_num + FORMAT_ND_C_INDEX);
    }
  }
  if (c_axes.size() > 0) {
    std::vector<int32_t>::iterator it;
    for (int32_t i = 0; i < static_cast<int32_t>(dim_num); i++) {
      it = find(c_axes.begin(), c_axes.end(), i);
      if (it == c_axes.end()) {
        axes.push_back(i);
      }
    }
  }
  return true;
}

bool BiasAddGradTiling(const std::string& op_type, const ge::Operator& op_paras,
                       const BiasAddGradCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input0 desc failed"),
                  return false);
  GeShape& shape = input_desc->MutableShape();
  GeShape ori_shape = input_desc->GetOriginShape();
  ge::Format format = input_desc->GetFormat();
  ge::Format ori_format = input_desc->GetOriginFormat();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  std::vector<int64_t> new_shape = shape.GetDims();

  OP_LOGI("BiasAddGrad",
          "input format [%s], ori_format [%s], shape: [%s], ori_shape: [%s], "
          "and ori_format lens not compare ori_shape lens.",
          to_string(format).c_str(), to_string(ori_format).c_str(), shape.ToString().c_str(),
          ori_shape.ToString().c_str());
  if (format == ge::FORMAT_FRACTAL_Z or format == ge::FORMAT_FRACTAL_Z_3D) {
    static const uint64_t target_shape = 4;
    if (shape.GetDimNum() == target_shape) {
      std::string str_ori_format = to_string(ori_format);
      if (str_ori_format.size() != ori_shape.GetDimNum()) {
        OP_LOGD("BiasAddGrad",
                "input format [%s], ori_format [%s], shape: [%s], ori_shape: [%s], "
                "and ori_format lens not compare ori_shape lens.",
                to_string(format).c_str(), str_ori_format.c_str(), shape.ToString().c_str(),
                ori_shape.ToString().c_str());
        return false;
      } else {
        std::unordered_map<char, int> zip_shape;
        for (uint64_t i = 0; i < str_ori_format.size(); ++i) {
          zip_shape[str_ori_format[i]] = ori_shape.GetDim(i);
        }
        OP_TILING_CHECK((zip_shape.count('H') < 1) || (zip_shape.count('W') < 1),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ori_format has no height or width, error!"),
                        return false);
        int64_t shape_h_dim = zip_shape['H'];
        int64_t shape_w_dim = zip_shape['W'];
        OP_TILING_CHECK((shape_h_dim <= 0) || (shape_w_dim <= 0),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "shape height or width error!"),
                        return false);
        int64_t shape_c1_dim = shape.GetDim(0) / (shape_h_dim * shape_w_dim);
        std::vector<int64_t> tmp_shape = {shape_c1_dim, shape_h_dim, shape_w_dim};
        tmp_shape.insert(tmp_shape.end(), new_shape.begin() + 1, new_shape.end());
        if ((format == ge::FORMAT_FRACTAL_Z_3D) && (zip_shape.count('D') > 0)) {
          int64_t shape_d_dim = zip_shape['D'];
          OP_TILING_CHECK((shape_d_dim <= 0),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "shape d dim error!"),
                          return false);
          shape_c1_dim = tmp_shape[0] / shape_d_dim;
          tmp_shape.insert(tmp_shape.begin(), {shape_d_dim});
          tmp_shape[1] = shape_c1_dim;
        }
        new_shape = tmp_shape;
      }
    }
  }
  OP_LOGI("BiasAddGrad", "is_unknown_rank : [%d]", parsed_info.is_unknown_rank);
  if (parsed_info.is_unknown_rank) {
    std::vector<int32_t> reduce_axis;
    OP_TILING_CHECK(!CalcShapeAndAxes(format, (ori_format == ge::FORMAT_NCHW), new_shape, reduce_axis),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "is_unknown_rank CalcShapeAndAxes failed."),
                    return false);
    PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
    std::vector<std::vector<int64_t>> shapes = {new_shape};
    ge::DataType type = input_desc->GetDataType();
    std::vector<std::vector<int32_t>> axes{reduce_axis};
    OpInfo eletwise_info(shapes, type, axes);
    PROFILING_TILING_AFTER_CALCU_TILING_REG();

    OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                    return false);
    bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
    PROFILING_TILING_END();
    return ret;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  std::vector<std::vector<int64_t>> shapes = {new_shape};
  ge::DataType type = input_desc->GetDataType();
  OpInfo eletwise_info(shapes, type);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                  return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BiasAddGradCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_REDUCE, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  // get parsed_info.is_unknown_rank value
  parsed_info.is_unknown_rank = false;
  GetCompileValue(compile_info, "is_unknown_rank", parsed_info.is_unknown_rank, false);
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(BiasAddGrad, BiasAddGradTiling, ParseJsonCompileInfo, BiasAddGradCompileInfo);
}  // namespace optiling
