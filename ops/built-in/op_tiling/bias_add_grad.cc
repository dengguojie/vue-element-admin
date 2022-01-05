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
#include "error_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
struct BiasAddGradCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

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
  std::unordered_map<char, int> zip_shape;
  std::vector<int64_t> new_shape = shape.GetDims();

  OP_LOGI("BiasAddGrad",
          "input format [%s], ori_format [%s], shape: [%s], ori_shape: [%s], "
          "and ori_format lens not compare ori_shape lens.",
          to_string(format).c_str(), to_string(ori_format).c_str(), shape.ToString().c_str(),
          ori_shape.ToString().c_str());
  if (format == ge::FORMAT_FRACTAL_Z or format == ge::FORMAT_FRACTAL_Z_3D) {
    uint64_t target_shape = 4;
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
        for (uint64_t i = 0; i < str_ori_format.size(); ++i) {
          zip_shape[str_ori_format[i]] = ori_shape.GetDim(i);
        }
        int64_t shape_h_dim = zip_shape['H'];
        int64_t shape_w_dim = zip_shape['W'];
        int64_t shape_c1_dim = shape.GetDim(0) / (shape_h_dim * shape_w_dim);
        std::vector<int64_t> tmp_shape = {shape_c1_dim, shape_h_dim, shape_w_dim};
        tmp_shape.insert(tmp_shape.end(), new_shape.begin() + 1, new_shape.end());
        if (format == ge::FORMAT_FRACTAL_Z_3D) {
          int64_t shape_d_dim = zip_shape['D'];
          shape_c1_dim = tmp_shape[0] / shape_d_dim;
          tmp_shape.insert(tmp_shape.begin(), {shape_d_dim});
          tmp_shape[1] = shape_c1_dim;
        }
        new_shape = tmp_shape;
      }
    }
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
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(BiasAddGrad, BiasAddGradTiling, ParseJsonCompileInfo, BiasAddGradCompileInfo);
}  // namespace optiling
