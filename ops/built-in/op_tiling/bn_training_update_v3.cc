/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include <algorithm>
#include "error_log.h"
#include "vector_tiling.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"

namespace optiling {
  struct BNTrainingUpdateV3CompileInfo {
    std::shared_ptr<AutoTilingHandler> tiling_handler;
  };

  bool BNTrainingUpdateV3Tiling(const std::string& op_type, const ge::Operator& op_paras,
                                const BNTrainingUpdateV3CompileInfo& parsed_info, utils::OpRunInfo& run_info) {
    auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    OP_TILING_CHECK(operator_info == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr"), return false);
    auto input_x_desc = operator_info->MutableInputDesc(0);
    OP_TILING_CHECK(input_x_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_x opdesc failed"),
                    return false);

    std::vector<int64_t> shape_x = input_x_desc->MutableShape().GetDims();
    ge::Format format_x = input_x_desc->GetFormat();

    constexpr int idx_0 = 0;
    constexpr int idx_1 = 1;
    constexpr int idx_2 = 2;
    constexpr int idx_3 = 3;
    constexpr int idx_4 = 4;

    int64_t num = 1;
    float num_rec = 1.0;
    float batch_var_scalar = 0.0;
    if (format_x == FORMAT_NDC1HWC0) {
      num = shape_x[idx_0] * shape_x[idx_1] * shape_x[idx_3] * shape_x[idx_4];
      if (op_type == "INTrainingUpdateV2") {
        num = shape_x[idx_1] * shape_x[idx_3] * shape_x[idx_4];
      }
    } else {
      num = shape_x[idx_0] * shape_x[idx_2] * shape_x[idx_3];
      if (op_type == "INTrainingUpdateV2") {
        num = shape_x[idx_3] * shape_x[idx_4];
      }
    }

    num_rec = 1.0 / static_cast<float>(num);

    if (num > 1) {
      batch_var_scalar = static_cast<float>(num) / static_cast<float>(num - 1);
    }

    bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "training_update tiling failed.");
      return false;
    }

    run_info.AddTilingData(static_cast<float>(num_rec));
    run_info.AddTilingData(static_cast<float>(batch_var_scalar));
    return true;
  }

  static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                   BNTrainingUpdateV3CompileInfo & parsed_info) {
    parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
    OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);

    return true;
  }
  REGISTER_OP_TILING_V3_CUSTOM(BNTrainingUpdateV3, BNTrainingUpdateV3Tiling, ParseJsonCompileInfo,
                               BNTrainingUpdateV3CompileInfo);
  REGISTER_OP_TILING_V3_CUSTOM(INTrainingUpdateV2, BNTrainingUpdateV3Tiling, ParseJsonCompileInfo,
                               BNTrainingUpdateV3CompileInfo);
}   // namespace optiling